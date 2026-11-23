import json
import random
import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18


NUMERIC_FEATURES = ["bedrooms", "bathrooms", "floor_area_m2", "land_area_m2"]
CATEGORICAL_FEATURES = ["city", "suburb", "property_type"]
TEXT_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
PRICE_RE = re.compile(r"(?i)(?:\bUSD\b|US\$|\$)\s*[\d,]+(?:\.\d+)?")
MAX_TEXT_LENGTH = 192
MAX_VOCAB_SIZE = 20000


def _read_json_list(value):
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _amenities_to_text(value):
    return " ".join(str(item) for item in _read_json_list(value))


def _first_photo_path(value, project_root):
    paths = _read_json_list(value)
    if not paths:
        return None
    raw_path = str(paths[0]).strip()
    if not raw_path:
        return None

    path = Path(raw_path)
    if path.is_absolute() and path.is_file():
        return path
    if "images" in path.parts:
        image_index = len(path.parts) - 1 - path.parts[::-1].index("images")
        return project_root.joinpath(*path.parts[image_index:])
    return project_root / path


def _clean_description(value):
    text = "" if pd.isna(value) else str(value)
    return PRICE_RE.sub(" PRICE ", text)


def _tokenize(text):
    return TEXT_TOKEN_RE.findall(text.lower())


def _build_vocabulary(training_texts):
    counts = {}
    for text in training_texts:
        for token in _tokenize(text):
            counts[token] = counts.get(token, 0) + 1
    vocabulary = {"<pad>": 0, "<unk>": 1}
    for token, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if len(vocabulary) >= MAX_VOCAB_SIZE:
            break
        vocabulary[token] = len(vocabulary)
    return vocabulary


def _encode_texts(texts, vocabulary):
    encoded = np.zeros((len(texts), MAX_TEXT_LENGTH), dtype=np.int64)
    for row_index, text in enumerate(texts):
        tokens = _tokenize(text)[:MAX_TEXT_LENGTH]
        encoded[row_index, : len(tokens)] = [vocabulary.get(token, 1) for token in tokens]
    return encoded


def _load_data(project_root):
    source = project_root / "reports" / "eda" / "cleaned_listings.csv"
    if not source.is_file():
        raise FileNotFoundError(
            f"Missing {source}. Run notebooks/01_clean_explore.ipynb first."
        )

    listings = pd.read_csv(source)
    required = {
        "price_usd",
        "description",
        "amenities_json",
        "photo_paths_json",
        *NUMERIC_FEATURES,
        *CATEGORICAL_FEATURES,
        "listing_id",
        "property_url",
    }
    missing = sorted(required.difference(listings.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")

    listings["price_usd"] = pd.to_numeric(listings["price_usd"], errors="coerce")
    listings = listings.loc[listings["price_usd"].gt(0)].copy()
    if len(listings) < 10:
        raise ValueError(f"Only {len(listings)} listings have a positive USD price.")

    for column in NUMERIC_FEATURES:
        listings[column] = pd.to_numeric(listings[column], errors="coerce")
    for column in CATEGORICAL_FEATURES:
        listings[column] = listings[column].fillna("Unknown").astype(str)

    listings["model_text"] = listings["description"].map(_clean_description)
    listings["amenities_text"] = listings["amenities_json"].map(_amenities_to_text)
    listings["image_path"] = listings["photo_paths_json"].map(
        lambda value: _first_photo_path(value, project_root)
    )
    return listings.reset_index(drop=True)


def _make_splits(target, seed):
    indices = np.arange(len(target))
    bins = pd.qcut(target, q=min(10, len(target)), labels=False, duplicates="drop")
    stratify = bins if pd.Series(bins).value_counts().min() >= 3 else None
    train_and_validation, test = train_test_split(
        indices, test_size=0.20, random_state=seed, stratify=stratify
    )
    validation_fraction = 0.12 / 0.80
    temp_stratify = bins[train_and_validation] if stratify is not None else None
    train, validation = train_test_split(
        train_and_validation,
        test_size=validation_fraction,
        random_state=seed,
        stratify=temp_stratify,
    )
    return train, validation, test


def _make_structured_preprocessor():
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clip_extremes",
                FunctionTransformer(
                    np.clip, kw_args={"a_min": -10.0, "a_max": 10.0}
                ),
            ),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        sparse_threshold=1.0,
    )


def _as_csr(matrix):
    if sparse.issparse(matrix):
        return matrix.tocsr()
    return sparse.csr_matrix(matrix)


def _make_linear_features(listings, train_indices, structured):
    structured.fit(listings.iloc[train_indices])
    descriptions = listings["model_text"]
    amenities = listings["amenities_text"]

    text_vectorizer = TfidfVectorizer(
        max_features=3000,
        min_df=2,
        max_df=0.98,
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words="english",
    )
    amenity_vectorizer = CountVectorizer(
        max_features=300,
        min_df=2,
        binary=True,
        token_pattern=r"(?u)\b\w[\w+\-/]*\b",
    )
    text_vectorizer.fit(descriptions.iloc[train_indices])
    amenity_vectorizer.fit(amenities.iloc[train_indices])

    structured_features = structured.transform(listings)
    text_features = text_vectorizer.transform(descriptions)
    amenity_features = amenity_vectorizer.transform(amenities)
    combined = sparse.hstack(
        [
            _as_csr(structured_features),
            text_features,
            amenity_features,
        ],
        format="csr",
    )
    return combined, structured, text_vectorizer, amenity_vectorizer


class _PhotoDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        if path is None or not path.is_file():
            return index, torch.zeros(3, 224, 224), False
        try:
            with Image.open(path) as image:
                tensor = self.transform(image.convert("RGB"))
            return index, tensor, True
        except (OSError, ValueError, UnidentifiedImageError):
            return index, torch.zeros(3, 224, 224), False


def _extract_image_features(listings, batch_size=32):
    weights = ResNet18_Weights.DEFAULT
    encoder = resnet18(weights=weights)
    encoder.fc = nn.Identity()
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    image_paths = listings["image_path"].tolist()
    dataset = _PhotoDataset(image_paths, weights.transforms())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    features = np.zeros((len(listings), 512), dtype=np.float32)
    unreadable = 0
    with torch.inference_mode():
        for indices, images, readable in loader:
            batch_features = encoder(images).cpu().numpy()
            readable_np = readable.numpy()
            for offset, index in enumerate(indices.numpy()):
                if readable_np[offset]:
                    features[index] = batch_features[offset]
                else:
                    unreadable += 1
    if unreadable:
        warnings.warn(
            f"{unreadable} of {len(listings)} listings have no readable local photo; "
            "their image features are zero-filled.",
            stacklevel=2,
        )
    return features, len(listings) - unreadable


class _MultimodalRegressor(nn.Module):
    def __init__(self, structured_size, vocabulary_size, embedding_size=64):
        super().__init__()
        self.structured_branch = nn.Sequential(
            nn.Linear(structured_size, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.embedding = nn.Embedding(
            vocabulary_size, embedding_size, padding_idx=0
        )
        self.text_projection = nn.Sequential(
            nn.Linear(embedding_size, 64),
            nn.ReLU(),
        )
        self.image_projection = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.fusion = nn.Sequential(
            nn.Linear(64 + 64 + 64 + 1, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

    def forward(self, structured, tokens, images, baseline):
        token_mask = tokens.ne(0).unsqueeze(-1)
        embedded = self.embedding(tokens)
        text = (embedded * token_mask).sum(dim=1)
        text = text / token_mask.sum(dim=1).clamp(min=1)
        joined = torch.cat(
            [
                self.structured_branch(structured),
                self.text_projection(text),
                self.image_projection(images),
                baseline.unsqueeze(1),
            ],
            dim=1,
        )
        return baseline + self.fusion(joined).squeeze(1)


def _prepare_neural_inputs(listings, structured, vocabulary, image_features):
    structured_features = structured.transform(listings)
    if sparse.issparse(structured_features):
        structured_features = structured_features.toarray()
    structured_features = np.asarray(structured_features, dtype=np.float32)
    token_features = _encode_texts(listings["model_text"].tolist(), vocabulary)
    return structured_features, token_features, image_features


def _usd_predictions(log_predictions):
    return np.expm1(np.maximum(log_predictions, 0.0))


def _metric_row(name, actual_usd, predicted_log, target_log):
    predicted_usd = _usd_predictions(predicted_log)
    return {
        "model": name,
        "MAE_USD": mean_absolute_error(actual_usd, predicted_usd),
        "RMSE_USD": mean_squared_error(actual_usd, predicted_usd) ** 0.5,
        "R2_USD": r2_score(actual_usd, predicted_usd),
        "MAE_log": mean_absolute_error(target_log, predicted_log),
    }


def run_training(project_root, max_epochs=60, seed=42):
    project_root = Path(project_root).resolve()
    report_dir = project_root / "reports" / "modeling"
    model_dir = project_root / "models"
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(torch.get_num_threads(), 4))

    listings = _load_data(project_root)
    actual_usd = listings["price_usd"].to_numpy(dtype=np.float64)
    target_log = np.log1p(actual_usd).astype(np.float32)
    train_indices, validation_indices, test_indices = _make_splits(target_log, seed)

    structured = _make_structured_preprocessor()
    features, structured, text_vectorizer, amenity_vectorizer = _make_linear_features(
        listings, train_indices, structured
    )
    linear_model = Ridge(alpha=15.0, solver="lsqr")
    linear_model.fit(features[train_indices], target_log[train_indices])
    baseline_predictions = linear_model.predict(features)

    linear_path = model_dir / "linear_baseline.joblib"
    joblib.dump(
        {
            "structured_preprocessor": structured,
            "text_vectorizer": text_vectorizer,
            "amenity_vectorizer": amenity_vectorizer,
            "model": linear_model,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
        },
        linear_path,
    )

    image_features, local_image_count = _extract_image_features(listings)
    vocabulary = _build_vocabulary(listings.iloc[train_indices]["model_text"])
    structured_features, token_features, image_features = _prepare_neural_inputs(
        listings, structured, vocabulary, image_features
    )

    device = torch.device("cpu")
    model = _MultimodalRegressor(
        structured_features.shape[1], len(vocabulary)
    ).to(device)
    structured_tensor = torch.from_numpy(structured_features).to(device)
    token_tensor = torch.from_numpy(token_features).to(device)
    image_tensor = torch.from_numpy(image_features).to(device)
    baseline_tensor = torch.from_numpy(
        baseline_predictions.astype(np.float32)
    ).to(device)
    target_tensor = torch.from_numpy(target_log).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_function = nn.SmoothL1Loss()
    train_loader = DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(train_indices.astype(np.int64))
        ),
        batch_size=64,
        shuffle=True,
    )

    def validation_loss():
        model.eval()
        with torch.inference_mode():
            prediction = model(
                structured_tensor[validation_indices],
                token_tensor[validation_indices],
                image_tensor[validation_indices],
                baseline_tensor[validation_indices],
            )
            return loss_function(
                prediction, target_tensor[validation_indices]
            ).item()

    baseline_validation_loss = validation_loss()
    best_loss = baseline_validation_loss
    best_epoch = 0
    best_state = {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }
    history = [
        {
            "epoch": 0,
            "train_loss": np.nan,
            "validation_loss": baseline_validation_loss,
            "validation_mae_usd": mean_absolute_error(
                actual_usd[validation_indices],
                _usd_predictions(baseline_predictions[validation_indices]),
            ),
        }
    ]
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for (batch_indices,) in train_loader:
            batch_indices = batch_indices.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                structured_tensor[batch_indices],
                token_tensor[batch_indices],
                image_tensor[batch_indices],
                baseline_tensor[batch_indices],
            )
            loss = loss_function(prediction, target_tensor[batch_indices])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            count = len(batch_indices)
            total_loss += loss.item() * count
            seen += count

        model.eval()
        with torch.inference_mode():
            validation_prediction = model(
                structured_tensor[validation_indices],
                token_tensor[validation_indices],
                image_tensor[validation_indices],
                baseline_tensor[validation_indices],
            )
            current_validation_loss = loss_function(
                validation_prediction, target_tensor[validation_indices]
            ).item()
            validation_usd = _usd_predictions(validation_prediction.cpu().numpy())

        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / seen,
                "validation_loss": current_validation_loss,
                "validation_mae_usd": mean_absolute_error(
                    actual_usd[validation_indices], validation_usd
                ),
            }
        )
        if current_validation_loss < best_loss - 1e-5:
            best_loss = current_validation_loss
            best_epoch = epoch
            best_state = {
                name: tensor.detach().clone()
                for name, tensor in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= 8:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        multimodal_predictions = model(
            structured_tensor[test_indices],
            token_tensor[test_indices],
            image_tensor[test_indices],
            baseline_tensor[test_indices],
        ).cpu().numpy()

    rows = [
        _metric_row(
            "Linear Ridge (structured + TF-IDF)",
            actual_usd[test_indices],
            baseline_predictions[test_indices],
            target_log[test_indices],
        ),
        _metric_row(
            "Multimodal ResNet18 + text embeddings + MLP",
            actual_usd[test_indices],
            multimodal_predictions,
            target_log[test_indices],
        ),
    ]
    metrics = pd.DataFrame(rows).sort_values("MAE_USD").reset_index(drop=True)

    predictions = pd.DataFrame(
        {
            "listing_id": listings.iloc[test_indices]["listing_id"].to_numpy(),
            "property_url": listings.iloc[test_indices]["property_url"].to_numpy(),
            "actual_price_usd": actual_usd[test_indices],
            "linear_predicted_price_usd": _usd_predictions(
                baseline_predictions[test_indices]
            ),
            "multimodal_predicted_price_usd": _usd_predictions(
                multimodal_predictions
            ),
        }
    )
    history_frame = pd.DataFrame(history)

    multimodal_path = model_dir / "multimodal_fusion.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocabulary": vocabulary,
            "max_text_length": MAX_TEXT_LENGTH,
            "structured_feature_count": structured_features.shape[1],
            "seed": seed,
            "best_epoch": best_epoch,
        },
        multimodal_path,
    )
    metrics_path = report_dir / "metrics.csv"
    predictions_path = report_dir / "test_predictions.csv"
    history_path = report_dir / "training_history.csv"
    split_path = report_dir / "split_summary.json"
    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    history_frame.to_csv(history_path, index=False)
    split_path.write_text(
        json.dumps(
            {
                "rows": len(listings),
                "train": len(train_indices),
                "validation": len(validation_indices),
                "test": len(test_indices),
                "local_images": local_image_count,
                "seed": seed,
                "best_multimodal_epoch": best_epoch,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Rows: {len(listings):,}; train/validation/test: "
        f"{len(train_indices):,}/{len(validation_indices):,}/{len(test_indices):,}"
    )
    print(f"Listings with a local image: {local_image_count:,}/{len(listings):,}")
    print(f"Multimodal best epoch: {best_epoch}")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:,.4f}"))
    print(f"Reports: {report_dir}")
    print(f"Saved models: {linear_path}, {multimodal_path}")

    return {
        "metrics": metrics,
        "predictions": predictions,
        "history": history_frame,
        "paths": {
            "metrics": metrics_path,
            "predictions": predictions_path,
            "history": history_path,
            "split_summary": split_path,
            "linear_model": linear_path,
            "multimodal_model": multimodal_path,
        },
    }
