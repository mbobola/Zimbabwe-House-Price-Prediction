# Zimbabwe House Listing Price Estimates

**Short project report — 23 November 2026**

## Summary

This project examines asking prices from Zimbabwean house and townhouse listings and compares a structured/text Ridge model with a multimodal model that also uses listing photos. On one price-stratified hold-out test set, the multimodal model achieved lower overall error and a higher R². Both models made much larger dollar errors on expensive listings, and the multimodal model's 95th-percentile absolute error was worse than Ridge's. These estimates are exploratory; they are not valuations of completed sales.

## Data and method

The supplied `data/houses.csv` contains 2,646 listing rows and 2,646 listing IDs. After duplicate property URLs were consolidated, 2,635 rows remained; 2,618 had a positive usable asking price. The target median was **$198,000**, the mean **$312,235**, and the range **$110 to $7.5 million**. The extreme range and the minimum value warrant source-level review. Harare accounts for about **72.9%** of price-eligible listings, so the data are geographically concentrated.

The source currency field is blank in 1,897 rows. Based on the project assumption that all prices are in USD, blank currency metadata is treated as USD; no exchange-rate conversion is applied. The `images/` directory contains 2,631 photos, with each filename prefixed by a `listing_id` found in the raw CSV. The data dictionary is in `data_dictionary.csv`.

The models use `log1p(price_usd)` as the training target and a seeded, price-stratified split: **1,779 train / 315 validation / 524 test**. Preprocessing is fit on training data; the test set is reserved for final evaluation. Ridge uses standardized structured features, categorical/amenity indicators, and train-only TF-IDF text. The multimodal model adds a frozen ImageNet ResNet18 image feature and a trainable description embedding, and is early-stopped on validation loss.

## Hold-out results

All metrics are calculated in USD on the same 524 test listings.

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| Ridge baseline | $301,013 | $121,781 | 0.356 |
| Multimodal image/text model | $241,359 | $110,581 | 0.586 |

Compared with Ridge, the multimodal model reduced MAE by **9.2%**, reduced RMSE by **19.8%**, and improved R² by **0.230** on this split. RMSE is substantially above MAE for both models, indicating that a relatively small number of large errors have an outsized effect.

Residuals are defined as actual minus predicted price, so positive residuals mean underestimation. Mean residual was about **+$39,842** for Ridge and **+$9,774** for the multimodal model; the multimodal median residual was about **+$2,490**. In the highest actual-price quintile, MAE was approximately **$373,749** for Ridge and **$351,446** for the multimodal model, and both models underpredicted on average. The multimodal model's 95th-percentile absolute error (**$500,974**) exceeded Ridge's (**$436,858**), despite its better overall MAE and RMSE.

The [residual plot](modeling/residual_diagnostics.png), [residual summary](modeling/residual_diagnostics.csv), and [price-band error table](modeling/residuals_by_actual_price_band.csv) provide further diagnostics. Full metrics and individual test predictions are in `reports/modeling/`.

## Limitations and conclusion

This is one seeded random split, not repeated cross-validation or an external-market evaluation. Similar neighborhoods or collection periods can appear in both train and test data, limiting claims about geographic or future-market generalization. Asking prices can differ from realized transaction prices, and this dataset is heavily weighted toward Harare.

Eleven eligible listings lack a readable local photo. The vision branch uses the first available local photo and a frozen ImageNet encoder, not a Zimbabwe-property-trained model; the text embedding is learned from this relatively small training set. Extreme standardized area values are clipped to limit their influence, which may also compress legitimate extremes. The high-price and tail errors remain material.

The multimodal model is the stronger candidate on this hold-out set, but the improvement is not uniform across error quantiles. Before using estimates for decisions, validate with repeated or location/time-grouped splits, review extreme source values, and evaluate against verified sale prices from a broader sample.
