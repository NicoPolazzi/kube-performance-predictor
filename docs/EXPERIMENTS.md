# Experiments

## Interpolation

- **Data**: CSV (`performance_results_normal.csv`) collected under normal load ( CPU percentage usage in range 10%-60%)
- **Split**: holds out middle user-count values as test set; train on surrounding values
- **Purpose**: baseline — tests whether the model can predict performance for load levels that fall within the training range
- **Config**: `split_strategy: interpolation`, `train_ratio: 0.9`

## Extrapolation

- **Data**: two CSVs — normal load for training, overload (`performance_results_overload.csv`) for testing, that is when CPU percentage usage is higher than 60%
- **Split**: all normal → train, all overload → test; no overlap between train and test distributions
- **Purpose**: demonstrates the difficulty of predicting microservice performance in an unseen stress region. The model must generalize beyond its training distribution to predict behavior under system overload, which is the real-world scenario operators care about most
- **Config**: `split_strategy: extrapolation`
- **Normalization**: `fit_on_combined=True` since test data falls outside training range

## Merged

- **Data**: both CSVs concatenated into a single dataset
- **Split**: middle user-count holdout
- **Purpose**: direct control for extrapolation — uses the same split logic as interpolation but with overload data available in the training set. If the model succeeds on merged but fails on extrapolation, it proves the difficulty is specifically about predicting an unseen region, not model capacity
- **Config**: `split_strategy: merged`, `train_ratio: 0.9`
- **Normalization**: `fit_on_combined=True` since data spans both normal and overload ranges
