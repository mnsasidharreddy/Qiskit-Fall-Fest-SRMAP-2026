Processed dataset
make_moons_processed.csv contains the same 100 samples after a stratified 80/20 train/test split and MinMax scaling to [-1, 1]. The scaler is fitted only on the training partition, then applied to the test partition.
Columns: sample_id, feature_1_scaled, feature_2_scaled, label, split.
Note: This is for reference
