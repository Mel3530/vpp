def preprocess_data(X):
    # Drop columns with excessive missing values
    columns_to_drop = ["SIFT_score", "PolyPhen_2_score", "PaPI_score"]
    X_cleaned = X.drop(columns=[col for col in columns_to_drop if col in X.columns])
    return X_cleaned
