import pandas as pd
from src.preprocessing import preprocess_data
from src.feature_selection import select_features
from src.models import model_configs
from src.train_pipeline import run_full_pipeline


def main():
    print("Loading training data (Clinvitae 2017)...")
    train_df = pd.read_csv("data/clinvitae_2017.csv")
    X_train = preprocess_data(train_df.drop(columns=["label"]))
    y_train = train_df["label"]

    print("Loading internal test data (Clinvitae 2019)...")
    test_df = pd.read_csv("data/clinvitae_2019.csv")
    X_test = preprocess_data(test_df.drop(columns=["label"]))
    y_test = test_df["label"]

    print("Loading external validation data (ICR639)...")
    val_df = pd.read_csv("data/icr639.csv")
    X_val = preprocess_data(val_df.drop(columns=["label"]))
    y_val = val_df["label"]

    print("Selecting relevant features using XGBoost...")
    selected_features = select_features(X_train, y_train)

    # Apply feature selection to all datasets
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]
    X_val = X_val[selected_features]

    print("Running training pipeline for all models...\n")
    results = []
    for model_name, config in model_configs.items():
        print(f"Training model: {model_name}")
        model_results = run_full_pipeline(
            model_name, config, X_train, y_train, X_test, y_test, X_val, y_val
        )
        results.append(model_results)
        print(f"Completed: {model_name}\n")

    # Save all results
    print("Saving results to results.csv...")
    all_results_df = pd.DataFrame(results)
    all_results_df.to_csv("results.csv", index=False)
    print("All models processed. Results saved!")


if __name__ == "__main__":
    main()
