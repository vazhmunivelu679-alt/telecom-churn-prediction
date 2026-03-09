import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score


def load_processed_data(path):
    """
    Load processed telecom churn dataset.
    """
    df = pd.read_csv(path)
    return df


def split_data(df):
    """
    Split dataset into features and target.
    """
    X = df.drop("churn", axis=1)
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, y_train):
    """
    Train Random Forest model.
    """

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance.
    """

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nModel Evaluation\n")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))


def save_model(model, path):
    """
    Save trained model.
    """
    joblib.dump(model, path)
    print(f"\nModel saved to {path}")


def main():

    data_path = "../data/processed/high_value_churn_data.csv"
    model_path = "../models/random_forest_model.pkl"

    df = load_processed_data(data_path)

    X_train, X_test, y_train, y_test = split_data(df)

    model = train_random_forest(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    save_model(model, model_path)


if __name__ == "__main__":
    main()