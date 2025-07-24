import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def model_fn(model_dir):
    """Load the trained model for inference."""
    return joblib.load(os.path.join(model_dir, "model.joblib"))

def train_model(args):
    print("Loading training data...")
    train_data_path = os.path.join(args.train_data, "cleaned_data.csv")
    cleaned_data = pd.read_csv(train_data_path)

    X = cleaned_data.drop('Placement', axis=1)
    y = cleaned_data['Placement']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=args.random_state)

    rf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state
    )
    rf.fit(X_train, y_train)

    val_score = rf.score(X_val, y_val)
    print(f"Validation Accuracy: {val_score:.4f}")

    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(rf, model_path)
    print(f"Model saved to {model_path}")

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_estimators", type=int, default=10)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--train_data", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print("Training with parameters:", args)
    train_model(args)