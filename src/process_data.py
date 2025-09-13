
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

def process_data():
    """
    Loads the sklearn digits dataset, splits into train/val/test, and saves them to the processed directory as CSVs.
    """
    output_dir = os.path.join(os.path.dirname(__file__), '../data/processed')
    digits = load_digits(as_frame=True)
    X = digits.data
    y = digits.target

    # First split into train+val and test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # Then split train+val into train and val (e.g., 80% train, 20% val of the remaining 80%)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
    # This results in: 60% train, 20% val, 20% test

    train_df = pd.concat([X_train, y_train.rename('target')], axis=1)
    val_df = pd.concat([X_val, y_val.rename('target')], axis=1)
    test_df = pd.concat([X_test, y_test.rename('target')], axis=1)

    os.makedirs(output_dir, exist_ok=True)
    print("Saving processed data...")
    train_path = os.path.join(output_dir, 'train.csv')
    val_path = os.path.join(output_dir, 'val.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Train, val, and test sets saved to {output_dir}")

if __name__ == "__main__":
	process_data()