import os
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def process_data():
    """
    Loads the sklearn digits dataset, splits into train/val/test (70%/10%/20%),
    and saves them to data/processed as CSV files.
    """
    output_dir = os.path.join(os.path.dirname(__file__), '../data/processed')
    digits = load_digits(as_frame=True)

    X = digits.data.copy()
    y = digits.target

    # Split train+val and test (80%-20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Split train+val into train and val (70%-10%-20%)
    # From the remaining 80%, we want 70/80 = 0.875 for train and 10/80 = 0.125 for val
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp
    )

    train_df = pd.concat([X_train, y_train.rename('target')], axis=1)
    val_df = pd.concat([X_val, y_val.rename('target')], axis=1)
    test_df = pd.concat([X_test, y_test.rename('target')], axis=1)

    os.makedirs(output_dir, exist_ok=True)

    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

    total_samples = len(X)
    print(f"Data split completed:")
    print(f"  Total samples: {total_samples}")
    print(f"  Train: {len(train_df)} samples ({len(train_df)/total_samples*100:.1f}%)")
    print(f"  Validation: {len(val_df)} samples ({len(val_df)/total_samples*100:.1f}%)")
    print(f"  Test: {len(test_df)} samples ({len(test_df)/total_samples*100:.1f}%)")
    print(f"Train, val, test sets saved to {output_dir}")

if __name__ == "__main__":
    process_data()
