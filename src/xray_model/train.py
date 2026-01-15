"""
Training script for X-ray classification model.
Located in: src/model/train.py
"""
import argparse
import glob
import os
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split

from .model_utils import XRayDataset, get_model, train_loop


def main(args):
    """Main training function"""
    # Enable MLflow autologging
    mlflow.autolog()

    # Read data
    df = get_csvs_df(args.training_data)

    # Split data
    train_df, test_df = split_data(df)

    # Train model
    train_model(args, train_df, test_df)


def get_csvs_df(path):
    """Load CSV file(s) from the data path"""
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")
    
    csv_files = glob.glob(f"{path}/*.csv")
    if not csv_files:
        raise RuntimeError(f"No CSV files found in provided data path: {path}")
    
    df = pd.concat((pd.read_csv(f) for f in csv_files), sort=False)
    print(f"Loaded {len(df)} samples from {len(csv_files)} CSV file(s)")
    print(f"Columns: {df.columns.tolist()}")
    
    return df


def split_data(df):
    """Split data into train and test sets"""
    train_df, test_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42
    )
    
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    return train_df, test_df


def train_model(args, train_df, test_df):
    """Train the X-ray classification model"""
    # Determine image directory
    img_dir = os.path.join(args.training_data, args.image_folder)
    if not os.path.exists(img_dir):
        img_dir = args.training_data
    
    print(f"Image directory: {img_dir}")
    
    # Get number of classes from dataframe
    label_cols = [col for col in train_df.columns if col != 'image_path']
    num_classes = len(label_cols)
    
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {label_cols}")
    
    # Get model
    model = get_model(
        model_name=args.model_name,
        num_classes=num_classes
    )
    
    # Train model
    trained_model = train_loop(
        model=model,
        train_df=train_df,
        test_df=test_df,
        img_dir=img_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        img_size=args.img_size
    )
    
    print("Training completed successfully!")
    return trained_model


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train X-ray classification model")
    
    # Data arguments
    parser.add_argument(
        "--training_data", 
        dest="training_data",
        type=str, 
        required=True,
        help="Path to training data directory"
    )
    parser.add_argument(
        "--image_folder",
        dest="image_folder",
        type=str,
        default="images",
        help="Subfolder containing images"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        dest="model_name",
        type=str,
        default="densenet121-res224-all",
        help="Pretrained model to use"
    )
    
    # Training arguments
    parser.add_argument(
        "--learning_rate",
        dest="learning_rate",
        type=float,
        default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--batch_size",
        dest="batch_size",
        type=int,
        default=4,
        help="Batch size"
    )
    parser.add_argument(
        "--num_epochs",
        dest="num_epochs",
        type=int,
        default=2,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--img_size",
        dest="img_size",
        type=int,
        default=224,
        help="Image size for resizing"
    )
    
    # Parse args
    args = parser.parse_args()
    
    # Return args
    return args


# Run script
if __name__ == "__main__":
    # Parse args
    args = parse_args()
    
    # Run main function
    main(args)