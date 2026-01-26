"""
Training script for X-ray classification model.
Located in: src/model/train.py
"""

import argparse
import glob
import os
import pandas as pd
import mlflow
import mlflow.pytorch
from sklearn.model_selection import train_test_split
import torch
from model_utils import get_model, train_loop

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def main(args):
    """Main training function"""

    # Iniciar run de MLflow
    with mlflow.start_run():
        print("MLflow run iniciado correctamente")
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print(f"Experiment ID: {mlflow.active_run().info.experiment_id}")

        # Log de parámetros básicos
        mlflow.log_param("model_name", args.model_name)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("num_epochs", args.num_epochs)
        mlflow.log_param("learning_rate", args.learning_rate)

        # Cargar y dividir datos
        df = get_csvs_df(args.training_data)
        train_df, test_df = split_data(df)

        # Entrenar modelo
        train_model(args, train_df, test_df)

        print("✓ Entrenamiento completado exitosamente")


def get_csvs_df(path):
    """Load CSV file(s) from the data path"""
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")

    csv_files = glob.glob(f"{path}/*.csv")
    if not csv_files:
        raise RuntimeError(f"No CSV files found in provided data path: {path}")

    df = pd.concat((pd.read_csv(f) for f in csv_files), sort=False)
    print(f"✓ Loaded {len(df)} samples from {len(csv_files)} CSV file(s)")
    print(f"  Columns: {df.columns.tolist()}")

    return df


def split_data(df):
    """Split data into train and test sets"""
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    print(f"✓ Train samples: {len(train_df)}")
    print(f"✓ Test samples: {len(test_df)}")

    return train_df, test_df


def train_model(args, train_df, test_df):
    """Train the X-ray classification model"""

    # Determinar directorio de imágenes
    img_dir = os.path.join(args.training_data, args.image_folder)
    if not os.path.exists(img_dir):
        img_dir = args.training_data

    print(f"✓ Image directory: {img_dir}")

    # Obtener número de clases
    label_cols = [col for col in train_df.columns if col != "image_path"]
    num_classes = len(label_cols)

    print(f"✓ Number of classes: {num_classes}")
    print(f"  Classes: {label_cols}")

    # Log de metadatos
    mlflow.log_param("num_classes", num_classes)
    mlflow.log_param("train_samples", len(train_df))
    mlflow.log_param("test_samples", len(test_df))

    # Obtener y entrenar modelo
    model = get_model(model_name=args.model_name, num_classes=num_classes)

    trained_model = train_loop(
        model=model,
        train_df=train_df,
        test_df=test_df,
        img_dir=img_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        img_size=args.img_size,
    )

    # Guardar modelo localmente (Azure ML automatically uploads 'outputs' folder)
    os.makedirs("outputs", exist_ok=True)
    model_path = "outputs/model.pt"
    torch.save(trained_model.state_dict(), model_path)
    print(f"✓ Model saved to: {model_path}")

    # Log model using MLflow's PyTorch model logging
    try:
        mlflow.pytorch.log_model(trained_model, "model")
        print("✓ Model logged to MLflow using pytorch.log_model")
    except Exception as e:
        print(f"⚠ Could not log model to MLflow: {e}")
        print("  Model is still saved locally in outputs folder")

    return trained_model


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train X-ray classification model")

    # Data arguments
    parser.add_argument(
        "--training_data",
        type=str,
        required=True,
        help="Path to training data directory",
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        default="images",
        help="Subfolder containing images",
    )

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="densenet121-res224-all",
        help="Pretrained model to use",
    )

    # Training arguments
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=2,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Image size for resizing",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
