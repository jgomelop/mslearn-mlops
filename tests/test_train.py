"""
Unit tests for training script.
Located in: tests/test_train.py

Run with: pytest tests/test_train.py
"""
import unittest
import os
import tempfile
import shutil
import pandas as pd
import numpy as np
from PIL import Image
import torch

# Add src to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'model'))

from xray_model.train import get_csvs_df, split_data, parse_args
from xray_model.model_utils import XRayDataset, get_model, create_data_loaders


class TestDataLoading(unittest.TestCase):
    """Test data loading functions"""
    
    def setUp(self):
        """Create temporary test data"""
        self.test_dir = tempfile.mkdtemp()
        
        # Create test CSV
        self.csv_path = os.path.join(self.test_dir, "test_data.csv")
        df = pd.DataFrame({
            'image_path': ['img1.png', 'img2.png', 'img3.png', 'img4.png'],
            'disease_1': [1, 0, 1, 0],
            'disease_2': [0, 1, 0, 1]
        })
        df.to_csv(self.csv_path, index=False)
        
        # Create test images
        self.img_dir = os.path.join(self.test_dir, "images")
        os.makedirs(self.img_dir)
        
        for i in range(1, 5):
            img = Image.fromarray(np.random.randint(0, 255, (256, 256), dtype=np.uint8))
            img.save(os.path.join(self.img_dir, f"img{i}.png"))
    
    def tearDown(self):
        """Clean up temporary files"""
        shutil.rmtree(self.test_dir)
    
    def test_get_csvs_df_success(self):
        df = get_csvs_df(self.test_dir)
        self.assertEqual(len(df), 4)
        self.assertIn('image_path', df.columns)
        self.assertIn('disease_1', df.columns)
        self.assertIn('disease_2', df.columns)
    
    def test_get_csvs_df_invalid_path(self):
        with self.assertRaises(RuntimeError) as context:
            get_csvs_df("/nonexistent/path")
        self.assertIn("Cannot use non-existent path", str(context.exception))
    
    def test_get_csvs_df_no_csv(self):
        empty_dir = tempfile.mkdtemp()
        with self.assertRaises(RuntimeError) as context:
            get_csvs_df(empty_dir)
        self.assertIn("No CSV files found", str(context.exception))
        shutil.rmtree(empty_dir)
    
    def test_split_data(self):
        df = pd.DataFrame({
            'image_path': [f'img{i}.png' for i in range(100)],
            'disease_1': np.random.randint(0, 2, 100),
            'disease_2': np.random.randint(0, 2, 100)
        })
        train_df, test_df = split_data(df)
        self.assertEqual(len(train_df), 80)
        self.assertEqual(len(test_df), 20)
        self.assertEqual(len(set(train_df['image_path']).intersection(set(test_df['image_path']))), 0)


class TestDataset(unittest.TestCase):
    """Test XRayDataset class with temporary images"""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.img_dir = os.path.join(self.test_dir, "images")
        os.makedirs(self.img_dir)
        for i in range(3):
            img = Image.fromarray(np.random.randint(0, 255, (256, 256), dtype=np.uint8))
            img.save(os.path.join(self.img_dir, f"img{i}.png"))
        self.df = pd.DataFrame({
            'image_path': [f'img{i}.png' for i in range(3)],
            'disease_1': [1, 0, 1],
            'disease_2': [0, 1, 0]
        })
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def test_dataset_length(self):
        dataset = XRayDataset(self.df, self.img_dir)
        self.assertEqual(len(dataset), 3)
    
    def test_dataset_getitem(self):
        dataset = XRayDataset(self.df, self.img_dir)
        img, labels = dataset[0]
        self.assertIsInstance(img, torch.Tensor)
        self.assertIsInstance(labels, torch.Tensor)
        self.assertEqual(len(labels), 2)
    
    def test_dataset_label_columns(self):
        dataset = XRayDataset(self.df, self.img_dir)
        self.assertEqual(dataset.label_cols, ['disease_1', 'disease_2'])


class TestModel(unittest.TestCase):
    """Test model utilities"""
    
    def test_get_model(self):
        model = get_model("densenet121-res224-all", num_classes=3)
        self.assertIsNotNone(model)
        self.assertEqual(model.classifier.out_features, 3)
    
    def test_model_forward_pass(self):
        model = get_model("densenet121-res224-all", num_classes=2)
        x = torch.randn(1, 1, 224, 224)
        output = model(x)
        self.assertEqual(output.shape, (1, 2))


class TestDataLoaders(unittest.TestCase):
    """Test data loader creation"""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.img_dir = os.path.join(self.test_dir, "images")
        os.makedirs(self.img_dir)
        for i in range(10):
            img = Image.fromarray(np.random.randint(0, 255, (256, 256), dtype=np.uint8))
            img.save(os.path.join(self.img_dir, f"img{i}.png"))
        self.train_df = pd.DataFrame({
            'image_path': [f'img{i}.png' for i in range(8)],
            'disease_1': np.random.randint(0, 2, 8),
            'disease_2': np.random.randint(0, 2, 8)
        })
        self.test_df = pd.DataFrame({
            'image_path': [f'img{i}.png' for i in range(8, 10)],
            'disease_1': np.random.randint(0, 2, 2),
            'disease_2': np.random.randint(0, 2, 2)
        })
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def test_create_data_loaders(self):
        train_loader, test_loader = create_data_loaders(
            self.train_df, self.test_df, self.img_dir, batch_size=2, img_size=224
        )
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(test_loader)
        for images, labels in train_loader:
            self.assertEqual(images.shape[1], 1)
            self.assertEqual(labels.shape[1], 2)
            break


class TestRealDataset(unittest.TestCase):
    """Test XRayDataset with real images and CSV from tests/datasets/xray"""
    
    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), "datasets", "xray")
        self.csv_path = os.path.join(self.data_dir, "labels.csv")
        self.df = pd.read_csv(self.csv_path)
        self.img_dir = os.path.join(self.data_dir, "images")
    
    def test_real_dataset_loading(self):
        dataset = XRayDataset(self.df, self.img_dir)
        self.assertGreater(len(dataset), 0)
        img, labels = dataset[0]
        self.assertIsInstance(img, torch.Tensor)
        self.assertIsInstance(labels, torch.Tensor)
        self.assertEqual(img.shape[0], 1)  # single channel
        self.assertEqual(labels.shape[0], len(dataset.label_cols))


class TestArgumentParsing(unittest.TestCase):
    """Test argument parsing"""
    
    def test_parse_args_defaults(self):
        import sys
        test_args = ['train.py', '--training_data', '/path/to/data']
        with unittest.mock.patch.object(sys, 'argv', test_args):
            args = parse_args()
        self.assertEqual(args.training_data, '/path/to/data')
        self.assertEqual(args.image_folder, 'images')
    
    def test_parse_args_custom(self):
        import sys
        test_args = [
            'train.py',
            '--training_data', '/custom/path',
            '--batch_size', '8',
            '--num_epochs', '5',
            '--learning_rate', '0.0001'
        ]
        with unittest.mock.patch.object(sys, 'argv', test_args):
            args = parse_args()
        self.assertEqual(args.batch_size, 8)
        self.assertEqual(args.num_epochs, 5)
        self.assertEqual(args.learning_rate, 0.0001)


if __name__ == '__main__':
    unittest.main()
