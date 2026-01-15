"""
Generate synthetic X-ray-like images for testing.
Usage: python generate_test_data.py --num_images 25 --output_dir ./data
"""
import argparse
import os
import pandas as pd
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_images", type=int, default=25)
    parser.add_argument("--output_dir", type=str, default="./data")
    args = parser.parse_args()
    
    print(f"Generating {args.num_images} synthetic images in {args.output_dir}")
    
    # Create output directories
    img_dir = os.path.join(args.output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    
    # Disease labels
    diseases = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Pneumonia', 'Mass', 'Nodule', 'Pneumothorax', 'Infiltration']
    
    rows = []
    
    for i in range(args.num_images):
        # Create synthetic grayscale image (256x256)
        img = np.random.randint(50, 200, (256, 256), dtype=np.uint8)
        
        # Add some structure (darker center simulating chest cavity)
        y, x = np.ogrid[:256, :256]
        mask = ((x - 128) ** 2 + (y - 128) ** 2) <= (85) ** 2
        img[mask] = img[mask] * 0.7
        
        # Save image
        img_filename = f"xray_{i:03d}.png"
        img_path = os.path.join(img_dir, img_filename)
        Image.fromarray(img).save(img_path)
        
        # Generate random labels
        row = {'image_path': img_filename}
        for disease in diseases:
            row[disease] = np.random.randint(0, 2)
        rows.append(row)
        
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{args.num_images}")
    
    # Save CSV
    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.output_dir, "labels.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"\n✅ Done!")
    print(f"   Images: {img_dir}")
    print(f"   Labels: {csv_path}")
    print(f"\nSample data:")
    print(df.head(3))


if __name__ == "__main__":
    main()