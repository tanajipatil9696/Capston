#!/usr/bin/env python
"""
Compress pickle files using gzip to reduce size for Heroku deployment.
Run from project root: python deployment/compress_models.py
Creates .pkl.gz files in deployment/ folder.
"""
import gzip
import shutil
import os
from pathlib import Path

# Files to compress
PICKLE_FILES = [
    'best_sentiment_model.pkl',
    'tfidf_vectorizer.pkl',
    'recommendation_matrix.pkl',
    'product_review_mapping.pkl'
]

def compress_file(input_path, output_path):
    """Compress a file using gzip."""
    try:
        with open(input_path, 'rb') as f_in:
            with gzip.open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        input_size = os.path.getsize(input_path) / (1024 ** 2)
        output_size = os.path.getsize(output_path) / (1024 ** 2)
        ratio = (1 - output_size / input_size) * 100
        
        print(f"✓ {input_path}")
        print(f"  {input_size:.1f}MB → {output_size:.1f}MB (compressed {ratio:.1f}%)")
        return True
    except FileNotFoundError:
        print(f"✗ {input_path} not found")
        return False
    except Exception as e:
        print(f"✗ Error compressing {input_path}: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("Compressing Pickle Files for Heroku Deployment")
    print("="*60 + "\n")
    
    deployment_dir = Path(__file__).parent
    project_root = deployment_dir.parent
    
    total_before = 0
    total_after = 0
    
    for pkl_file in PICKLE_FILES:
        # Try to find in project root first, then deployment folder
        input_path = project_root / pkl_file
        if not input_path.exists():
            input_path = deployment_dir / pkl_file
        
        if not input_path.exists():
            print(f"✗ {pkl_file} not found in {project_root} or {deployment_dir}")
            continue
        
        output_path = deployment_dir / f"{pkl_file}.gz"
        
        if compress_file(str(input_path), str(output_path)):
            total_before += os.path.getsize(input_path) / (1024 ** 2)
            total_after += os.path.getsize(output_path) / (1024 ** 2)
    
    print("\n" + "="*60)
    if total_before > 0:
        overall_ratio = (1 - total_after / total_before) * 100
        print(f"Total: {total_before:.1f}MB → {total_after:.1f}MB (compressed {overall_ratio:.1f}%)")
        print(f"\nCompressed files saved in: {deployment_dir}")
        print("Next: commit to git and push to Heroku")
    else:
        print("No pickle files found to compress.")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()
