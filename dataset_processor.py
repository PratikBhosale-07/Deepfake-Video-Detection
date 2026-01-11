import os
import cv2
import pandas as pd
import random
import json
from math import floor
from datetime import datetime

# -------- CONFIG ----------
SRC_DIR = "Dataset/faces"      # input folder (already contains 'real' and 'fake')
OUT_DIR = "Dataset/processed"  # output processed dataset
IMG_SIZE = (224, 224)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42
PROCESSING_PROGRESS_FILE = "dataset_processing_progress.json"  # file to track processing progress
# --------------------------

random.seed(RANDOM_SEED)

def load_processing_progress():
    """Load dataset processing progress from JSON file"""
    if os.path.exists(PROCESSING_PROGRESS_FILE):
        with open(PROCESSING_PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {
        "train": {
            "real": {"processed_images": [], "total_processed": 0},
            "fake": {"processed_images": [], "total_processed": 0}
        },
        "val": {
            "real": {"processed_images": [], "total_processed": 0},
            "fake": {"processed_images": [], "total_processed": 0}
        },
        "test": {
            "real": {"processed_images": [], "total_processed": 0},
            "fake": {"processed_images": [], "total_processed": 0}
        },
        "dataset_splits_created": False,
        "dataset_balanced": False,
        "last_updated": None
    }

def save_processing_progress(progress):
    """Save dataset processing progress to JSON file"""
    progress["last_updated"] = datetime.now().isoformat()
    with open(PROCESSING_PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)

def get_user_input():
    """Get processing options from user"""
    # Get split selection
    while True:
        print("\n📊 Which dataset split would you like to process?")
        print("1. Train set only")
        print("2. Validation set only")
        print("3. Test set only")
        print("4. All splits (train, val, test)")
        
        choice = input("\nEnter your choice (1/2/3/4): ").strip()
        
        if choice == "1":
            splits_to_process = ["train"]
            break
        elif choice == "2":
            splits_to_process = ["val"]
            break
        elif choice == "3":
            splits_to_process = ["test"]
            break
        elif choice == "4":
            splits_to_process = ["train", "val", "test"]
            break
        else:
            print("❌ Please enter 1, 2, 3, or 4.")
    
    # Get category selection
    while True:
        print(f"\n🏷️ Which categories would you like to process?")
        print("1. Real images only")
        print("2. Fake images only")
        print("3. Both (real and fake)")
        
        choice = input("\nEnter your choice (1/2/3): ").strip()
        
        if choice == "1":
            categories = ["real"]
            break
        elif choice == "2":
            categories = ["fake"]
            break
        elif choice == "3":
            categories = ["real", "fake"]
            break
        else:
            print("❌ Please enter 1, 2, or 3.")
    
    # Get balance preference
    balance_dataset = False
    if len(categories) == 2:  # Both real and fake
        while True:
            print(f"\n⚖️ Do you want to balance the dataset (equal real and fake images)?")
            print("y. Yes - Balance dataset (recommended)")
            print("n. No - Use all available images")
            
            balance_choice = input("\nEnter your choice (y/n): ").strip().lower()
            
            if balance_choice in ['y', 'yes']:
                balance_dataset = True
                break
            elif balance_choice in ['n', 'no']:
                balance_dataset = False
                break
            else:
                print("❌ Please enter 'y' or 'n'.")
    
    # Get number of images to process per category
    while True:
        try:
            if len(splits_to_process) == 1 and len(categories) == 1:
                num_images = int(input(f"\n🎬 How many {categories[0]} images would you like to process for {splits_to_process[0]} set? (0 for all): "))
            else:
                num_images = int(input(f"\n🎬 How many images would you like to process per category per split? (0 for all): "))
            
            if num_images >= 0:
                return splits_to_process, categories, balance_dataset, num_images if num_images > 0 else None
            else:
                print("❌ Please enter a non-negative number.")
        except ValueError:
            print("❌ Please enter a valid number.")

def ensure_dir(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)

def process_and_copy_images(src_list, dest_dir, label, split, progress, max_images=None):
    """Process and copy images with progress tracking"""
    meta = []
    processed_count = 0
    skipped_count = 0
    
    # Get already processed images for this split and label
    processed_images = set(progress[split][label]["processed_images"])
    
    # Filter out already processed images
    unprocessed_images = [img for img in src_list if os.path.basename(img) not in processed_images]
    
    print(f"📁 Total {label} images available: {len(src_list)}")
    print(f"✅ Already processed: {len(processed_images)}")
    print(f"📋 Remaining to process: {len(unprocessed_images)}")
    
    if len(unprocessed_images) == 0:
        print(f"🎉 All {label} images for {split} split have already been processed!")
        return meta, 0, 0
    
    # Limit images to process if specified
    if max_images:
        images_to_process = unprocessed_images[:max_images]
    else:
        images_to_process = unprocessed_images
    
    print(f"🚀 Processing {len(images_to_process)} {label} images for {split} set...")
    
    for src_path in images_to_process:
        img_name = os.path.basename(src_path)
        
        # Check if destination already exists
        dest_path = os.path.join(dest_dir, img_name)
        if os.path.exists(dest_path):
            print(f"⚠️ Already exists, skipping: {img_name}")
            skipped_count += 1
            progress[split][label]["processed_images"].append(img_name)
            progress[split][label]["total_processed"] += 1
            continue
        
        # Read and process image
        img = cv2.imread(src_path)
        if img is None:
            print(f"❌ Cannot read {src_path}")
            continue
        
        # Normalize: resize + scale to [0,255] (CV2 saves automatically)
        img_resized = cv2.resize(img, IMG_SIZE)
        cv2.imwrite(dest_path, img_resized)
        meta.append([dest_path, label])
        
        # Update progress
        progress[split][label]["processed_images"].append(img_name)
        progress[split][label]["total_processed"] += 1
        processed_count += 1
        
        # Save progress every 50 images
        if processed_count % 50 == 0:
            save_processing_progress(progress)
            print(f"💾 Progress saved: {processed_count}/{len(images_to_process)} images processed")
        
        print(f"✅ Processed: {img_name} ({processed_count}/{len(images_to_process)})")
    
    return meta, processed_count, skipped_count

def split_list(data, train_ratio, val_ratio):
    """Split data into train, validation, and test sets"""
    n = len(data)
    n_train = floor(train_ratio * n)
    n_val = floor(val_ratio * n)
    return data[:n_train], data[n_train:n_train+n_val], data[n_train+n_val:]

def create_dataset_splits(progress, balance_dataset=True):
    """Create and save dataset splits"""
    if progress["dataset_splits_created"]:
        print("📊 Dataset splits already created, loading existing splits...")
        # Load existing splits from saved files or recreate
        return load_existing_splits()
    
    print("📊 Creating dataset splits...")
    
    # Collect all image paths
    real_dir = os.path.join(SRC_DIR, "real")
    fake_dir = os.path.join(SRC_DIR, "fake")
    
    if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
        print(f"❌ Source directories not found: {real_dir} or {fake_dir}")
        print("Please run face extraction first!")
        return None
    
    real_imgs = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith((".jpg", ".png", ".jpeg"))]
    fake_imgs = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith((".jpg", ".png", ".jpeg"))]
    
    print(f"📁 Found {len(real_imgs)} real images and {len(fake_imgs)} fake images")
    
    # Balance dataset (optional)
    if balance_dataset and len(real_imgs) > 0 and len(fake_imgs) > 0:
        n = min(len(real_imgs), len(fake_imgs))
        real_imgs, fake_imgs = real_imgs[:n], fake_imgs[:n]
        print(f"⚖️ Balanced dataset to {n} images per class")
        progress["dataset_balanced"] = True
    
    # Shuffle
    random.shuffle(real_imgs)
    random.shuffle(fake_imgs)
    
    # Split
    train_real, val_real, test_real = split_list(real_imgs, TRAIN_RATIO, VAL_RATIO)
    train_fake, val_fake, test_fake = split_list(fake_imgs, TRAIN_RATIO, VAL_RATIO)
    
    splits = {
        "train": {"real": train_real, "fake": train_fake},
        "val": {"real": val_real, "fake": val_fake},
        "test": {"real": test_real, "fake": test_fake},
    }
    
    # Save splits information
    splits_info = {
        "train": {"real": len(train_real), "fake": len(train_fake)},
        "val": {"real": len(val_real), "fake": len(val_fake)},
        "test": {"real": len(test_real), "fake": len(test_fake)},
    }
    
    # Ensure output directory exists before saving splits info
    ensure_dir(OUT_DIR)
    with open(os.path.join(OUT_DIR, "splits_info.json"), 'w') as f:
        json.dump(splits_info, f, indent=2)
    
    progress["dataset_splits_created"] = True
    save_processing_progress(progress)
    
    print("📊 Dataset splits created:")
    for split, counts in splits_info.items():
        print(f"  {split}: Real={counts['real']}, Fake={counts['fake']}, Total={counts['real'] + counts['fake']}")
    
    return splits

def load_existing_splits():
    """Load existing dataset splits (simplified version for demo)"""
    # In a real implementation, you might save and load the actual file lists
    # For now, we'll recreate them
    return create_fresh_splits()

def create_fresh_splits():
    """Create fresh splits from source directory"""
    real_dir = os.path.join(SRC_DIR, "real")
    fake_dir = os.path.join(SRC_DIR, "fake")
    
    real_imgs = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith((".jpg", ".png", ".jpeg"))]
    fake_imgs = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith((".jpg", ".png", ".jpeg"))]
    
    # Shuffle with same seed for consistency
    random.shuffle(real_imgs)
    random.shuffle(fake_imgs)
    
    train_real, val_real, test_real = split_list(real_imgs, TRAIN_RATIO, VAL_RATIO)
    train_fake, val_fake, test_fake = split_list(fake_imgs, TRAIN_RATIO, VAL_RATIO)
    
    return {
        "train": {"real": train_real, "fake": train_fake},
        "val": {"real": val_real, "fake": val_fake},
        "test": {"real": test_real, "fake": test_fake},
    }

def main():
    """Main dataset processing function"""
    # Load existing progress
    progress = load_processing_progress()
    
    # Display current status
    print("📊 DATASET PROCESSING STATUS:")
    for split in ["train", "val", "test"]:
        real_processed = progress[split]["real"]["total_processed"]
        fake_processed = progress[split]["fake"]["total_processed"]
        print(f"🔄 {split.upper()}: Real={real_processed}, Fake={fake_processed}")
    
    if progress["last_updated"]:
        print(f"🕒 Last updated: {progress['last_updated']}")
    else:
        print("🆕 Starting fresh dataset processing")
    
    # Get user input
    splits_to_process, categories, balance_dataset, max_images = get_user_input()
    
    # Create dataset splits
    splits = create_dataset_splits(progress, balance_dataset)
    if splits is None:
        return
    
    # Create output directories for selected splits and categories
    for split in splits_to_process:
        for category in categories:
            ensure_dir(os.path.join(OUT_DIR, split, category))
    
    # Process selected splits and categories
    total_processed = 0
    total_skipped = 0
    
    print(f"\n🎯 Processing {splits_to_process} split(s) for {categories} categories")
    print(f"⚖️ Dataset balanced: {'YES' if balance_dataset else 'NO'}")
    print(f"🔢 Max images per category per split: {max_images if max_images else 'ALL'}")
    print("=" * 60)
    
    for split in splits_to_process:
        print(f"\n📁 Processing {split.upper()} split:")
        split_metadata = []
        
        for category in categories:
            if category not in splits[split] or len(splits[split][category]) == 0:
                print(f"⚠️ No {category} images available for {split} split")
                continue
            
            print(f"\n🏷️ Processing {category} images for {split} split...")
            dest_dir = os.path.join(OUT_DIR, split, category)
            
            meta, processed, skipped = process_and_copy_images(
                splits[split][category], dest_dir, category, split, progress, max_images
            )
            
            split_metadata.extend(meta)
            total_processed += processed
            total_skipped += skipped
            
            print(f"✅ {category} processing complete: {processed} processed, {skipped} skipped")
        
        # Save metadata for this split
        if split_metadata:
            df = pd.DataFrame(split_metadata, columns=["filename", "label"])
            metadata_file = os.path.join(OUT_DIR, f"metadata_{split}.csv")
            
            # Combine with existing metadata if it exists
            if os.path.exists(metadata_file):
                existing_df = pd.read_csv(metadata_file)
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=['filename'])
            else:
                combined_df = df
            
            combined_df.to_csv(metadata_file, index=False)
            print(f"📂 Saved metadata_{split}.csv with {len(combined_df)} records")
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎉 DATASET PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"📊 Images processed this session: {total_processed}")
    print(f"⚠️ Images skipped (already existed): {total_skipped}")
    
    # Display final statistics
    for split in ["train", "val", "test"]:
        real_total = progress[split]["real"]["total_processed"]
        fake_total = progress[split]["fake"]["total_processed"]
        split_total = real_total + fake_total
        print(f"📈 {split.upper()} total: {split_total} (Real: {real_total}, Fake: {fake_total})")
    
    grand_total = sum(progress[split]["real"]["total_processed"] + progress[split]["fake"]["total_processed"] 
                     for split in ["train", "val", "test"])
    print(f"🏆 Grand total processed: {grand_total} images")
    print(f"✅ Processed dataset saved at: {OUT_DIR}")
    
    # Save final progress
    save_processing_progress(progress)

if __name__ == "__main__":
    main()