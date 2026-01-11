

import cv2
import numpy as np
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Configuration
DATASET_DIR = "Dataset/faces"
COMPARISON_OUTPUT_DIR = "Dataset/comparison"
PROGRESS_FILE = "clahe_comparison_progress.json"

# CLAHE parameters
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# ============ USER OPTIONS ============
# Choose which categories to process: ['real'], ['fake'], or ['real', 'fake']
CATEGORIES_TO_PROCESS = ['real', 'fake']

# Limit number of images per category (None = process all images)
MAX_IMAGES_PER_CATEGORY = None  # Change to a number like 100 to limit processing
# ======================================

def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    Path(directory).mkdir(parents=True, exist_ok=True)

def load_progress():
    """Load processing progress from JSON file."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {'real': [], 'fake': []}

def save_progress(progress):
    """Save processing progress to JSON file."""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)

def create_histogram(image, title):
    """Create histogram for an image."""
    # Convert to YUV to get the Y channel (luminance)
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    y_channel = img_yuv[:, :, 0]
    
    # Calculate histogram
    hist = cv2.calcHist([y_channel], [0], None, [256], [0, 256])
    
    return hist, y_channel

def create_comparison_image(original_path, category):
    """Create a comparison image with original, histogram, and CLAHE enhanced version."""
    # Read original image
    original = cv2.imread(original_path)
    if original is None:
        return False
    
    # Resize to standard size
    original = cv2.resize(original, (224, 224))
    
    # Apply CLAHE
    img_yuv = cv2.cvtColor(original, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
    clahe_enhanced = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
    # Get histograms
    hist_original, y_original = create_histogram(original, "Original")
    hist_clahe, y_clahe = create_histogram(clahe_enhanced, "CLAHE Enhanced")
    
    # Create figure with 4 subplots: original, original histogram, CLAHE histogram, CLAHE enhanced
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'CLAHE Comparison - {os.path.basename(original_path)}', fontsize=14, fontweight='bold')
    
    # Original image
    axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image', fontsize=12)
    axes[0, 0].axis('off')
    
    # Original histogram
    axes[0, 1].plot(hist_original, color='blue', linewidth=2)
    axes[0, 1].set_title('Original Histogram (Y Channel)', fontsize=12)
    axes[0, 1].set_xlabel('Pixel Intensity')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_xlim([0, 256])
    axes[0, 1].grid(True, alpha=0.3)
    
    # CLAHE histogram
    axes[1, 0].plot(hist_clahe, color='green', linewidth=2)
    axes[1, 0].set_title('CLAHE Enhanced Histogram (Y Channel)', fontsize=12)
    axes[1, 0].set_xlabel('Pixel Intensity')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_xlim([0, 256])
    axes[1, 0].grid(True, alpha=0.3)
    
    # CLAHE enhanced image
    axes[1, 1].imshow(cv2.cvtColor(clahe_enhanced, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('CLAHE Enhanced Image', fontsize=12)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # Save the comparison figure
    output_dir = os.path.join(COMPARISON_OUTPUT_DIR, category)
    ensure_dir(output_dir)
    
    filename = os.path.basename(original_path)
    output_path = os.path.join(output_dir, filename)
    
    # Save as high-quality image
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return True

def process_category_comparison(category):
    """Process all images in a category (real or fake)."""
    input_dir = os.path.join(DATASET_DIR, category)
    
    if not os.path.exists(input_dir):
        print(f"❌ Directory not found: {input_dir}")
        return 0
    
    # Load progress
    progress = load_progress()
    processed_files = set(progress.get(category, []))
    
    # Get all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    all_files = [f for f in os.listdir(input_dir) 
                 if f.lower().endswith(image_extensions)]
    
    # Filter out already processed files
    files_to_process = [f for f in all_files if f not in processed_files]
    
    # Apply limit if specified
    if MAX_IMAGES_PER_CATEGORY is not None:
        remaining_quota = MAX_IMAGES_PER_CATEGORY - len(processed_files)
        if remaining_quota <= 0:
            print(f"✅ Category '{category}' reached limit of {MAX_IMAGES_PER_CATEGORY} images!")
            return 0
        files_to_process = files_to_process[:remaining_quota]
    
    if not files_to_process:
        print(f"✅ All {len(all_files)} images in '{category}' category already processed!")
        return 0
    
    print(f"\n📊 Processing '{category}' category:")
    print(f"   Total images: {len(all_files)}")
    print(f"   Already processed: {len(processed_files)}")
    print(f"   To process: {len(files_to_process)}")
    if MAX_IMAGES_PER_CATEGORY is not None:
        print(f"   Limit per category: {MAX_IMAGES_PER_CATEGORY}")
    
    successful = 0
    failed = 0
    
    for idx, filename in enumerate(files_to_process, 1):
        input_path = os.path.join(input_dir, filename)
        
        try:
            if create_comparison_image(input_path, category):
                successful += 1
                processed_files.add(filename)
                
                # Save progress every 10 images
                if successful % 10 == 0:
                    progress[category] = list(processed_files)
                    save_progress(progress)
                    print(f"   ✓ Processed {successful}/{len(files_to_process)} images...")
            else:
                failed += 1
                print(f"   ⚠ Failed to process: {filename}")
        except Exception as e:
            failed += 1
            print(f"   ❌ Error processing {filename}: {str(e)}")
    
    # Final progress save
    progress[category] = list(processed_files)
    save_progress(progress)
    
    print(f"\n✅ Category '{category}' complete:")
    print(f"   Successfully processed: {successful}")
    print(f"   Failed: {failed}")
    
    return successful

def get_user_configuration():
    """Get processing configuration from user."""
    print("\n" + "=" * 70)
    print("⚙️  CONFIGURATION")
    print("=" * 70)
    
    # Get category choice
    print("\n1️⃣  Which category do you want to process?")
    print("   1. Real images only")
    print("   2. Fake images only")
    print("   3. Both (Real + Fake)")
    
    while True:
        choice = input("\nEnter your choice (1/2/3): ").strip()
        if choice == '1':
            categories = ['real']
            break
        elif choice == '2':
            categories = ['fake']
            break
        elif choice == '3':
            categories = ['real', 'fake']
            break
        else:
            print("❌ Invalid choice! Please enter 1, 2, or 3.")
    
    # Get limit choice
    print("\n2️⃣  How many images do you want to process per category?")
    print("   - Enter a number (e.g., 50, 100, 200)")
    print("   - Enter 'all' to process all images")
    
    while True:
        limit_input = input("\nEnter limit: ").strip().lower()
        if limit_input == 'all':
            limit = None
            break
        else:
            try:
                limit = int(limit_input)
                if limit > 0:
                    break
                else:
                    print("❌ Please enter a positive number!")
            except ValueError:
                print("❌ Invalid input! Enter a number or 'all'.")
    
    return categories, limit

def main():
    """Main function to process all categories."""
    print("=" * 70)
    print("🔬 CLAHE COMPARISON GENERATOR")
    print("=" * 70)
    print(f"\nInput directory: {DATASET_DIR}")
    print(f"Output directory: {COMPARISON_OUTPUT_DIR}")
    print(f"Progress file: {PROGRESS_FILE}")
    
    # Get user configuration
    categories_to_process, max_images = get_user_configuration()
    
    # Display configuration
    print(f"\n⚙️  Your Configuration:")
    print(f"   Categories to process: {categories_to_process}")
    if max_images is not None:
        print(f"   Max images per category: {max_images}")
    else:
        print(f"   Max images per category: All images")
    
    print("\nThis will generate comparison images showing:")
    print("  • Original image")
    print("  • Original histogram (Y channel)")
    print("  • CLAHE enhanced histogram (Y channel)")
    print("  • CLAHE enhanced image")
    
    # Ensure output directory exists
    ensure_dir(COMPARISON_OUTPUT_DIR)
    
    # Process only selected categories with user's limit
    total_processed = 0
    
    # Temporarily set the global limit for processing
    global MAX_IMAGES_PER_CATEGORY
    MAX_IMAGES_PER_CATEGORY = max_images
    
    for category in categories_to_process:
        if category not in ['real', 'fake']:
            print(f"⚠️  Warning: Unknown category '{category}' - skipping")
            continue
        processed = process_category_comparison(category)
        total_processed += processed
    
    print("\n" + "=" * 70)
    print("🎉 COMPARISON GENERATION COMPLETE!")
    print("=" * 70)
    print(f"\nTotal comparison images generated: {total_processed}")
    print(f"Output location: {COMPARISON_OUTPUT_DIR}/")
    print(f"  • {COMPARISON_OUTPUT_DIR}/real/")
    print(f"  • {COMPARISON_OUTPUT_DIR}/fake/")
    print("\nYou can resume processing anytime if interrupted.")
    print("Progress is saved in:", PROGRESS_FILE)

if __name__ == "__main__":
    main()
