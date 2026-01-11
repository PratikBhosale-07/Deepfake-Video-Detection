import cv2
import os
import pandas as pd
import json
from datetime import datetime

# -------- CONFIG ----------
DATASET_DIR = "Dataset/faces"   # folder containing 'real/' and 'fake/'
CLAHE_OUTPUT_DIR = "Dataset/faces_clahe"  # folder for CLAHE enhanced images
CLAHE_PROGRESS_FILE = "clahe_enhancement_progress.json"  # file to track progress
# --------------------------

def load_clahe_progress():
    """Load CLAHE enhancement progress from JSON file"""
    if os.path.exists(CLAHE_PROGRESS_FILE):
        with open(CLAHE_PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {
        "real": {
            "processed_images": [],
            "clahe_applied": 0,
            "total_processed": 0
        },
        "fake": {
            "processed_images": [],
            "clahe_applied": 0,
            "total_processed": 0
        },
        "settings": {
            "apply_clahe": False
        },
        "last_updated": None
    }

def save_clahe_progress(progress):
    """Save CLAHE enhancement progress to JSON file"""
    progress["last_updated"] = datetime.now().isoformat()
    with open(CLAHE_PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)

def get_user_input():
    """Get CLAHE enhancement options from user"""
    # Get category selection
    while True:
        print("\n🏷️ Which category would you like to process?")
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
    
    # Get CLAHE preference
    while True:
        print(f"\n🎨 Apply CLAHE enhancement to images?")
        print("1. No CLAHE (keep original images)")
        print("2. Apply CLAHE (improve contrast and lighting)")
        
        choice = input("\nEnter your choice (1/2): ").strip()
        
        if choice == "1":
            apply_clahe = False
            break
        elif choice == "2":
            apply_clahe = True
            break
        else:
            print("❌ Please enter 1 or 2.")
    
    # Get number of images to process
    while True:
        try:
            if len(categories) == 1:
                num_images = int(input(f"\n🎬 How many {categories[0]} images would you like to process? (0 for all): "))
            else:
                num_images = int(input(f"\n🎬 How many images would you like to process per category? (0 for all): "))
            
            if num_images >= 0:
                return categories, apply_clahe, num_images if num_images > 0 else None
            else:
                print("❌ Please enter a non-negative number.")
        except ValueError:
            print("❌ Please enter a valid number.")

def apply_clahe_enhancement(image_path, category):
    """Apply CLAHE enhancement to image and save to separate directory"""
    img = cv2.imread(image_path)
    if img is None:
        return False, None
    
    # Convert to YUV color space
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    
    # Apply CLAHE to luminance channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
    
    # Convert back to BGR
    img_enhanced = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
    # Create CLAHE output directory structure
    clahe_category_dir = os.path.join(CLAHE_OUTPUT_DIR, category)
    os.makedirs(clahe_category_dir, exist_ok=True)
    
    # Create output path
    img_name = os.path.basename(image_path)
    clahe_output_path = os.path.join(clahe_category_dir, img_name)
    
    # Save enhanced image to CLAHE directory
    cv2.imwrite(clahe_output_path, img_enhanced)
    return True, clahe_output_path

def process_category_clahe(category, apply_clahe, progress, max_images=None):
    """Process CLAHE enhancement for a specific category"""
    folder = os.path.join(DATASET_DIR, category)
    
    if not os.path.exists(folder):
        print(f"❌ Directory not found: {folder}")
        return 0, 0
    
    # Get all image files
    all_images = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    
    # Filter out already processed images
    processed_images = set(progress[category]["processed_images"])
    unprocessed_images = [img for img in all_images if img not in processed_images]
    
    print(f"\n🎨 {category.upper()} CLAHE ENHANCEMENT:")
    print(f"📁 Total {category} images available: {len(all_images)}")
    print(f"✅ Already processed: {progress[category]['total_processed']}")
    print(f"📋 Remaining to process: {len(unprocessed_images)}")
    
    if len(unprocessed_images) == 0:
        print(f"🎉 All {category} images have already been processed!")
        return 0, 0
    
    # Limit images to process if specified
    if max_images:
        images_to_process = unprocessed_images[:max_images]
    else:
        images_to_process = unprocessed_images
    
    if len(images_to_process) == 0:
        print(f"⚠️ No {category} images to process")
        return 0, 0
    
    print(f"\n🚀 Processing {len(images_to_process)} {category} images...")
    print(f"🎨 Apply CLAHE: {'YES' if apply_clahe else 'NO'}")
    if apply_clahe:
        print(f"📁 CLAHE output directory: {CLAHE_OUTPUT_DIR}")
    print("=" * 50)
    
    processed_count = 0
    clahe_applied_count = 0
    
    for img_name in images_to_process:
        img_path = os.path.join(folder, img_name)
        
        print(f"\n📸 Processing: {img_name}")
        
        # Check if file still exists
        if not os.path.exists(img_path):
            print(f"⚠️ Image no longer exists: {img_name}")
            progress[category]["processed_images"].append(img_name)
            progress[category]["total_processed"] += 1
            processed_count += 1
            continue
        
        # Apply CLAHE if requested
        if apply_clahe:
            success, clahe_path = apply_clahe_enhancement(img_path, category)
            if success:
                print(f"🎨 CLAHE applied and saved to: {clahe_path}")
                clahe_applied_count += 1
                progress[category]["clahe_applied"] += 1
            else:
                print(f"❌ Failed to apply CLAHE to: {img_name}")
        else:
            print(f"✅ Kept original: {img_name}")
        
        # Update progress
        progress[category]["processed_images"].append(img_name)
        progress[category]["total_processed"] += 1
        processed_count += 1
        
        # Save progress every 25 images
        if processed_count % 25 == 0:
            save_clahe_progress(progress)
            print(f"💾 Progress saved: {processed_count}/{len(images_to_process)} images")
        
        print(f"✅ Progress: {processed_count}/{len(images_to_process)} images processed")
    
    # Save final progress
    save_clahe_progress(progress)
    
    return processed_count, clahe_applied_count

def main():
    """Main CLAHE enhancement process"""
    # Load existing progress
    progress = load_clahe_progress()
    
    # Display current status
    total_real_processed = progress["real"]["total_processed"]
    total_fake_processed = progress["fake"]["total_processed"]
    total_real_clahe = progress["real"]["clahe_applied"]
    total_fake_clahe = progress["fake"]["clahe_applied"]
    
    print("🎨 CLAHE ENHANCEMENT STATUS:")
    print(f"🟢 Real: Processed={total_real_processed}, CLAHE={total_real_clahe}")
    print(f"🔴 Fake: Processed={total_fake_processed}, CLAHE={total_fake_clahe}")
    if progress["last_updated"]:
        print(f"🕒 Last updated: {progress['last_updated']}")
        if progress["settings"]["apply_clahe"]:
            print(f"⚙️ Previous settings: CLAHE={progress['settings']['apply_clahe']}")
    else:
        print("🆕 Starting fresh CLAHE enhancement process")
    
    # Get user input
    categories, apply_clahe, max_images = get_user_input()
    
    # Update settings in progress
    progress["settings"]["apply_clahe"] = apply_clahe
    
    total_processed = 0
    total_clahe_applied = 0
    
    print(f"\n🎯 Processing {categories} categories")
    print(f"🎨 Apply CLAHE: {'YES' if apply_clahe else 'NO'}")
    if apply_clahe:
        print(f"📁 CLAHE output directory: {CLAHE_OUTPUT_DIR}")
    print(f"🔢 Max images per category: {max_images if max_images else 'ALL'}")
    print("=" * 60)
    
    for category in categories:
        processed, clahe_applied = process_category_clahe(
            category, apply_clahe, progress, max_images
        )
        
        total_processed += processed
        total_clahe_applied += clahe_applied
        
        print(f"\n✅ {category.upper()} processing complete:")
        print(f"   📊 Processed: {processed}")
        print(f"   🎨 CLAHE applied: {clahe_applied}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎉 CLAHE ENHANCEMENT COMPLETE!")
    print("=" * 60)
    print(f"📊 Total processed this session: {total_processed}")
    print(f"🎨 CLAHE enhancements applied: {total_clahe_applied}")
    
    # Display final statistics
    print(f"\n📈 OVERALL STATISTICS:")
    for category in ["real", "fake"]:
        cat_processed = progress[category]["total_processed"]
        cat_clahe = progress[category]["clahe_applied"]
        print(f"🏷️ {category.upper()}: Processed={cat_processed}, CLAHE={cat_clahe}")
    
    grand_processed = progress["real"]["total_processed"] + progress["fake"]["total_processed"]
    grand_clahe = progress["real"]["clahe_applied"] + progress["fake"]["clahe_applied"]
    
    print(f"🏆 GRAND TOTAL: Processed={grand_processed}, CLAHE={grand_clahe}")
    
    if grand_clahe > 0:
        print(f"📁 CLAHE enhanced images saved to: {CLAHE_OUTPUT_DIR}")
        print(f"   📂 Structure: {CLAHE_OUTPUT_DIR}/real/ and {CLAHE_OUTPUT_DIR}/fake/")
    
    # Save final progress
    save_clahe_progress(progress)
    print(f"💾 Progress saved to: {CLAHE_PROGRESS_FILE}")

if __name__ == "__main__":
    main()