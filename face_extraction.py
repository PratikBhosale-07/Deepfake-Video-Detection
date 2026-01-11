import os
import cv2
import pandas as pd
import json
from datetime import datetime
# Suppress TensorFlow info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from mtcnn import MTCNN

# ---------- CONFIGURATION ----------
FRAME_PATH = "Dataset/frames"   # from Step 1
FACE_PATH = "Dataset/faces"     # output folder for cropped faces
FACE_PROGRESS_FILE = "face_extraction_progress.json"  # file to track face extraction progress
REQUIRED_SIZE = (224, 224)     # output face size

# Create output directories
os.makedirs(os.path.join(FACE_PATH, "real"), exist_ok=True)
os.makedirs(os.path.join(FACE_PATH, "fake"), exist_ok=True)

# Initialize MTCNN detector
detector = MTCNN()
metadata = []

def load_face_progress():
    """Load face extraction progress from JSON file"""
    if os.path.exists(FACE_PROGRESS_FILE):
        with open(FACE_PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {
        "real": {
            "processed_frames": [],
            "faces_extracted": 0,
            "frames_deleted": 0,
            "total_processed": 0
        },
        "fake": {
            "processed_frames": [],
            "faces_extracted": 0,
            "frames_deleted": 0,
            "total_processed": 0
        },
        "last_updated": None
    }

def save_face_progress(progress):
    """Save face extraction progress to JSON file"""
    progress["last_updated"] = datetime.now().isoformat()
    with open(FACE_PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)

def get_user_input():
    """Get category and processing options from user"""
    # Get category selection
    while True:
        print("\n👤 Which category would you like to process for face extraction?")
        print("1. Real frames")
        print("2. Fake frames")
        print("3. Both (real and fake)")
        
        choice = input("\nEnter your choice (1/2/3): ").strip()
        
        if choice == "1":
            category = "real"
            break
        elif choice == "2":
            category = "fake"
            break
        elif choice == "3":
            category = "both"
            break
        else:
            print("❌ Please enter 1, 2, or 3.")
    
    # Get deletion preference
    while True:
        print(f"\n🗑️ Do you want to delete frames where no face is detected?")
        print("y. Yes - Delete frames with no faces (saves space)")
        print("n. No - Keep all frames (safer option)")
        
        delete_choice = input("\nEnter your choice (y/n): ").strip().lower()
        
        if delete_choice in ['y', 'yes']:
            delete_no_face = True
            break
        elif delete_choice in ['n', 'no']:
            delete_no_face = False
            break
        else:
            print("❌ Please enter 'y' or 'n'.")
    
    # Get number of frames to process
    while True:
        try:
            if category == "both":
                num_frames = int(input(f"\n🎬 How many frames would you like to process from EACH category? (0 for all remaining): "))
            else:
                num_frames = int(input(f"\n🎬 How many {category} frames would you like to process? (0 for all remaining): "))
            
            if num_frames >= 0:
                return category, delete_no_face, num_frames if num_frames > 0 else None
            else:
                print("❌ Please enter a non-negative number.")
        except ValueError:
            print("❌ Please enter a valid number.")

def extract_face(image_path, label, out_dir, delete_no_face=False, required_size=(224,224)):
    """Extract face from image and optionally delete original if no face found"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Cannot read image: {image_path}")
        return None, False
    
    results = detector.detect_faces(img)
    if results:
        # Face detected - extract and save
        x, y, w, h = results[0]['box']
        # Fix negative values and ensure within image bounds
        x, y = max(0, x), max(0, y)
        x = min(x, img.shape[1] - 1)
        y = min(y, img.shape[0] - 1)
        w = min(w, img.shape[1] - x)
        h = min(h, img.shape[0] - y)
        
        face = img[y:y+h, x:x+w]
        face = cv2.resize(face, required_size)
        
        # Create face filename
        face_filename = os.path.join(out_dir, os.path.basename(image_path))
        cv2.imwrite(face_filename, face)
        metadata.append([face_filename, label])
        
        print(f"✅ Face extracted: {os.path.basename(image_path)}")
        return face_filename, False
    else:
        # No face detected
        print(f"👤❌ No face detected: {os.path.basename(image_path)}")
        
        if delete_no_face:
            try:
                os.remove(image_path)
                print(f"🗑️ Deleted frame with no face: {os.path.basename(image_path)}")
                return None, True  # Return True for deleted
            except Exception as e:
                print(f"❌ Failed to delete {image_path}: {e}")
                return None, False
        else:
            return None, False

def process_category_faces(category, num_frames, delete_no_face, progress):
    """Process face extraction for a specific category"""
    
    # Load frame metadata
    metadata_file = os.path.join(FRAME_PATH, "metadata.csv")
    if not os.path.exists(metadata_file):
        print(f"❌ Frame metadata not found: {metadata_file}")
        print("Please run frame extraction first!")
        return 0, 0, 0
    
    df = pd.read_csv(metadata_file)
    category_frames = df[df['label'] == category]
    
    if len(category_frames) == 0:
        print(f"❌ No {category} frames found in metadata")
        return 0, 0, 0
    
    # Filter out already processed frames
    processed_frames = set(progress[category]["processed_frames"])
    unprocessed_frames = category_frames[~category_frames['filename'].apply(
        lambda x: os.path.basename(x)).isin(processed_frames)]
    
    print(f"\n👤 {category.upper()} FACE EXTRACTION:")
    print(f"📁 Total {category} frames available: {len(category_frames)}")
    print(f"✅ Already processed: {progress[category]['total_processed']} frames")
    print(f"📋 Remaining: {len(unprocessed_frames)} frames")
    
    if len(unprocessed_frames) == 0:
        print(f"🎉 All {category} frames have already been processed!")
        return 0, 0, 0
    
    # Limit frames to process if specified
    if num_frames:
        frames_to_process = unprocessed_frames.head(num_frames)
    else:
        frames_to_process = unprocessed_frames
    
    if len(frames_to_process) == 0:
        print(f"⚠️ No {category} frames to process")
        return 0, 0, 0
    
    print(f"\n🚀 Processing {len(frames_to_process)} {category} frames...")
    print("=" * 50)
    
    out_dir = os.path.join(FACE_PATH, category)
    faces_extracted = 0
    frames_deleted = 0
    processed_count = 0
    
    for _, row in frames_to_process.iterrows():
        image_path = row['filename']
        frame_name = os.path.basename(image_path)
        
        print(f"\n📸 Processing: {frame_name}")
        
        # Check if file still exists (might have been deleted in previous run)
        if not os.path.exists(image_path):
            print(f"⚠️ Frame no longer exists: {frame_name}")
            progress[category]["processed_frames"].append(frame_name)
            progress[category]["total_processed"] += 1
            processed_count += 1
            continue
        
        # Extract face
        face_path, was_deleted = extract_face(image_path, category, out_dir, delete_no_face, REQUIRED_SIZE)
        
        if face_path:
            faces_extracted += 1
            progress[category]["faces_extracted"] += 1
        
        if was_deleted:
            frames_deleted += 1  
            progress[category]["frames_deleted"] += 1
        
        # Update progress
        progress[category]["processed_frames"].append(frame_name)
        progress[category]["total_processed"] += 1
        processed_count += 1
        
        # Save progress after every 10 frames
        if processed_count % 10 == 0:
            save_face_progress(progress)
            print(f"💾 Progress saved: {processed_count}/{len(frames_to_process)} frames")
        
        print(f"✅ Progress: {processed_count}/{len(frames_to_process)} frames processed")
    
    # Save final progress
    save_face_progress(progress)
    
    return processed_count, faces_extracted, frames_deleted

def main():
    """Main face extraction process"""
    # Load existing progress
    progress = load_face_progress()
    
    # Display current status
    total_real_processed = progress["real"]["total_processed"]
    total_fake_processed = progress["fake"]["total_processed"]
    total_real_faces = progress["real"]["faces_extracted"]
    total_fake_faces = progress["fake"]["faces_extracted"]
    total_real_deleted = progress["real"]["frames_deleted"]
    total_fake_deleted = progress["fake"]["frames_deleted"]
    
    print("👤 FACE EXTRACTION STATUS:")
    print(f"🟢 Real frames processed: {total_real_processed} (Faces: {total_real_faces}, Deleted: {total_real_deleted})")
    print(f"🔴 Fake frames processed: {total_fake_processed} (Faces: {total_fake_faces}, Deleted: {total_fake_deleted})")
    if progress["last_updated"]:
        print(f"🕒 Last updated: {progress['last_updated']}")
    else:
        print("🆕 Starting fresh face extraction process")
    
    # Get user input
    category, delete_no_face, num_frames = get_user_input()
    
    total_processed = 0
    total_faces_extracted = 0
    total_frames_deleted = 0
    
    if category == "both":
        # Process both categories
        print(f"\n🎯 Processing {num_frames if num_frames else 'all remaining'} frames from EACH category")
        print(f"🗑️ Delete frames with no faces: {'YES' if delete_no_face else 'NO'}")
        
        real_processed, real_faces, real_deleted = process_category_faces("real", num_frames, delete_no_face, progress)
        fake_processed, fake_faces, fake_deleted = process_category_faces("fake", num_frames, delete_no_face, progress)
        
        total_processed = real_processed + fake_processed
        total_faces_extracted = real_faces + fake_faces
        total_frames_deleted = real_deleted + fake_deleted
        
    else:
        # Process single category
        print(f"\n🎯 Processing {num_frames if num_frames else 'all remaining'} {category} frames")
        print(f"🗑️ Delete frames with no faces: {'YES' if delete_no_face else 'NO'}")
        
        total_processed, total_faces_extracted, total_frames_deleted = process_category_faces(
            category, num_frames, delete_no_face, progress)
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎉 FACE EXTRACTION COMPLETE!")
    print("=" * 60)
    print(f"📊 Frames processed this session: {total_processed}")
    print(f"👤 Faces extracted this session: {total_faces_extracted}")
    print(f"🗑️ Frames deleted this session: {total_frames_deleted}")
    print(f"🟢 Total real: Processed={progress['real']['total_processed']}, Faces={progress['real']['faces_extracted']}, Deleted={progress['real']['frames_deleted']}")
    print(f"🔴 Total fake: Processed={progress['fake']['total_processed']}, Faces={progress['fake']['faces_extracted']}, Deleted={progress['fake']['frames_deleted']}")
    
    if not metadata:
        print("📂 No new faces extracted")
        return
    
    # Load existing face metadata if it exists
    face_metadata_file = os.path.join(FACE_PATH, "metadata_faces.csv")
    if os.path.exists(face_metadata_file):
        existing_df = pd.read_csv(face_metadata_file)
        new_df = pd.DataFrame(metadata, columns=["filename", "label"])
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = pd.DataFrame(metadata, columns=["filename", "label"])
    
    # Save face metadata as CSV
    combined_df.to_csv(face_metadata_file, index=False)
    print(f"📂 Face metadata updated at: {face_metadata_file}")
    print(f"👤 Total faces in dataset: {len(combined_df)}")
    print(f"✅ Faces saved at: {FACE_PATH}")

if __name__ == "__main__":
    main()