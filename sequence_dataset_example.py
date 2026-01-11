"""
Video Sequence Dataset Processor

This script processes video frame sequences with pause/resume functionality.
It features a menu-driven interface for configuring processing parameters.

Features:
- Interactive menu for dataset split selection (train/val/test)
- Configurable sequence length, batch size, and save frequency
- Automatic progress tracking and resume capability
- Graceful pause with Ctrl+C
- Separate progress files for each dataset split

Usage:
    python sequence_dataset_example.py
    
    Follow the menu prompts to:
    1. Select dataset split (train/val/test)
    2. Configure sequence length (frames per video)
    3. Set batch size (sequences per batch)
    4. Set save frequency (batches between progress saves)
    5. Resume from previous session or start fresh
    
Author: Deepfake Detection Project
"""

import torch
from torch.utils.data import DataLoader
from sequence_dataset import VideoSequenceDataset
from torchvision import transforms
import signal
import sys

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Global flag for graceful shutdown
should_stop = False

def signal_handler(sig, frame):
    """Handle Ctrl+C to pause gracefully"""
    global should_stop
    print("\n\n⏸️  Pause signal received! Finishing current batch and saving progress...")
    should_stop = True

# Register signal handler for Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

def process_sequences(dataset, batch_size=8, save_every=10):
    """
    Process video sequences with automatic progress saving
    
    Args:
        dataset: VideoSequenceDataset instance
        batch_size: Number of sequences per batch
        save_every: Save progress every N batches
    """
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Display initial stats
    stats = dataset.get_progress_stats()
    print("\n" + "="*60)
    print("📊 SEQUENCE PROCESSING STATUS")
    print("="*60)
    print(f"📹 Total videos: {stats['total_videos']}")
    print(f"✅ Already processed: {stats['processed']}")
    print(f"📋 Remaining: {stats['remaining']}")
    if stats['last_updated']:
        print(f"🕒 Last updated: {stats['last_updated']}")
    print("="*60)
    print("\n💡 Press Ctrl+C to pause and save progress at any time\n")
    
    batch_count = 0
    processed_count = 0
    
    try:
        for batch_idx, (sequences, labels) in enumerate(dataloader):
            if should_stop:
                print("\n⏸️  Pausing after current batch...")
                break
            
            # Process batch (replace this with your actual processing logic)
            # For example: predictions = model(sequences)
            
            # Get video keys for this batch
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(dataset))
            batch_video_keys = dataset.videos[start_idx:end_idx]
            
            # Save progress for each video in batch
            for video_key in batch_video_keys:
                dataset.save_progress(video_key)
                processed_count += 1
            
            batch_count += 1
            
            # Progress update
            if batch_count % save_every == 0:
                print(f"💾 Progress saved: {processed_count}/{stats['remaining']} videos processed")
            
            print(f"✅ Batch {batch_count}: Processed {len(batch_video_keys)} sequences "
                  f"(Total: {processed_count}/{stats['remaining']})")
        
        # Final summary
        print("\n" + "="*60)
        if should_stop:
            print("⏸️  PROCESSING PAUSED")
            print(f"✅ Processed: {processed_count} videos this session")
            print(f"📋 Remaining: {len(dataset.videos) - processed_count} videos")
            print("\n💡 Run this script again to resume from where you left off")
        else:
            print("🎉 PROCESSING COMPLETE!")
            print(f"✅ Total processed: {stats['processed'] + processed_count} videos")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        print(f"💾 Progress saved up to: {processed_count} videos")
        raise

def get_user_config():
    """Get configuration from user through menu"""
    print("\n" + "="*60)
    print("⚙️  CONFIGURATION")
    print("="*60)
    
    # Select dataset split
    while True:
        print("\n📊 Which dataset split would you like to process?")
        print("1. Train set (metadata_train.csv)")
        print("2. Validation set (metadata_val.csv)")
        print("3. Test set (metadata_test.csv)")
        
        choice = input("\nEnter your choice (1/2/3): ").strip()
        
        if choice == "1":
            metadata_csv = "dataset/processed/metadata_train.csv"
            split_name = "train"
            break
        elif choice == "2":
            metadata_csv = "dataset/processed/metadata_val.csv"
            split_name = "val"
            break
        elif choice == "3":
            metadata_csv = "dataset/processed/metadata_test.csv"
            split_name = "test"
            break
        else:
            print("❌ Please enter 1, 2, or 3.")
    
    # Get sequence length
    while True:
        try:
            seq_len = int(input("\n🎬 Sequence length (number of frames per video, default 10): ").strip() or "10")
            if seq_len > 0:
                break
            else:
                print("❌ Sequence length must be positive.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    # Get batch size
    while True:
        try:
            batch_size = int(input("\n📦 Batch size (number of sequences per batch, default 8): ").strip() or "8")
            if batch_size > 0:
                break
            else:
                print("❌ Batch size must be positive.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    # Get save frequency
    while True:
        try:
            save_every = int(input("\n💾 Save progress every N batches (default 10): ").strip() or "10")
            if save_every > 0:
                break
            else:
                print("❌ Save frequency must be positive.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    progress_file = f"sequence_processing_progress_{split_name}.json"
    
    return metadata_csv, seq_len, batch_size, save_every, progress_file, split_name

def main():
    """Main function to demonstrate pausable/resumable processing"""
    
    print("\n" + "="*60)
    print("🎬 VIDEO SEQUENCE DATASET PROCESSOR")
    print("="*60)
    print("📋 This tool processes video sequences with pause/resume support")
    print("💡 Press Ctrl+C at any time to pause and save progress")
    print("="*60)
    
    # Get configuration from user
    metadata_csv, seq_len, batch_size, save_every, progress_file, split_name = get_user_config()
    
    # Check if metadata file exists
    import os
    if not os.path.exists(metadata_csv):
        print(f"\n❌ Error: Metadata file not found: {metadata_csv}")
        print("Please ensure the dataset has been processed first.")
        return
    
    # Check if resuming or starting fresh
    if os.path.exists(progress_file):
        print(f"\n📁 Found existing progress file for {split_name} set")
        
        while True:
            print("\nWhat would you like to do?")
            print("1. Resume from previous session")
            print("2. Start fresh (reset progress)")
            print("3. Cancel")
            
            choice = input("\nEnter your choice (1/2/3): ").strip()
            
            if choice == "1":
                print("✅ Resuming from previous session")
                reset = False
                break
            elif choice == "2":
                confirm = input("⚠️  This will delete existing progress. Continue? (y/n): ").lower().strip()
                if confirm in ['y', 'yes']:
                    reset = True
                    break
                else:
                    print("❌ Cancelled. Exiting.")
                    return
            elif choice == "3":
                print("❌ Cancelled. Exiting.")
                return
            else:
                print("❌ Please enter 1, 2, or 3.")
        
        # Create dataset
        dataset = VideoSequenceDataset(
            metadata_csv=metadata_csv,
            transform=transform,
            seq_len=seq_len,
            progress_file=progress_file
        )
        
        if reset:
            dataset.reset_progress()
            print("🆕 Starting fresh processing session")
    else:
        print(f"\n🆕 Starting fresh processing session for {split_name} set")
        # Create dataset
        dataset = VideoSequenceDataset(
            metadata_csv=metadata_csv,
            transform=transform,
            seq_len=seq_len,
            progress_file=progress_file
        )
    
    # Display configuration summary
    print("\n" + "="*60)
    print("📝 PROCESSING CONFIGURATION")
    print("="*60)
    print(f"📊 Dataset split: {split_name}")
    print(f"📄 Metadata file: {metadata_csv}")
    print(f"🎬 Sequence length: {seq_len} frames")
    print(f"📦 Batch size: {batch_size} sequences")
    print(f"💾 Save frequency: every {save_every} batches")
    print(f"📁 Progress file: {progress_file}")
    print("="*60)
    
    # Confirm before processing
    confirm = input("\n▶️  Start processing? (y/n): ").lower().strip()
    if confirm not in ['y', 'yes']:
        print("❌ Processing cancelled.")
        return
    
    # Process sequences
    process_sequences(dataset, batch_size=batch_size, save_every=save_every)

if __name__ == "__main__":
    main()
