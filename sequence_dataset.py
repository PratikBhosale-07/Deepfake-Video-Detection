import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from collections import defaultdict
import json
from datetime import datetime

class VideoSequenceDataset(Dataset):
    def __init__(self, metadata_csv, transform=None, seq_len=10, progress_file=None):
        """
        Video Sequence Dataset with pause/resume capability
        
        Args:
            metadata_csv: Path to metadata CSV file
            transform: Transform to apply to images
            seq_len: Number of frames per sequence
            progress_file: Path to progress file for pause/resume functionality
        """
        self.df = pd.read_csv(metadata_csv)
        self.transform = transform
        self.seq_len = seq_len
        self.video_dict = defaultdict(list)
        self.progress_file = progress_file or "sequence_processing_progress.json"
        
        # Load progress if it exists
        self.progress = self._load_progress()

        # Group frames by video
        for _, row in self.df.iterrows():
            # Extract just the filename from the path
            filename = os.path.basename(row["filename"])
            # Split by _frame to get video ID (handle both _frame and _frame_ patterns)
            parts = filename.split("_frame")
            if len(parts) > 1:
                video_id = parts[0]
                self.video_dict[video_id].append((row["filename"], row["label"]))

        self.videos = list(self.video_dict.keys())
        
        # Filter out already processed videos if resuming
        if self.progress["processed_videos"]:
            processed_set = set(self.progress["processed_videos"])
            self.videos = [v for v in self.videos if v not in processed_set]
            print(f"✅ Resuming from previous session: {self.progress['total_processed']} videos already processed")
            print(f"📋 Remaining videos: {len(self.videos)}")

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_key = self.videos[idx]
        frames = self.video_dict[video_key]

        # Sort frames by frame index
        def extract_frame_number(frame_path):
            try:
                filename = os.path.basename(frame_path)
                # Split by _frame and get the number after it
                parts = filename.split("_frame")
                if len(parts) > 1:
                    # Extract number from the second part (e.g., "62.jpg" -> 62)
                    num_str = parts[1].split(".")[0]
                    # Handle cases like "62" or "_62" or "62_something"
                    return int(''.join(filter(str.isdigit, num_str)))
            except (ValueError, IndexError):
                pass
            return 0
        
        frames.sort(key=lambda x: extract_frame_number(x[0]))

        # Ensure we have exactly seq_len frames
        if len(frames) < self.seq_len:
            # Pad by repeating the last frame
            while len(frames) < self.seq_len:
                frames.append(frames[-1])
        else:
            # Take only the first seq_len frames
            frames = frames[:self.seq_len]

        images = []
        for img_path, _ in frames:
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            images.append(img)

        images = torch.stack(images)
        label = 0 if frames[0][1] == "real" else 1

        return images, label
    
    def _load_progress(self):
        """Load progress from JSON file"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️ Could not read progress file, starting fresh")
                return self._init_progress()
        return self._init_progress()
    
    def _init_progress(self):
        """Initialize progress structure"""
        return {
            "processed_videos": [],
            "total_processed": 0,
            "last_updated": None
        }
    
    def save_progress(self, video_key):
        """Save progress after processing a video"""
        self.progress["processed_videos"].append(video_key)
        self.progress["total_processed"] += 1
        self.progress["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def reset_progress(self):
        """Reset progress to start from beginning"""
        self.progress = self._init_progress()
        if os.path.exists(self.progress_file):
            os.remove(self.progress_file)
        self.videos = list(self.video_dict.keys())
        print("🔄 Progress reset, starting from beginning")
    
    def get_progress_stats(self):
        """Get current progress statistics"""
        return {
            "total_videos": len(self.videos) + self.progress["total_processed"],
            "processed": self.progress["total_processed"],
            "remaining": len(self.videos),
            "last_updated": self.progress["last_updated"]
        }
