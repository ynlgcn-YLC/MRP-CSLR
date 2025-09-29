"""
Data preprocessing utilities for Sign Language Recognition
Handles video loading, preprocessing, and augmentation
"""
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import random


class VideoTransform:
    """
    Video preprocessing and augmentation pipeline
    """
    def __init__(self, 
                 input_size: Tuple[int, int] = (224, 224),
                 mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                 std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
                 augment: bool = True):
        
        self.input_size = input_size
        self.augment = augment
        
        # Base transforms
        self.base_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        # Augmentation transforms
        if augment:
            self.augment_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((int(input_size[0] * 1.1), int(input_size[1] * 1.1))),
                transforms.RandomCrop(input_size),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
    
    def __call__(self, frames: np.ndarray) -> torch.Tensor:
        """
        Transform video frames
        
        Args:
            frames: (T, H, W, C) - video frames in numpy format
            
        Returns:
            transformed_frames: (T, C, H, W) - transformed frames
        """
        transformed = []
        
        for frame in frames:
            # Convert BGR to RGB if necessary
            if frame.shape[-1] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Apply transforms
            if self.augment and random.random() > 0.5:
                transformed_frame = self.augment_transform(frame)
            else:
                transformed_frame = self.base_transform(frame)
                
            transformed.append(transformed_frame)
        
        return torch.stack(transformed)  # (T, C, H, W)


class VideoLoader:
    """
    Video loading utilities for sign language videos
    """
    @staticmethod
    def load_video(video_path: str, 
                   max_frames: Optional[int] = None,
                   fps: Optional[int] = None) -> np.ndarray:
        """
        Load video frames from file
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to load
            fps: Target FPS (for frame sampling)
            
        Returns:
            frames: (T, H, W, C) - video frames
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_skip = int(original_fps / fps) if fps else 1
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_skip == 0:
                frames.append(frame)
                
            frame_count += 1
            
            if max_frames and len(frames) >= max_frames:
                break
        
        cap.release()
        return np.array(frames)  # (T, H, W, C)
    
    @staticmethod
    def temporal_sampling(frames: np.ndarray, 
                         target_length: int, 
                         sampling_strategy: str = 'uniform') -> np.ndarray:
        """
        Sample frames to target length
        
        Args:
            frames: (T, H, W, C) - input frames
            target_length: Target number of frames
            sampling_strategy: 'uniform', 'random', or 'center'
            
        Returns:
            sampled_frames: (target_length, H, W, C)
        """
        T = frames.shape[0]
        
        if T == target_length:
            return frames
        elif T < target_length:
            # Repeat frames if too short
            indices = np.linspace(0, T-1, target_length).astype(int)
            return frames[indices]
        else:
            # Sample frames if too long
            if sampling_strategy == 'uniform':
                indices = np.linspace(0, T-1, target_length).astype(int)
            elif sampling_strategy == 'random':
                indices = sorted(np.random.choice(T, target_length, replace=False))
            elif sampling_strategy == 'center':
                start = (T - target_length) // 2
                indices = np.arange(start, start + target_length)
            else:
                raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")
                
            return frames[indices]


class SignLanguageVocabulary:
    """
    Sign language vocabulary management
    """
    def __init__(self, vocab_file: Optional[str] = None):
        # Special tokens
        self.blank_token = '<blank>'
        self.unk_token = '<unk>'
        self.pad_token = '<pad>'
        
        # Initialize vocabulary
        self.token_to_id = {
            self.blank_token: 0,
            self.pad_token: 1,
            self.unk_token: 2
        }
        
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        
        if vocab_file:
            self.load_vocabulary(vocab_file)
    
    def load_vocabulary(self, vocab_file: str):
        """Load vocabulary from file"""
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for line in f:
                token = line.strip()
                if token and token not in self.token_to_id:
                    token_id = len(self.token_to_id)
                    self.token_to_id[token] = token_id
                    self.id_to_token[token_id] = token
    
    def encode(self, tokens: List[str]) -> List[int]:
        """Encode tokens to IDs"""
        return [self.token_to_id.get(token, self.token_to_id[self.unk_token]) 
                for token in tokens]
    
    def decode(self, token_ids: List[int]) -> List[str]:
        """Decode IDs to tokens"""
        return [self.id_to_token.get(token_id, self.unk_token) 
                for token_id in token_ids]
    
    def __len__(self):
        return len(self.token_to_id)
    
    @property
    def vocab_size(self):
        return len(self.token_to_id)


class DataCollator:
    """
    Data collation for batching variable-length sequences
    """
    def __init__(self, pad_token_id: int = 1):
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of samples
        
        Args:
            batch: List of samples, each containing:
                - 'video': (T, C, H, W) tensor
                - 'labels': List[int] or tensor
                - 'video_id': str (optional)
        
        Returns:
            Batched data dictionary
        """
        videos = [sample['video'] for sample in batch]
        labels = [sample['labels'] for sample in batch]
        
        # Pad videos to same length
        max_video_length = max(v.shape[0] for v in videos)
        padded_videos = []
        video_lengths = []
        
        for video in videos:
            T, C, H, W = video.shape
            video_lengths.append(T)
            
            if T < max_video_length:
                padding = torch.zeros(max_video_length - T, C, H, W)
                padded_video = torch.cat([video, padding], dim=0)
            else:
                padded_video = video
                
            padded_videos.append(padded_video)
        
        # Pad labels
        if labels[0] is not None:
            max_label_length = max(len(label) for label in labels)
            padded_labels = []
            label_lengths = []
            
            for label in labels:
                label_lengths.append(len(label))
                if len(label) < max_label_length:
                    padded_label = label + [self.pad_token_id] * (max_label_length - len(label))
                else:
                    padded_label = label
                padded_labels.append(padded_label)
                
            return {
                'videos': torch.stack(padded_videos),  # (B, T, C, H, W)
                'labels': torch.tensor(padded_labels),  # (B, max_label_length)
                'video_lengths': torch.tensor(video_lengths),
                'label_lengths': torch.tensor(label_lengths)
            }
        else:
            return {
                'videos': torch.stack(padded_videos),  # (B, T, C, H, W)
                'video_lengths': torch.tensor(video_lengths)
            }


if __name__ == "__main__":
    # Test data processing utilities
    
    # Test video transform
    print("Testing VideoTransform...")
    transform = VideoTransform(augment=True)
    dummy_frames = np.random.randint(0, 255, (10, 240, 320, 3), dtype=np.uint8)
    transformed = transform(dummy_frames)
    print(f"Input shape: {dummy_frames.shape}, Output shape: {transformed.shape}")
    
    # Test vocabulary
    print("\nTesting SignLanguageVocabulary...")
    vocab = SignLanguageVocabulary()
    vocab.token_to_id.update({f'sign_{i}': i+3 for i in range(100)})
    vocab.id_to_token.update({i+3: f'sign_{i}' for i in range(100)})
    
    test_tokens = ['sign_1', 'sign_5', 'unknown_sign']
    encoded = vocab.encode(test_tokens)
    decoded = vocab.decode(encoded)
    print(f"Original: {test_tokens}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    print(f"Vocabulary size: {vocab.vocab_size}")
    
    # Test data collator
    print("\nTesting DataCollator...")
    collator = DataCollator()
    
    batch = [
        {'video': torch.randn(15, 3, 224, 224), 'labels': [1, 5, 10, 3]},
        {'video': torch.randn(12, 3, 224, 224), 'labels': [2, 8, 15]}
    ]
    
    collated = collator(batch)
    print(f"Batch video shape: {collated['videos'].shape}")
    print(f"Batch labels shape: {collated['labels'].shape}")
    print(f"Video lengths: {collated['video_lengths']}")
    print(f"Label lengths: {collated['label_lengths']}")
    
    print("All data processing tests passed!")