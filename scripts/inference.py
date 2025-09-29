#!/usr/bin/env python3
"""
Inference script for MRP-CSLR model
Demonstrates how to use the trained model for sign language recognition
"""
import os
import sys
import argparse
import torch
import numpy as np
from typing import List, Dict
import cv2

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_mrp_cslr_model
from utils.data_processing import VideoTransform, VideoLoader, SignLanguageVocabulary
from configs.config import get_config


class MRPCSLRInference:
    """
    Inference class for MRP-CSLR model
    """
    def __init__(self, 
                 model_path: str,
                 config_name: str = 'default',
                 vocab_file: str = None,
                 device: str = None):
        
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load configuration
        self.config = get_config(config_name)
        
        # Setup vocabulary
        self.vocab = SignLanguageVocabulary(vocab_file)
        
        # Setup transforms
        self.transform = VideoTransform(
            input_size=self.config.data.input_size,
            mean=self.config.data.mean,
            std=self.config.data.std,
            augment=False
        )
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        print(f"MRP-CSLR inference initialized on {self.device}")
        print(f"Vocabulary size: {len(self.vocab)}")
    
    def _load_model(self, model_path: str):
        """Load trained model from checkpoint"""
        # Create model
        model = build_mrp_cslr_model(
            vocab_size=self.config.model.vocab_size,
            visual_backbone=self.config.model.visual_backbone,
            feature_dim=self.config.model.feature_dim,
            embed_dim=self.config.model.embed_dim,
            num_heads=self.config.model.num_heads,
            num_prompts=self.config.model.num_prompts,
            dropout=self.config.model.dropout,
            use_ctc_loss=self.config.model.use_ctc_loss
        )
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        print(f"Model loaded from: {model_path}")
        return model
    
    def predict_video(self, video_path: str) -> Dict:
        """
        Predict sign language sequence from video file
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary containing predictions and confidence scores
        """
        # Load video
        print(f"Loading video: {video_path}")
        frames = VideoLoader.load_video(
            video_path, 
            max_frames=self.config.data.max_sequence_length,
            fps=self.config.data.fps
        )
        
        if len(frames) == 0:
            raise ValueError(f"Could not load video: {video_path}")
        
        print(f"Loaded {len(frames)} frames")
        
        # Transform frames
        transformed_frames = self.transform(frames)  # (T, C, H, W)
        
        # Add batch dimension
        video_tensor = transformed_frames.unsqueeze(0).to(self.device)  # (1, T, C, H, W)
        input_lengths = torch.tensor([len(frames)]).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(
                video_frames=video_tensor,
                input_lengths=input_lengths
            )
        
        # Decode predictions
        predictions = self.model.decode_predictions(outputs['log_probs'], input_lengths)
        predicted_sequence = predictions[0]
        
        # Convert to tokens
        predicted_tokens = self.vocab.decode(predicted_sequence)
        
        # Get confidence scores
        log_probs = outputs['log_probs'][0].cpu().numpy()  # (T, vocab_size)
        confidence_scores = np.exp(np.max(log_probs, axis=1))  # Max probability at each timestep
        avg_confidence = np.mean(confidence_scores)
        
        return {
            'predicted_ids': predicted_sequence,
            'predicted_tokens': predicted_tokens,
            'confidence_scores': confidence_scores.tolist(),
            'average_confidence': float(avg_confidence),
            'num_frames': len(frames),
            'attention_weights': outputs['attention_info']['cross_attention'].cpu().numpy()
        }
    
    def predict_frames(self, frames: np.ndarray) -> Dict:
        """
        Predict sign language sequence from numpy frames
        
        Args:
            frames: (T, H, W, C) numpy array of video frames
            
        Returns:
            Dictionary containing predictions and confidence scores
        """
        print(f"Processing {len(frames)} frames")
        
        # Transform frames
        transformed_frames = self.transform(frames)  # (T, C, H, W)
        
        # Add batch dimension
        video_tensor = transformed_frames.unsqueeze(0).to(self.device)  # (1, T, C, H, W)
        input_lengths = torch.tensor([len(frames)]).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(
                video_frames=video_tensor,
                input_lengths=input_lengths
            )
        
        # Decode predictions
        predictions = self.model.decode_predictions(outputs['log_probs'], input_lengths)
        predicted_sequence = predictions[0]
        
        # Convert to tokens
        predicted_tokens = self.vocab.decode(predicted_sequence)
        
        # Get confidence scores
        log_probs = outputs['log_probs'][0].cpu().numpy()  # (T, vocab_size)
        confidence_scores = np.exp(np.max(log_probs, axis=1))
        avg_confidence = np.mean(confidence_scores)
        
        return {
            'predicted_ids': predicted_sequence,
            'predicted_tokens': predicted_tokens,
            'confidence_scores': confidence_scores.tolist(),
            'average_confidence': float(avg_confidence),
            'num_frames': len(frames)
        }
    
    def batch_predict(self, video_paths: List[str]) -> List[Dict]:
        """
        Batch prediction for multiple videos
        
        Args:
            video_paths: List of video file paths
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for video_path in video_paths:
            try:
                result = self.predict_video(video_path)
                result['video_path'] = video_path
                results.append(result)
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                results.append({
                    'video_path': video_path,
                    'error': str(e)
                })
        
        return results


def create_demo_video(output_path: str, duration: int = 5, fps: int = 25):
    """Create a dummy demo video for testing"""
    width, height = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame_idx in range(duration * fps):
        # Create a simple animated frame
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        # Add some moving elements to simulate sign language
        t = frame_idx / fps
        center_x = int(width // 2 + 50 * np.sin(2 * np.pi * t))
        center_y = int(height // 2 + 30 * np.cos(2 * np.pi * t))
        
        cv2.circle(frame, (center_x, center_y), 20, (255, 255, 255), -1)
        cv2.putText(frame, f'Frame {frame_idx}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"Demo video created: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='MRP-CSLR Inference')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--video_path', type=str, default=None,
                       help='Path to input video file')
    parser.add_argument('--config', type=str, default='default',
                       help='Configuration name')
    parser.add_argument('--vocab_file', type=str, default=None,
                       help='Path to vocabulary file')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--create_demo', action='store_true',
                       help='Create demo video for testing')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create demo video if requested
    if args.create_demo:
        demo_path = os.path.join(args.output_dir, 'demo_video.mp4')
        create_demo_video(demo_path)
        if args.video_path is None:
            args.video_path = demo_path
    
    # Check if model path exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model checkpoint not found: {args.model_path}")
        print("Please train a model first using scripts/train.py")
        return
    
    # Check video path
    if args.video_path is None:
        print("Error: Please provide --video_path or use --create_demo")
        return
    
    try:
        # Initialize inference
        inference = MRPCSLRInference(
            model_path=args.model_path,
            config_name=args.config,
            vocab_file=args.vocab_file,
            device=args.device
        )
        
        # Run inference
        print(f"Running inference on: {args.video_path}")
        result = inference.predict_video(args.video_path)
        
        # Print results
        print("\n=== PREDICTION RESULTS ===")
        print(f"Predicted sequence: {result['predicted_tokens']}")
        print(f"Predicted IDs: {result['predicted_ids']}")
        print(f"Average confidence: {result['average_confidence']:.4f}")
        print(f"Number of frames: {result['num_frames']}")
        
        # Save results
        import json
        output_file = os.path.join(args.output_dir, 'prediction_results.json')
        
        # Convert numpy arrays to lists for JSON serialization
        json_result = {
            'predicted_tokens': result['predicted_tokens'],
            'predicted_ids': result['predicted_ids'],
            'average_confidence': result['average_confidence'],
            'num_frames': result['num_frames'],
            'confidence_scores': result['confidence_scores']
        }
        
        with open(output_file, 'w') as f:
            json.dump(json_result, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()