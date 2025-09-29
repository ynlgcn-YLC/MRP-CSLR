"""
Training utilities for MRP-CSLR model
Includes trainer class, metrics, and optimization utilities
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
import time
from tqdm import tqdm
import logging


class WarmupCosineScheduler:
    """
    Learning rate scheduler with warmup and cosine annealing
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Warmup phase
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing phase
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


class SignLanguageMetrics:
    """
    Metrics for sign language recognition evaluation
    """
    @staticmethod
    def sequence_accuracy(predictions: List[List[int]], 
                         targets: List[List[int]], 
                         ignore_blank=True) -> float:
        """
        Calculate sequence-level accuracy
        """
        correct = 0
        total = len(predictions)
        
        for pred, target in zip(predictions, targets):
            if ignore_blank:
                pred = [x for x in pred if x != 0]
                target = [x for x in target if x != 0]
            
            if pred == target:
                correct += 1
        
        return correct / total if total > 0 else 0.0
    
    @staticmethod
    def token_accuracy(predictions: List[List[int]], 
                      targets: List[List[int]], 
                      ignore_blank=True) -> float:
        """
        Calculate token-level accuracy
        """
        correct_tokens = 0
        total_tokens = 0
        
        for pred, target in zip(predictions, targets):
            if ignore_blank:
                pred = [x for x in pred if x != 0]
                target = [x for x in target if x != 0]
            
            # Align sequences for comparison
            min_len = min(len(pred), len(target))
            correct_tokens += sum(1 for i in range(min_len) if pred[i] == target[i])
            total_tokens += max(len(pred), len(target))
        
        return correct_tokens / total_tokens if total_tokens > 0 else 0.0
    
    @staticmethod
    def edit_distance(pred_seq: List[int], target_seq: List[int]) -> int:
        """
        Calculate edit distance (Levenshtein distance) between sequences
        """
        m, n = len(pred_seq), len(target_seq)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred_seq[i-1] == target_seq[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        return dp[m][n]


class MRPCSLRTrainer:
    """
    Main trainer class for MRP-CSLR model
    """
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: Dict):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup scheduler
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_epochs=config.get('warmup_epochs', 5),
            total_epochs=config.get('num_epochs', 100),
            base_lr=config.get('learning_rate', 1e-3)
        )
        
        # Setup logging
        self.logger = self._setup_logger()
        self.writer = SummaryWriter(config.get('log_dir', 'logs'))
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.train_step = 0
        self.val_step = 0
        
        # Metrics
        self.metrics = SignLanguageMetrics()
        
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer with different learning rates for different components"""
        # Group parameters by component
        backbone_params = []
        attention_params = []
        classifier_params = []
        
        for name, param in self.model.named_parameters():
            if 'visual_extractor.backbone' in name:
                backbone_params.append(param)
            elif 'prompt_guided_attention' in name:
                attention_params.append(param)
            else:
                classifier_params.append(param)
        
        # Different learning rates for different components
        param_groups = [
            {'params': backbone_params, 'lr': self.config.get('backbone_lr', 1e-4)},
            {'params': attention_params, 'lr': self.config.get('attention_lr', 1e-3)},
            {'params': classifier_params, 'lr': self.config.get('classifier_lr', 1e-3)}
        ]
        
        optimizer_type = self.config.get('optimizer', 'adamw')
        if optimizer_type == 'adamw':
            return optim.AdamW(param_groups, weight_decay=self.config.get('weight_decay', 1e-4))
        elif optimizer_type == 'adam':
            return optim.Adam(param_groups)
        else:
            return optim.SGD(param_groups, momentum=0.9)
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for training"""
        logger = logging.getLogger('MRP-CSLR')
        logger.setLevel(logging.INFO)
        
        # Create handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            videos = batch['videos'].to(self.device)  # (B, T, C, H, W)
            labels = batch['labels'].to(self.device)  # (B, max_seq_len)
            video_lengths = batch['video_lengths'].to(self.device)
            label_lengths = batch['label_lengths'].to(self.device)
            
            # Flatten labels for CTC loss
            flattened_labels = []
            for i, length in enumerate(label_lengths):
                flattened_labels.extend(labels[i, :length].tolist())
            flattened_labels = torch.tensor(flattened_labels, device=self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            outputs = self.model(
                video_frames=videos,
                target_classes=labels,
                input_lengths=video_lengths,
                target_lengths=label_lengths
            )
            
            loss = outputs['loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg_Loss': f'{total_loss / (batch_idx + 1):.4f}'
            })
            
            # Log to tensorboard
            if self.train_step % self.config.get('log_interval', 50) == 0:
                self.writer.add_scalar('Train/Loss', loss.item(), self.train_step)
                self.writer.add_scalar('Train/Learning_Rate', 
                                     self.optimizer.param_groups[0]['lr'], self.train_step)
            
            self.train_step += 1
        
        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Move data to device
                videos = batch['videos'].to(self.device)
                labels = batch['labels'].to(self.device)
                video_lengths = batch['video_lengths'].to(self.device)
                label_lengths = batch['label_lengths'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    video_frames=videos,
                    target_classes=None,  # No teacher forcing in validation
                    input_lengths=video_lengths
                )
                
                # Decode predictions
                predictions = self.model.decode_predictions(outputs['log_probs'], video_lengths)
                
                # Prepare targets
                targets = []
                for i, length in enumerate(label_lengths):
                    target_seq = labels[i, :length].cpu().tolist()
                    # Remove padding tokens
                    target_seq = [x for x in target_seq if x != 1]  # 1 is pad token
                    targets.append(target_seq)
                
                all_predictions.extend(predictions)
                all_targets.extend(targets)
        
        # Calculate metrics
        seq_acc = self.metrics.sequence_accuracy(all_predictions, all_targets)
        token_acc = self.metrics.token_accuracy(all_predictions, all_targets)
        
        # Calculate average edit distance
        edit_distances = [self.metrics.edit_distance(pred, target) 
                         for pred, target in zip(all_predictions, all_targets)]
        avg_edit_dist = np.mean(edit_distances)
        
        return {
            'sequence_accuracy': seq_acc,
            'token_accuracy': token_acc,
            'edit_distance': avg_edit_dist
        }
    
    def train(self):
        """Main training loop"""
        self.logger.info(f"Starting training for {self.config.get('num_epochs', 100)} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.config.get('num_epochs', 100)):
            self.current_epoch = epoch
            
            # Update learning rate
            current_lr = self.scheduler.step(epoch)
            
            # Train for one epoch
            train_metrics = self.train_epoch()
            
            # Validate
            if (epoch + 1) % self.config.get('val_interval', 5) == 0:
                val_metrics = self.validate()
                
                # Log validation metrics
                for key, value in val_metrics.items():
                    self.writer.add_scalar(f'Val/{key}', value, epoch)
                
                self.logger.info(
                    f"Epoch {epoch + 1}/{self.config.get('num_epochs', 100)} - "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Val Seq Acc: {val_metrics['sequence_accuracy']:.4f}, "
                    f"Val Token Acc: {val_metrics['token_accuracy']:.4f}, "
                    f"LR: {current_lr:.6f}"
                )
                
                # Save best model
                if val_metrics['sequence_accuracy'] > self.best_val_acc:
                    self.best_val_acc = val_metrics['sequence_accuracy']
                    self.save_checkpoint('best_model.pth')
                    self.logger.info(f"New best validation accuracy: {self.best_val_acc:.4f}")
            
            # Save regular checkpoint
            if (epoch + 1) % self.config.get('checkpoint_interval', 10) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')
        
        self.logger.info("Training completed!")
        self.writer.close()
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.config
        }
        
        os.makedirs(self.config.get('checkpoint_dir', 'checkpoints'), exist_ok=True)
        checkpoint_path = os.path.join(self.config.get('checkpoint_dir', 'checkpoints'), filename)
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint['best_val_acc']
        
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
        self.logger.info(f"Resumed from epoch {self.current_epoch + 1}, best val acc: {self.best_val_acc:.4f}")


if __name__ == "__main__":
    # Test training utilities
    print("Testing SignLanguageMetrics...")
    
    metrics = SignLanguageMetrics()
    
    # Test sequence accuracy
    predictions = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    targets = [[1, 2, 3], [4, 6], [6, 7, 8, 9]]
    
    seq_acc = metrics.sequence_accuracy(predictions, targets)
    token_acc = metrics.token_accuracy(predictions, targets)
    
    print(f"Sequence accuracy: {seq_acc:.4f}")
    print(f"Token accuracy: {token_acc:.4f}")
    
    # Test edit distance
    edit_dist = metrics.edit_distance([1, 2, 3, 4], [1, 3, 4, 5])
    print(f"Edit distance: {edit_dist}")
    
    print("Training utilities test passed!")