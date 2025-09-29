"""
MRP-CSLR: A Multi-Representation Prompt–Guided Framework for Continuous Sign Language Recognition
Main model architecture combining multi-representation encoding and prompt-guided attention
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .multi_representation_encoder import MultiRepresentationEncoder
from .prompt_guided_attention import PromptGuidedAttention


class CTC_Loss_Module(nn.Module):
    """
    CTC Loss wrapper for sequence-to-sequence learning without alignment
    """
    def __init__(self):
        super(CTC_Loss_Module, self).__init__()
        self.ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        
    def forward(self, log_probs, targets, input_lengths, target_lengths):
        """
        Args:
            log_probs: (T, B, vocab_size) - log probabilities
            targets: (B * target_lengths) - concatenated target sequences
            input_lengths: (B,) - lengths of input sequences
            target_lengths: (B,) - lengths of target sequences
        """
        return self.ctc_loss(log_probs, targets, input_lengths, target_lengths)


class SequenceClassifier(nn.Module):
    """
    Sequence classification head for continuous sign language recognition
    """
    def __init__(self, input_dim, vocab_size, hidden_dim=512, dropout=0.3):
        super(SequenceClassifier, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, vocab_size)
        )
        
    def forward(self, features):
        """
        Args:
            features: (B, T, input_dim) - sequence features
        Returns:
            logits: (B, T, vocab_size) - classification logits
        """
        return self.classifier(features)


class MRP_CSLR(nn.Module):
    """
    Main MRP-CSLR model: Multi-Representation Prompt–Guided Framework 
    for Continuous Sign Language Recognition
    """
    def __init__(self, 
                 vocab_size: int,
                 visual_backbone: str = 'resnet50',
                 feature_dim: int = 2048,
                 embed_dim: int = 512,
                 num_heads: int = 8,
                 num_prompts: int = 100,
                 dropout: float = 0.1,
                 use_ctc_loss: bool = True):
        super(MRP_CSLR, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.use_ctc_loss = use_ctc_loss
        
        # Multi-representation encoder
        self.multi_repr_encoder = MultiRepresentationEncoder(
            visual_backbone=visual_backbone,
            feature_dim=feature_dim,
            fusion_dim=embed_dim
        )
        
        # Prompt-guided attention mechanism
        self.prompt_guided_attention = PromptGuidedAttention(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_prompts=num_prompts,
            dropout=dropout
        )
        
        # Sequence classifier
        self.classifier = SequenceClassifier(
            input_dim=embed_dim,
            vocab_size=vocab_size,
            dropout=dropout
        )
        
        # Loss modules
        if use_ctc_loss:
            self.ctc_loss = CTC_Loss_Module()
        else:
            self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
            
        # Additional components for enhanced learning
        self.temporal_smoothing = nn.Conv1d(
            embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim
        )
        self.feature_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, 
                video_frames: torch.Tensor,
                target_classes: Optional[torch.Tensor] = None,
                input_lengths: Optional[torch.Tensor] = None,
                target_lengths: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of MRP-CSLR model
        
        Args:
            video_frames: (B, T, C, H, W) - input video sequences
            target_classes: (B, T) or (sum(target_lengths),) - target sequences
            input_lengths: (B,) - lengths of input sequences
            target_lengths: (B,) - lengths of target sequences
            
        Returns:
            Dict containing:
                - logits: (B, T, vocab_size) - classification logits
                - log_probs: (T, B, vocab_size) - log probabilities for CTC
                - loss: scalar loss value (if targets provided)
                - attention_info: attention weights and prompts
        """
        batch_size, seq_len = video_frames.shape[:2]
        
        # Step 1: Multi-representation encoding
        multi_repr_features, repr_components = self.multi_repr_encoder(video_frames)
        
        # Step 2: Create feature mask for variable-length sequences
        if input_lengths is not None:
            feature_mask = self._create_mask(batch_size, seq_len, input_lengths)
        else:
            feature_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=video_frames.device)
        
        # Step 3: Prompt-guided attention
        enhanced_features, attention_info = self.prompt_guided_attention(
            multi_repr_features, target_classes, feature_mask
        )
        
        # Step 4: Temporal smoothing
        smoothed_features = enhanced_features.transpose(1, 2)  # (B, embed_dim, T)
        smoothed_features = self.temporal_smoothing(smoothed_features)
        smoothed_features = smoothed_features.transpose(1, 2)  # (B, T, embed_dim)
        
        # Step 5: Feature normalization
        final_features = self.feature_norm(smoothed_features + enhanced_features)
        
        # Step 6: Classification
        logits = self.classifier(final_features)  # (B, T, vocab_size)
        log_probs = F.log_softmax(logits, dim=-1)  # (B, T, vocab_size)
        
        # Prepare output
        output = {
            'logits': logits,
            'log_probs': log_probs.transpose(0, 1),  # (T, B, vocab_size) for CTC
            'features': final_features,
            'multi_repr_components': repr_components,
            'attention_info': attention_info
        }
        
        # Compute loss if targets are provided
        if target_classes is not None:
            loss = self._compute_loss(
                log_probs.transpose(0, 1), target_classes, 
                input_lengths, target_lengths
            )
            output['loss'] = loss
            
        return output
    
    def _create_mask(self, batch_size: int, seq_len: int, lengths: torch.Tensor) -> torch.Tensor:
        """Create mask for variable-length sequences"""
        mask = torch.arange(seq_len, device=lengths.device).expand(batch_size, seq_len)
        return mask < lengths.unsqueeze(1)
    
    def _compute_loss(self, 
                     log_probs: torch.Tensor, 
                     targets: torch.Tensor,
                     input_lengths: Optional[torch.Tensor] = None,
                     target_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute loss based on the chosen loss function"""
        if self.use_ctc_loss:
            # CTC loss expects flattened targets
            if targets.dim() == 2:  # (B, T) -> flatten
                targets = targets.view(-1)
                if target_lengths is None:
                    target_lengths = torch.full((log_probs.shape[1],), targets.shape[0] // log_probs.shape[1])
            
            if input_lengths is None:
                input_lengths = torch.full((log_probs.shape[1],), log_probs.shape[0])
                
            return self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        else:
            # Cross-entropy loss
            logits = log_probs.transpose(0, 1)  # (B, T, vocab_size)
            return self.ce_loss(logits.reshape(-1, self.vocab_size), targets.view(-1))
    
    def decode_predictions(self, log_probs: torch.Tensor, input_lengths: Optional[torch.Tensor] = None) -> list:
        """
        Decode predictions using greedy decoding or beam search
        
        Args:
            log_probs: (T, B, vocab_size) - log probabilities
            input_lengths: (B,) - sequence lengths
            
        Returns:
            List of predicted sequences
        """
        # Simple greedy decoding
        predictions = []
        probs = log_probs.transpose(0, 1)  # (B, T, vocab_size)
        
        for i in range(probs.shape[0]):
            seq_len = input_lengths[i] if input_lengths is not None else probs.shape[1]
            pred_seq = probs[i, :seq_len].argmax(dim=-1)  # (T,)
            
            # Remove blanks and consecutive duplicates (basic CTC decoding)
            decoded = []
            prev_token = -1
            for token in pred_seq:
                if token != 0 and token != prev_token:  # 0 is blank token
                    decoded.append(token.item())
                prev_token = token
                
            predictions.append(decoded)
            
        return predictions


def build_mrp_cslr_model(vocab_size: int, **kwargs) -> MRP_CSLR:
    """
    Factory function to build MRP-CSLR model with default parameters
    """
    default_config = {
        'visual_backbone': 'resnet50',
        'feature_dim': 2048,
        'embed_dim': 512,
        'num_heads': 8,
        'num_prompts': 100,
        'dropout': 0.1,
        'use_ctc_loss': True
    }
    
    # Update with provided kwargs
    config = {**default_config, **kwargs}
    
    return MRP_CSLR(vocab_size=vocab_size, **config)


if __name__ == "__main__":
    # Test the complete MRP-CSLR model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model parameters
    vocab_size = 1000
    batch_size, seq_len, channels, height, width = 2, 16, 3, 224, 224
    
    # Create model
    model = build_mrp_cslr_model(vocab_size).to(device)
    
    # Create dummy data
    video_frames = torch.randn(batch_size, seq_len, channels, height, width).to(device)
    target_classes = torch.randint(1, vocab_size, (batch_size, seq_len)).to(device)
    input_lengths = torch.tensor([seq_len, seq_len-2]).to(device)
    target_lengths = torch.tensor([seq_len-3, seq_len-5]).to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(video_frames, target_classes, input_lengths, target_lengths)
        
    print("=== MRP-CSLR Model Test ===")
    print(f"Input video shape: {video_frames.shape}")
    print(f"Output logits shape: {outputs['logits'].shape}")
    print(f"Log probabilities shape: {outputs['log_probs'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test decoding
    predictions = model.decode_predictions(outputs['log_probs'], input_lengths)
    print(f"Decoded predictions: {predictions}")
    
    print("MRP-CSLR model test passed!")