"""
Prompt-Guided Attention Mechanism for Sign Language Recognition
This module implements the prompt-guided framework for enhanced feature learning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math


class PromptEmbedding(nn.Module):
    """
    Learnable prompt embeddings for sign language recognition
    """
    def __init__(self, vocab_size, embed_dim=512, num_prompts=100):
        super(PromptEmbedding, self).__init__()
        
        # Learnable prompt embeddings
        self.prompt_embeddings = nn.Parameter(torch.randn(num_prompts, embed_dim))
        
        # Sign class embeddings (vocabulary)
        self.class_embeddings = nn.Embedding(vocab_size, embed_dim)
        
        # Context-aware prompt generation
        self.context_proj = nn.Linear(embed_dim, embed_dim)
        self.prompt_gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, num_prompts),
            nn.Softmax(dim=-1)
        )
        
        self.num_prompts = num_prompts
        self.embed_dim = embed_dim
        
    def forward(self, context_features, target_classes=None):
        """
        Generate context-aware prompts
        
        Args:
            context_features: (B, T, embed_dim) - contextual features
            target_classes: (B, T) - target class indices (for training)
        Returns:
            prompts: (B, T, embed_dim) - generated prompts
            prompt_weights: (B, T, num_prompts) - attention weights over prompts
        """
        B, T, _ = context_features.shape
        
        # Generate context-aware prompt weights
        context_proj = self.context_proj(context_features)  # (B, T, embed_dim)
        prompt_weights = self.prompt_gate(context_proj)  # (B, T, num_prompts)
        
        # Weight prompt embeddings by attention
        prompts = torch.einsum('btn,ne->bte', prompt_weights, self.prompt_embeddings)
        
        # If target classes are provided (training), incorporate class information
        if target_classes is not None:
            class_embeds = self.class_embeddings(target_classes)  # (B, T, embed_dim)
            prompts = prompts + 0.5 * class_embeds
            
        return prompts, prompt_weights


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention between visual features and prompts
    """
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super(CrossModalAttention, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, query, key, value, mask=None):
        """
        Cross-modal attention computation
        
        Args:
            query: (B, T_q, embed_dim) - query features (visual)
            key: (B, T_k, embed_dim) - key features (prompts)
            value: (B, T_v, embed_dim) - value features (prompts)
            mask: attention mask
        Returns:
            attended_features: (B, T_q, embed_dim)
            attention_weights: (B, num_heads, T_q, T_k)
        """
        B, T_q, _ = query.shape
        T_k = key.shape[1]
        
        # Project to Q, K, V
        Q = self.q_proj(query)  # (B, T_q, embed_dim)
        K = self.k_proj(key)    # (B, T_k, embed_dim)
        V = self.v_proj(value)  # (B, T_k, embed_dim)
        
        # Reshape for multi-head attention
        Q = Q.view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, T_q, head_dim)
        K = K.view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, T_k, head_dim)
        V = V.view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, T_k, head_dim)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, num_heads, T_q, T_k)
        
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)  # (B, num_heads, T_q, head_dim)
        attended = attended.transpose(1, 2).contiguous().view(B, T_q, self.embed_dim)
        
        # Output projection
        output = self.out_proj(attended)
        
        return output, attention_weights


class PromptGuidedAttention(nn.Module):
    """
    Main prompt-guided attention module
    """
    def __init__(self, vocab_size, embed_dim=512, num_heads=8, num_prompts=100, dropout=0.1):
        super(PromptGuidedAttention, self).__init__()
        
        # Prompt embedding module
        self.prompt_embedding = PromptEmbedding(vocab_size, embed_dim, num_prompts)
        
        # Cross-modal attention
        self.cross_attention = CrossModalAttention(embed_dim, num_heads, dropout)
        
        # Self-attention for enhanced feature learning
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
        
        # Feed-forward networks
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ln3 = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, visual_features, target_classes=None, feature_mask=None):
        """
        Apply prompt-guided attention to visual features
        
        Args:
            visual_features: (B, T, embed_dim) - visual features from encoder
            target_classes: (B, T) - target class indices (for training)
            feature_mask: (B, T) - mask for variable-length sequences
        Returns:
            enhanced_features: (B, T, embed_dim) - prompt-guided features
            attention_info: dict containing attention weights and prompts
        """
        # Generate context-aware prompts
        prompts, prompt_weights = self.prompt_embedding(visual_features, target_classes)
        
        # Cross-modal attention: visual features attend to prompts
        cross_attended, cross_attn_weights = self.cross_attention(
            query=visual_features,
            key=prompts,
            value=prompts,
            mask=None
        )
        
        # Residual connection and layer norm
        visual_features = self.ln1(visual_features + self.dropout(cross_attended))
        
        # Self-attention for enhanced feature learning
        self_attended, self_attn_weights = self.self_attention(
            visual_features, visual_features, visual_features,
            key_padding_mask=~feature_mask if feature_mask is not None else None
        )
        
        # Residual connection and layer norm
        enhanced_features = self.ln2(visual_features + self.dropout(self_attended))
        
        # Feed-forward network
        ffn_output = self.ffn(enhanced_features)
        enhanced_features = self.ln3(enhanced_features + self.dropout(ffn_output))
        
        attention_info = {
            'prompts': prompts,
            'prompt_weights': prompt_weights,
            'cross_attention': cross_attn_weights,
            'self_attention': self_attn_weights
        }
        
        return enhanced_features, attention_info


if __name__ == "__main__":
    # Test the prompt-guided attention mechanism
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Parameters
    batch_size, seq_len, embed_dim = 2, 16, 512
    vocab_size = 1000
    
    # Create dummy input
    visual_features = torch.randn(batch_size, seq_len, embed_dim).to(device)
    target_classes = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    feature_mask = torch.ones(batch_size, seq_len, dtype=torch.bool).to(device)
    
    # Initialize model
    model = PromptGuidedAttention(vocab_size, embed_dim).to(device)
    
    # Forward pass
    with torch.no_grad():
        enhanced_features, attention_info = model(visual_features, target_classes, feature_mask)
        
    print(f"Input features shape: {visual_features.shape}")
    print(f"Enhanced features shape: {enhanced_features.shape}")
    print(f"Prompts shape: {attention_info['prompts'].shape}")
    print(f"Cross attention shape: {attention_info['cross_attention'].shape}")
    print("Prompt-guided attention test passed!")