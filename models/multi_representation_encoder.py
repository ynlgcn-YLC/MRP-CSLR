"""
Multi-Representation Encoder for Sign Language Recognition
This module implements the core multi-representation encoding framework
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import timm


class VisualFeatureExtractor(nn.Module):
    """
    Visual feature extraction using pre-trained CNN backbone
    """
    def __init__(self, backbone_name='resnet50', pretrained=True, feature_dim=2048):
        super(VisualFeatureExtractor, self).__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained)
        # Remove the classifier layer
        self.backbone.fc = nn.Identity()
        self.feature_dim = feature_dim
        
        # Feature projection layer
        self.proj = nn.Linear(self.backbone.num_features, feature_dim)
        
    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) - batch of video frames
        Returns:
            features: (B, T, feature_dim) - visual features
        """
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        
        # Extract features from backbone
        features = self.backbone(x)  # (B*T, backbone_features)
        features = self.proj(features)  # (B*T, feature_dim)
        
        # Reshape back to sequence format
        features = features.view(B, T, self.feature_dim)
        return features


class SpatialEncoder(nn.Module):
    """
    Spatial representation encoder using 2D convolutions
    """
    def __init__(self, input_dim, hidden_dim=512, num_layers=3):
        super(SpatialEncoder, self).__init__()
        
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.LayerNorm(hidden_dim)
            ])
            current_dim = hidden_dim
            
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Args:
            x: (B, T, input_dim) - input features
        Returns:
            spatial_repr: (B, T, hidden_dim) - spatial representations
        """
        return self.encoder(x)


class TemporalEncoder(nn.Module):
    """
    Temporal representation encoder using LSTM/Transformer
    """
    def __init__(self, input_dim, hidden_dim=512, num_layers=2, encoder_type='lstm'):
        super(TemporalEncoder, self).__init__()
        
        self.encoder_type = encoder_type
        
        if encoder_type == 'lstm':
            self.encoder = nn.LSTM(
                input_dim, hidden_dim, num_layers, 
                batch_first=True, bidirectional=True
            )
            self.proj = nn.Linear(hidden_dim * 2, hidden_dim)
        else:  # transformer
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=input_dim, nhead=8, dim_feedforward=hidden_dim * 2,
                dropout=0.1, batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
            self.proj = nn.Linear(input_dim, hidden_dim)
    
    def forward(self, x):
        """
        Args:
            x: (B, T, input_dim) - input features
        Returns:
            temporal_repr: (B, T, hidden_dim) - temporal representations
        """
        if self.encoder_type == 'lstm':
            output, _ = self.encoder(x)
            return self.proj(output)
        else:
            output = self.encoder(x)
            return self.proj(output)


class MultiRepresentationEncoder(nn.Module):
    """
    Core multi-representation encoder that combines spatial and temporal encodings
    """
    def __init__(self, 
                 visual_backbone='resnet50',
                 feature_dim=2048,
                 spatial_hidden_dim=512,
                 temporal_hidden_dim=512,
                 fusion_dim=512):
        super(MultiRepresentationEncoder, self).__init__()
        
        # Visual feature extraction
        self.visual_extractor = VisualFeatureExtractor(
            backbone_name=visual_backbone, 
            feature_dim=feature_dim
        )
        
        # Multi-representation encoders
        self.spatial_encoder = SpatialEncoder(
            input_dim=feature_dim, 
            hidden_dim=spatial_hidden_dim
        )
        
        self.temporal_encoder = TemporalEncoder(
            input_dim=feature_dim, 
            hidden_dim=temporal_hidden_dim,
            encoder_type='transformer'
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(spatial_hidden_dim + temporal_hidden_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(fusion_dim)
        )
        
    def forward(self, video_frames):
        """
        Args:
            video_frames: (B, T, C, H, W) - input video frames
        Returns:
            multi_repr: (B, T, fusion_dim) - multi-representation features
        """
        # Extract visual features
        visual_features = self.visual_extractor(video_frames)  # (B, T, feature_dim)
        
        # Encode spatial and temporal representations
        spatial_repr = self.spatial_encoder(visual_features)  # (B, T, spatial_hidden_dim)
        temporal_repr = self.temporal_encoder(visual_features)  # (B, T, temporal_hidden_dim)
        
        # Fuse representations
        combined_repr = torch.cat([spatial_repr, temporal_repr], dim=-1)  # (B, T, spatial+temporal)
        multi_repr = self.fusion(combined_repr)  # (B, T, fusion_dim)
        
        return multi_repr, {'spatial': spatial_repr, 'temporal': temporal_repr}


if __name__ == "__main__":
    # Test the multi-representation encoder
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dummy input
    batch_size, seq_len, channels, height, width = 2, 16, 3, 224, 224
    dummy_input = torch.randn(batch_size, seq_len, channels, height, width).to(device)
    
    # Initialize model
    model = MultiRepresentationEncoder().to(device)
    
    # Forward pass
    with torch.no_grad():
        output, components = model(dummy_input)
        
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Spatial component shape: {components['spatial'].shape}")
    print(f"Temporal component shape: {components['temporal'].shape}")
    print("Multi-representation encoder test passed!")