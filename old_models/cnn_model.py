"""
1D CNN MODEL FOR TDE CLASSIFICATION

Alternative to LSTM - often better for sequential data with spatial patterns
Simpler, faster, less prone to gradient issues
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TDE_CNN(nn.Module):
    """
    1D Convolutional Neural Network for TDE Classification
    
    CONCEPT: 1D CNN
    - Applies convolutions across time dimension
    - Captures local patterns (peaks, trends)
    - Faster than LSTM, less memory
    - Good for time series classification
    """
    def __init__(self, input_size, num_filters=64, dropout=0.3):
        super(TDE_CNN, self).__init__()
        
        # Input batch normalization
        self.input_bn = nn.BatchNorm1d(input_size)
        
        # CNN layers with different kernel sizes (multi-scale)
        # Kernel size 3: local patterns
        self.conv1 = nn.Conv1d(input_size, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(num_filters)
        
        # Kernel size 5: medium patterns
        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(num_filters)
        
        # Kernel size 7: longer patterns
        self.conv3 = nn.Conv1d(num_filters, num_filters*2, kernel_size=7, padding=3)
        self.bn3 = nn.BatchNorm1d(num_filters*2)
        
        # Global pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # FC layers
        self.fc1 = nn.Linear(num_filters*2 * 2, 64)  # *2 for avg+max pooling
        self.bn_fc = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 2)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x shape: (batch, timesteps, features)
        
        # Transpose for Conv1d: (batch, features, timesteps)
        x = x.transpose(1, 2)
        
        # Input normalization
        x = self.input_bn(x)
        
        # Conv layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)  # Reduce temporal dimension
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)
        
        x = self.conv3(x)
        x = self.bn3(x)        
        x = F.relu(x)
        
        # Global pooling (both avg and max)
        avg_pool = self.global_avg_pool(x).squeeze(-1)
        max_pool = self.global_max_pool(x).squeeze(-1)
        x = torch.cat([avg_pool, max_pool], dim=1)
        
        # FC layers
        x = self.fc1(x)
        x = self.bn_fc(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

print("1D CNN model loaded successfully!")
