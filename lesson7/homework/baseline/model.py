# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseMLPBlock(nn.Module, ABC):
    def __init__(self, dim, activation='gelu', dropout=0.0):
        super().__init__()
        self.dim = dim
        self.activation = {'relu': nn.ReLU(), 'gelu': nn.GELU(), 'swish': nn.SiLU()}.get(activation, nn.GELU())
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    @abstractmethod
    def forward(self, x):
        pass


class BottleneckBlock(BaseMLPBlock):
    def __init__(self, dim, activation='gelu', dropout=0.0):
        super().__init__(dim, activation, dropout)
        self.bottleneck_dim = max(dim // 4, 1)
        self.fc1 = nn.Linear(self.dim, self.bottleneck_dim)
        self.fc2 = nn.Linear(self.bottleneck_dim, self.dim)
    
    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.activation(out)
        if self.dropout:
            out = self.dropout(out)
        out = self.fc2(out)
        return out + identity


class InvertedBottleneckBlock(BaseMLPBlock):
    def __init__(self, dim, expansion_factor=4, activation='gelu', dropout=0.0):
        super().__init__(dim, activation, dropout)
        self.expanded_dim = dim * expansion_factor
        self.fc1 = nn.Linear(self.dim, self.expanded_dim)
        self.fc2 = nn.Linear(self.expanded_dim, self.dim)
    
    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.activation(out)
        if self.dropout:
            out = self.dropout(out)
        out = self.fc2(out)
        return out + identity


class RegularBlock(BaseMLPBlock):
    def __init__(self, dim, hidden_dim=None, activation='gelu', dropout=0.0):
        super().__init__(dim, activation, dropout)
        self.hidden_dim = hidden_dim if hidden_dim else dim * 2
        self.fc1 = nn.Linear(self.dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.dim)
    
    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.activation(out)
        if self.dropout:
            out = self.dropout(out)
        out = self.fc2(out)
        return out + identity


class MultiBranchMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_blocks=4,
        dropout=0.1,
        combine_mode='concat'
    ):
        super().__init__()
        self.output_dim = output_dim
        self.combine_mode = combine_mode
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.bottleneck_branch = nn.ModuleList([
            BottleneckBlock(hidden_dim, activation='gelu', dropout=dropout)
            for _ in range(num_blocks)
        ])
        
        self.inverted_branch = nn.ModuleList([
            InvertedBottleneckBlock(hidden_dim, expansion_factor=4, activation='gelu', dropout=dropout)
            for _ in range(num_blocks)
        ])
        
        self.regular_branch = nn.ModuleList([
            RegularBlock(hidden_dim, hidden_dim=hidden_dim * 2, activation='gelu', dropout=dropout)
            for _ in range(num_blocks)
        ])
        
        if combine_mode == 'concat':
            output_proj_input_dim = hidden_dim * 3
        else:
            output_proj_input_dim = hidden_dim
        
        self.output_proj = nn.Linear(output_proj_input_dim, output_dim)
    
    def forward(self, x):
        x = self.input_proj(x)
        
        bottleneck_out = x
        for block in self.bottleneck_branch:
            bottleneck_out = block(bottleneck_out)
        
        inverted_out = x
        for block in self.inverted_branch:
            inverted_out = block(inverted_out)
        
        regular_out = x
        for block in self.regular_branch:
            regular_out = block(regular_out)
        
        if self.combine_mode == 'concat':
            combined = torch.cat([bottleneck_out, inverted_out, regular_out], dim=1)
        else:
            combined = bottleneck_out + inverted_out + regular_out
        
        out = self.output_proj(combined)
        
        return out
