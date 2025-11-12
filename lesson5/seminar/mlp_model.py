"""
Deep MLP model for text classification
"""
import torch
import torch.nn as nn


class DeepMLP(nn.Module):
    """
    Deep MLP с поддержкой batch normalization и dropout
    
    Args:
        input_dim: размерность входа
        output_dim: размерность выхода (количество классов)
        hidden_dims: список размерностей скрытых слоев
        activation: функция активации ('relu', 'gelu', 'silu')
        dropout: dropout rate
        use_batch_norm: использовать ли batch normalization
    """
    
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims=[512, 256, 128],
        activation='relu',
        dropout=0.3,
        use_batch_norm=True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims if isinstance(hidden_dims, list) else [hidden_dims]
        
        # Выбираем активацию
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Строим сеть
        layers = []
        dims = [input_dim] + self.hidden_dims
        
        for i in range(len(dims) - 1):
            # Linear layer
            layers.append(nn.Linear(dims[i], dims[i+1]))
            
            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(dims[i+1]))
            
            # Activation
            layers.append(self.activation)
            
            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(dims[-1], output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    
    def get_num_parameters(self):
        """Возвращает количество параметров модели"""
        return sum(p.numel() for p in self.parameters())

