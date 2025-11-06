import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims=[64, 32],
        activation='relu',
        output_activation=None,
        dropout=0.0,
        use_batch_norm=False
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims if isinstance(hidden_dims, list) else [hidden_dims]
        self.activation_name = activation
        self.output_activation_name = output_activation
        self.dropout_rate = dropout
        self.use_batch_norm = use_batch_norm
        
        self.activation = self._get_activation(activation)
        
        layers = []
        dims = [input_dim] + self.hidden_dims
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(dims[i+1]))
            
            layers.append(self.activation)
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(dims[-1], output_dim))
        
        self.network = nn.Sequential(*layers)
        
        if output_activation == 'sigmoid':
            self.output_activation = nn.Sigmoid()
        elif output_activation == 'softmax':
            self.output_activation = nn.Softmax(dim=1)
        elif output_activation == 'tanh':
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = None
    
    def _get_activation(self, activation_name):
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'gelu': nn.GELU(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'swish': nn.SiLU(),
        }
        
        if activation_name.lower() not in activations:
            raise ValueError(
                f"Неизвестная активация '{activation_name}'. "
                f"Доступные: {list(activations.keys())}"
            )
        
        return activations[activation_name.lower()]
    
    def forward(self, x):
        output = self.network(x)
        
        if self.output_activation is not None:
            output = self.output_activation(output)
        
        if self.output_dim == 1 and output.dim() > 1:
            output = output.squeeze(-1)
        
        return output


class MLPRegressor(MLP):
    def __init__(
        self,
        input_dim,
        hidden_dims=[64, 32],
        activation='relu',
        dropout=0.0,
        use_batch_norm=False
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=1,
            hidden_dims=hidden_dims,
            activation=activation,
            output_activation=None,
            dropout=dropout,
            use_batch_norm=use_batch_norm
        )


class MLPClassifier(MLP):
    def __init__(
        self,
        input_dim,
        n_classes,
        hidden_dims=[64, 32],
        activation='relu',
        dropout=0.0,
        use_batch_norm=False
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=n_classes,
            hidden_dims=hidden_dims,
            activation=activation,
            output_activation=None,
            dropout=dropout,
            use_batch_norm=use_batch_norm
        )
