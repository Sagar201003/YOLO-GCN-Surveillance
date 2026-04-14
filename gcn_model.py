import torch
import torch.nn as nn
import torch.nn.functional as F
from graph_construction import Graph

class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, K=3):
        """
        Calculates Spatial Graph Convolution mapping features across the Adjacency relations.
        K is the number of spatial partitions (e.g., 3: root, closer, further).
        """
        super().__init__()
        self.K = K
        # Expands channels to map against our unique K partitions separately
        self.conv = nn.Conv2d(in_channels, out_channels * K, kernel_size=1)
        
    def forward(self, x, A):
        # x shape: (N, C, T, V)
        # A shape: (K, V, V)
        N, C, T, V = x.size()
        
        # 1x1 Convolution 
        x = self.conv(x) 
        x = x.view(N, self.K, -1, T, V) # shape: (N, K, out_channels, T, V)
        
        # Matrix Multiplication between Joint Tensors and Adjacency Structure
        # Utilizes Einstein Summation for optimal batch matrix reduction
        x = torch.einsum('n k c t v, k v w -> n c t w', x, A)
        return x.contiguous()


class TCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, temporal_window_size=9, stride=1):
        """
        Analyzes motion patterns OVER TIME (Temporal Convolutional Network).
        """
        super().__init__()
        # Pad temporal dimension appropriately
        padding = ((temporal_window_size - 1) // 2, 0)
        
        # Convolution isolated safely over the TIME axis
        self.temporal_conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(temporal_window_size, 1),
            stride=(stride, 1),
            padding=padding
        )
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.temporal_conv(x)
        return self.bn(x)


class STGCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, K=3, stride=1):
        super().__init__()
        # Binds the Space (Joint relations) and Time (Temporal motion)
        self.sgcn = SpatialGraphConv(in_channels, out_channels, K)
        self.tcn = TCN_Block(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual bypass if channels alter or we stride down Time limits
        if (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x, A):
        res = self.residual(x)
        x = self.sgcn(x, A)
        x = self.tcn(x)
        return self.relu(x + res)


class ActionRecognitionGCN(nn.Module):
    def __init__(self, num_classes=2, in_channels=3, graph_strategy='spatial'):
        """
        Main Model integrating Step 6 Graph with stacked ST-GCN neural layers.
        num_classes: e.g. 2 -> [0: Normal Activity, 1: Suspicious Activity]
        """
        super().__init__()
        
        # Inject the Hardcoded Step 6 Structural Graph
        self.graph = Graph(strategy=graph_strategy)
        # Register matrix A into PyTorch so GPU handles the weights effectively
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        
        K = A.size(0) 
        
        # Stabilize incoming joint data
        self.data_bn = nn.BatchNorm1d(in_channels * self.graph.num_node)
        
        # Deep Convolution layer stacks
        self.layers = nn.ModuleList([
            STGCN_Block(in_channels, 64, K, stride=1),
            STGCN_Block(64, 64, K, stride=1),
            STGCN_Block(64, 128, K, stride=2), # Downsamples T
            STGCN_Block(128, 128, K, stride=1),
            STGCN_Block(128, 256, K, stride=2), # Downsamples T
            STGCN_Block(256, 256, K, stride=1)
        ])
        
        # Dense classification layer
        self.fc = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # x is assumed: (N, C, T, V, M)
        N, C, T, V, M = x.size()
        
        # 1. Standardize formatting and apply BN wrapper
        x = x.permute(0, 4, 3, 1, 2).contiguous() # (N, M, V, C, T)
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        
        # Return to (Batch_Persons, Channels, Time, Vertices) 
        x = x.view(N * M, V, C, T).permute(0, 2, 3, 1).contiguous()
        
        # 2. Ripple through Convolutional Layers
        for layer in self.layers:
            x = layer(x, self.A)
            
        # 3. Global Average Pooling (Aggregates T and V to find abstract meaning)
        x = F.avg_pool2d(x, x.size()[2:]) # Dim: (N*M, 256, 1, 1)
        x = x.view(N, M, -1).mean(dim=1)  # Mean across interacting Persons (M)
        
        # 4. Multi-Layer Perceptron Classification
        return self.fc(x)

# --- Scaffold Testing Logic ---
if __name__ == '__main__':
    # Define Binary classifier (Normal vs Suspicious)
    model = ActionRecognitionGCN(num_classes=2, in_channels=3)
    
    print(f"Model instantiated! Automatically pulled Adjacency Matrix A of shape {model.A.shape} \n")
    
    # ---------------------------------------------
    # Simulated T=30 Sequence passed from step 5!
    # N=4 sequences (Batch), C=3 channels, T=30, Vertices=17, M=1 person 
    # ---------------------------------------------
    dummy_input = torch.randn(4, 3, 30, 17, 1) 
    
    # Forward Pass inference
    logits = model(dummy_input)
    predictions = F.softmax(logits, dim=1)
    
    print(f"Input Data Shape:  {dummy_input.shape}")
    print(f"Prediction Array Shape: {logits.shape} -> Matrix array outputs:\n{predictions.detach().numpy()}\n")
    
    # ---------------------------------------------
    # TRAINING ALGORITHM HOOKS
    # ---------------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Fake Ground Truth Labels [1 = Suspicious, 0 = Normal]
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    
    optimizer.zero_grad()                 # Clear old gradients
    loss = criterion(logits, labels)      # Calculate error map
    loss.backward()                       # Compute Backward adjustments
    optimizer.step()                      # Trigger weight modification
    
    print("Full Loop Success!")
    print(f"Backpropagation computed. Exact loss triggered: {loss.item():.4f}")
