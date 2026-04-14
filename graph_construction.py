import numpy as np

class Graph:
    def __init__(self, strategy='spatial', max_hop=1, center_node=0):
        """
        Constructs the multi-scale, structured Adjacency array `A` mapped directly to COCO 17-joints.
        
        Args:
            strategy (str): 'spatial' is the standard ST-GCN partition strategy (root, centripetal, centrifugal).
            max_hop (int): how far out the graph reaches. 1 = direct connections.
            center_node (int): center node to root the spatial gravity partition (default is 0 for Nose).
        """
        self.num_node = 17
        self.strategy = strategy
        self.max_hop = max_hop
        self.center_node = center_node
        
        # Build connections
        self.get_edge()
        
        # Calculate theoretical path lengths
        self.hop_dis = self.get_hop_distance(self.num_node, self.edge, self.max_hop)
        
        # Shape adjacency layouts according to temporal/spatial mechanics
        self.get_adjacency()

    def get_edge(self):
        """
        Defines the physical human anatomical skeletal structure (Nodes & Edges).
        Using COCO 17-joints topology corresponding to YOLOv8-pose.
        """
        self.self_link = [(i, i) for i in range(self.num_node)]
        
        self.neighbor_link = [
            (15, 13), (13, 11), # Left leg
            (16, 14), (14, 12), # Right leg
            (11, 12),           # Hip connection
            (5, 11), (6, 12),   # Torso 
            (5, 6),             # Shoulders
            (5, 7), (7, 9),     # Left arm
            (6, 8), (8, 10),    # Right arm
            (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), # Face
            (0, 5), (0, 6)      # Neck approximation (Nose to shoulders)
        ]
        
        # Full edge list = Static Identity Maps + Spatial Physical Connections
        self.edge = self.self_link + self.neighbor_link

    def get_hop_distance(self, num_node, edge, max_hop):
        """
        Computes shortest paths to define relationship ranges.
        A Max-Hop > 1 means connecting nodes even if they don't explicitly share a physical bone.
        """
        A = np.zeros((num_node, num_node))
        for i, j in edge:
            A[j, i] = 1
            A[i, j] = 1

        hop_dis = np.zeros((num_node, num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        
        for d in range(max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
            
        return hop_dis

    def get_adjacency(self):
        """
        Partitions the physical layout into separate spatial subset channels.
        Returns: self.A with shape (K, V, V) where K is the dimension of subsets 
        (usually 3 for spatial root/closer/further).
        """
        valid_hop = self.hop_dis <= self.max_hop
        
        if self.strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = valid_hop.astype(float) / np.sum(valid_hop, axis=1, keepdims=True)
            self.A = A
            
        elif self.strategy == 'spatial':
            A = []
            
            a_root = np.zeros((self.num_node, self.num_node))
            a_centripetal = np.zeros((self.num_node, self.num_node))
            a_centrifugal = np.zeros((self.num_node, self.num_node))
            
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if valid_hop[i, j]:
                        if self.hop_dis[j, self.center_node] == self.hop_dis[i, self.center_node]:
                            a_root[j, i] = 1
                        elif self.hop_dis[j, self.center_node] < self.hop_dis[i, self.center_node]:
                            a_centripetal[j, i] = 1
                        else:
                            a_centrifugal[j, i] = 1
            
            # Row-normalized adjacency subsets
            def normalize(matrix):
                row_sum = np.sum(matrix, axis=1, keepdims=True)
                row_sum[row_sum == 0] = 1
                return matrix / row_sum
                
            A.append(normalize(a_root))
            A.append(normalize(a_centripetal))
            A.append(normalize(a_centrifugal))
            
            # Complete Adjacency Tensor
            self.A = np.stack(A)
            
        else:
            raise ValueError(f"Strategy {self.strategy} is not supported.")

# --- Testing Snippet ---
if __name__ == '__main__':
    print("Compiling Human Adjacency Graph...")
    graph = Graph(strategy='spatial')
    
    A = graph.A
    print(f"\n✅ Graph Construction Successful.")
    print(f"Computed Adjacency Tensor Shape (K, V, V): {A.shape}")
    print(f"  - K=3 Partitions (Static Center, Centripetal Motion, Centrifugal Motion)")
    print(f"  - V=17 Keypoints")
    
    # The GCN uses this static 'A' array as a fixed weight layout against our moving features
    print("\nVisualizing Root Partition Sample (Node 0 - Nose connections):")
    root_connections = A[0, 0, :]
    print(f"Weights tied closely to the root: {np.round(root_connections, 2)}")
