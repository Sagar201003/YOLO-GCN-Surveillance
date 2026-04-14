import numpy as np
import cv2
from collections import deque

class SkeletonBuffer:
    def __init__(self, max_frames=30, num_joints=17, num_channels=3):
        """
        Maintains a temporal sliding window of skeleton keypoints per person ID.
        Output shape per sequence: (T, V, C) where T=max_frames, V=17, C=x,y,conf.
        """
        self.max_frames = max_frames
        self.num_joints = num_joints
        self.num_channels = num_channels
        self.buffers = {}
        self.last_updated_frame = {}
        
    def update(self, track_id, keypoints, current_frame_idx):
        """
        Inserts new keypoints into the sliding window for a given track id.
        """
        keypoints = np.array(keypoints, dtype=np.float32)
        
        if track_id not in self.buffers:
            self.buffers[track_id] = deque(maxlen=self.max_frames)
            self.buffers[track_id].append(keypoints)
            self.last_updated_frame[track_id] = current_frame_idx
        else:
            # Handle missing frames (up to 5) by copying the last known pose
            frame_gap = current_frame_idx - self.last_updated_frame[track_id] - 1
            if frame_gap > 0:
                last_kpts = self.buffers[track_id][-1]
                for _ in range(min(frame_gap, 5)):
                    self.buffers[track_id].append(last_kpts)
            
            self.buffers[track_id].append(keypoints)
            self.last_updated_frame[track_id] = current_frame_idx

    def get_sequence(self, track_id):
        """
        Extracts the (T, V, C) tensor. Interpolates/pads with zeros or copies
        if we haven't reached T frames yet.
        """
        if track_id not in self.buffers:
            return None
            
        seq_list = list(self.buffers[track_id])
        current_len = len(seq_list)
        seq_arr = np.array(seq_list)
        
        # Exact expected length
        if current_len == self.max_frames:
            return seq_arr
            
        # If less than T frames, pad by repeating the very first frame captured
        repeats = self.max_frames - current_len
        first_frame = seq_arr[0]
        pad_arr = np.tile(first_frame, (repeats, 1, 1))
        out_tensor = np.concatenate((pad_arr, seq_arr), axis=0)
        
        return out_tensor

    def remove_stale_tracks(self, current_frame_idx, max_gap=30):
        """
        Clears memory of track IDs that have vanished for too long.
        """
        stale_tracks = [tid for tid, last_f in self.last_updated_frame.items() 
                        if current_frame_idx - last_f > max_gap]
        for tid in stale_tracks:
            del self.buffers[tid]
            del self.last_updated_frame[tid]

class GCNPreprocessor:
    def __init__(self, sequence_length=30, num_joints=17, max_persons=1):
        """
        Prepares the skeleton sequence for a Graph Convolutional Network.
        Target shape for GCNs (CTR-GCN / 2s-AGCN) is usually: (N, C, T, V, M)
        N = Batch Size (1 for real-time inference)
        C = Channels (e.g., 3 for x, y, confidence)
        T = Temporal Length (frames, e.g., 30)
        V = Vertices (17 joints)
        M = Max persons per sequence (e.g., 1 for independent classification)
        """
        self.T = sequence_length
        self.V = num_joints
        self.M = max_persons
        
        # COCO 17-Joint formatting parent mapping (used for bone features)
        # Root is generally pelvis (midpoint of hips). 
        # Parent mapping: joint -> parent_joint
        self.bone_pairs = [
            (0, 0),   # 0: Nose -> Nose (root)
            (1, 0),   # 1: L-eye -> Nose
            (2, 0),   # 2: R-eye -> Nose
            (3, 1),   # 3: L-ear -> L-eye
            (4, 2),   # 4: R-ear -> R-eye
            (5, 0),   # 5: L-shoulder -> Nose
            (6, 0),   # 6: R-shoulder -> Nose
            (7, 5),   # 7: L-elbow -> L-shoulder
            (8, 6),   # 8: R-elbow -> R-shoulder
            (9, 7),   # 9: L-wrist -> L-elbow
            (10, 8),  # 10: R-wrist -> R-elbow
            (11, 5),  # 11: L-hip -> L-shoulder
            (12, 6),  # 12: R-hip -> R-shoulder
            (13, 11), # 13: L-knee -> L-hip
            (14, 12), # 14: R-knee -> R-hip
            (15, 13), # 15: L-ankle -> L-knee
            (16, 14)  # 16: R-ankle -> R-knee
        ]

    def normalize_keypoints(self, sequence):
        """
        Normalizes the keypoints relatively by subtracting the center of the body.
        Center is determined via the midpoint of the left and right hip (11, 12).
        sequence shape: (T, V, 3)
        """
        seq_norm = sequence.copy()
        
        # Midpoint of hips (index 11 and 12)
        hip_left = seq_norm[:, 11, :2]  # Shape (T, 2)
        hip_right = seq_norm[:, 12, :2]
        center = (hip_left + hip_right) / 2.0  # Shape (T, 2)
        
        # Subtract center from all joints' x,y coordinates
        # Expand center to (T, 1, 2) to broadcast across V (17) joints
        seq_norm[:, :, :2] = seq_norm[:, :, :2] - center[:, np.newaxis, :]
        
        return seq_norm

    def extract_motion_features(self, sequence):
        """
        Extracts temporal velocity per joint.
        Velocity(t) = Point(t+1) - Point(t)
        Returns: Velocity (T, V, 3)
        """
        velocity = np.zeros_like(sequence)
        # Shifted difference across the T axis for x and y
        velocity[:-1, :, :2] = sequence[1:, :, :2] - sequence[:-1, :, :2]
        # Keep confidence the same
        velocity[:, :, 2] = sequence[:, :, 2]
        
        return velocity

    def extract_bone_features(self, sequence):
        """
        Extracts spatial bone vectors.
        Bone(j) = Point(j) - Point(parent_j)
        Returns: Bone (T, V, 3)
        """
        bone = np.zeros_like(sequence)
        for target, parent in self.bone_pairs:
            bone[:, target, :2] = sequence[:, target, :2] - sequence[:, parent, :2]
            bone[:, target, 2] = sequence[:, target, 2] # Preserve joint confidence
            
        return bone

    def process(self, sequence):
        """
        Main runner: takes raw T=30 tensor and creates optimized dimensions.
        Expects: `sequence` array of shape (T, 17, 3).
        
        Returns: Dictionary of correctly formatted structured tensors for GCN streams.
        Output Shape per stream: (N=1, C=3, T=30, V=17, M=1)
        """
        if sequence.shape != (self.T, self.V, 3):
            raise ValueError(f"Expected shape ({self.T}, {self.V}, 3), got {sequence.shape}")
            
        # 1. Coordinate Normalization
        norm_joints = self.normalize_keypoints(sequence)
        
        # 2. Extract Streams
        velocity_stream = self.extract_motion_features(norm_joints)
        bone_stream = self.extract_bone_features(norm_joints)
        
        # 3. Reshape to Standard GCN Layout: (N, C, T, V, M)
        # Current shape: (T, V, 3). We need to transpose to (3, T, V) first.
        norm_joints_trans = np.transpose(norm_joints, (2, 0, 1))
        vel_trans = np.transpose(velocity_stream, (2, 0, 1))
        bone_trans = np.transpose(bone_stream, (2, 0, 1))
        
        # Expand dims to add Batch (N=1) and Max_Persons (M=1)
        # Array manipulation: (C, T, V) -> (1, C, T, V, 1)
        joint_tensor = np.expand_dims(np.expand_dims(norm_joints_trans, axis=0), axis=-1)
        velocity_tensor = np.expand_dims(np.expand_dims(vel_trans, axis=0), axis=-1)
        bone_tensor = np.expand_dims(np.expand_dims(bone_trans, axis=0), axis=-1)
        
        return {
            "joint_data": joint_tensor,
            "velocity_data": velocity_tensor,
            "bone_data": bone_tensor
        }

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
