import numpy as np

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

# --- Testing snippet ---
if __name__ == "__main__":
    preprocessor = GCNPreprocessor(sequence_length=30)
    
    # Simulating a raw sequence from Step 4 buffer output: (T=30, V=17, C=3)
    dummy_sequence = np.random.rand(30, 17, 3) 
    
    # Process it
    gcn_inputs = preprocessor.process(dummy_sequence)
    
    print("--- Preprocessing Outcomes ---")
    print(f"Input Shape: {dummy_sequence.shape}")
    print(f"Norm Joint Tensor Shape (N, C, T, V, M): {gcn_inputs['joint_data'].shape}")
    print(f"Velocity Tensor Shape   (N, C, T, V, M): {gcn_inputs['velocity_data'].shape}")
    print(f"Bone Tensor Shape       (N, C, T, V, M): {gcn_inputs['bone_data'].shape}")
    
    # Show internal normalization example (Frame 0, center hip logic)
    print("\nDemonstrating Relative Center Rooting:")
    norm_j = gcn_inputs['joint_data'][0, :, 0, :, 0] # Extract the first frame
    left_hip_x = norm_j[0, 11] # C=0 (x), Joint=11 (L-hip)
    right_hip_x = norm_j[0, 12] # C=0 (x), Joint=12 (R-hip)
    center_x = (left_hip_x + right_hip_x) / 2
    
    print(f"New Hip Center Coordinates should theoretically approach 0.0.")
    print(f"Average X position of origin (hips): {center_x:.5f}")
