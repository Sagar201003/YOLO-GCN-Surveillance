import numpy as np
import cv2
from collections import deque
from step3_tracking_and_pose import PoseTracker

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

# --- Testing / Integration Snippet ---
def process_with_temporal_buffer(video_source=0, sequence_length=30):
    # Initialize modular components
    tracker = PoseTracker("yolov8n-pose.pt")
    skeleton_buffer = SkeletonBuffer(max_frames=sequence_length)
    
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Failed to open video")
        return
        
    frame_id = 0
    print(f"Starting Buffer pipeline (T={sequence_length}). Press 'q' to exit.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # 1. Run YOLO inference
        results = tracker.model(frame, classes=0, verbose=False)
        
        yolo_detections = []
        yolo_keypoints = []
        
        for result in results:
            boxes = result.boxes
            keypoints = result.keypoints
            if boxes is None or len(boxes) == 0: continue
                
            for i in range(len(boxes)):
                x1, y1, x2, y2 = map(int, boxes.xyxy[i].cpu().numpy())
                conf = float(boxes.conf[i].cpu().numpy())
                yolo_detections.append(([x1, y1, x2-x1, y2-y1], conf, 0))
                
                kpts_list = []
                if keypoints.data is not None and len(keypoints.data) > i:
                    person_kpts = keypoints.data[i].cpu().numpy()
                    for kpt in person_kpts:
                        kpts_list.append((float(kpt[0]), float(kpt[1]), float(kpt[2])))
                yolo_keypoints.append({"bbox": [x1, y1, x2, y2], "keypoints": kpts_list})

        # 2. Update DeepSORT mapping
        tracks = tracker.tracker.update_tracks(yolo_detections, frame=frame)
        
        # 3. Match keys and append to Slide Window Sequence
        for track in tracks:
            if not track.is_confirmed(): continue
            track_id = track.track_id
            
            tx1, ty1, tx2, ty2 = map(int, track.to_ltrb())
            best_iou, best_kpts = 0, None
            
            for yk in yolo_keypoints:
                box1, box2 = [tx1, ty1, tx2, ty2], yk["bbox"]
                xl = max(box1[0], box2[0])
                yt = max(box1[1], box2[1])
                xr = min(box1[2], box2[2])
                yb = min(box1[3], box2[3])
                
                if xr < xl or yb < yt: iou = 0.0
                else:
                    inter = (xr - xl) * (yb - yt)
                    a1 = (box1[2]-box1[0])*(box1[3]-box1[1])
                    a2 = (box2[2]-box2[0])*(box2[3]-box2[1])
                    iou = inter / float(a1 + a2 - inter)
                    
                if iou > best_iou:
                    best_iou, best_kpts = iou, yk["keypoints"]
                    
            if best_kpts is not None and best_iou > 0.3:
                # ---------------------------------------------
                # 4. BUFFER UPDATE: Update the Sliding Sequence
                # ---------------------------------------------
                skeleton_buffer.update(track_id, best_kpts, frame_id)
                
                # Retrieve the full tensor (T, 17, 3) 
                sequence_tensor = skeleton_buffer.get_sequence(track_id)
                
                # For output display
                cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (0, 255, 255), 2)
                tensor_shape_text = f"{sequence_tensor.shape}" 
                cv2.putText(frame, f"ID: {track_id} | Shape: {tensor_shape_text}", 
                            (tx1, max(ty1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Optional: draw the joints
                for point in best_kpts:
                    if point[2] > 0.5:
                        cv2.circle(frame, (int(point[0]), int(point[1])), 3, (0, 255, 0), -1)

        skeleton_buffer.remove_stale_tracks(frame_id)
        
        cv2.imshow("Step 4 - Dynamic Neural Buffer Pipeline", frame)
        if cv2.waitKey(1) == ord('q'): break
        frame_id += 1
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # T=30 Sliding Window Buffer
    process_with_temporal_buffer(video_source=0, sequence_length=30)
