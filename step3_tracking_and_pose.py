import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

def compute_iou(box1, box2):
    """
    Computes Intersection over Union (IoU) between two boxes.
    Box format: [x1, y1, x2, y2]
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0

    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area

class PoseTracker:
    def __init__(self, model_type="yolov8n-pose.pt"):
        print(f"Loading pose model: {model_type}...")
        self.model = YOLO(model_type)
        
        # Initialize DeepSORT Tracker
        # Using typical params. Adjust max_age if people disappear/reappear briefly.
        self.tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0, max_cosine_distance=0.2)
        
    def process_video(self, video_path=0, output_path=None, display=True):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video source {video_path}")
            return {}

        if output_path:
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            if fps == 0: 
                fps = 30
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        # This will hold the skeleton sequence per track_id over time
        # Formatted: { track_id: [ {frame_id: id, keypoints: [...]}, ... ] }
        skeleton_sequences = {}
        frame_id = 0

        print(f"Started Multi-Person Pose Tracking on: {video_path}. Press 'q' to stop.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 1. Run YOLO Pose inference
            results = self.model(frame, classes=0, verbose=False)
            
            yolo_detections = []
            yolo_keypoints = []
            
            for result in results:
                boxes = result.boxes
                keypoints = result.keypoints
                
                if boxes is None or keypoints is None or len(boxes) == 0:
                    continue
                    
                for i in range(len(boxes)):
                    # Extract bbox and convert to DeepSORT format [left, top, w, h]
                    x1, y1, x2, y2 = map(int, boxes.xyxy[i].cpu().numpy())
                    box_conf = float(boxes.conf[i].cpu().numpy())
                    
                    w = x2 - x1
                    h = y2 - y1
                    # deep_sort_realtime detection format
                    yolo_detections.append(([x1, y1, w, h], box_conf, 0))
                    
                    # Extract 17 keypoints
                    kpts_list = []
                    if keypoints.data is not None and len(keypoints.data) > i:
                        person_kpts = keypoints.data[i].cpu().numpy()
                        for kpt in person_kpts:
                            kpts_list.append((float(kpt[0]), float(kpt[1]), float(kpt[2])))
                            
                    # Cache YOLO data for matching since DeepSORT returns updated tracks
                    yolo_keypoints.append({
                        "bbox": [x1, y1, x2, y2],
                        "keypoints": kpts_list
                    })

            # 2. Update tracks in DeepSORT
            # It extracts appearance embeddings from the frame
            tracks = self.tracker.update_tracks(yolo_detections, frame=frame)
            
            # 3. Process Tracks and Match with Keypoints
            for track in tracks:
                if not track.is_confirmed():
                    continue
                    
                track_id = track.track_id
                # Get the smoothed tracking bounding box
                ltrb = track.to_ltrb()
                tx1, ty1, tx2, ty2 = map(int, ltrb)
                
                best_iou = 0
                best_kpts = None
                
                # Simple spatial matching (IoU) to find which YOLO pose belongs to this track
                for yk in yolo_keypoints:
                    iou = compute_iou([tx1, ty1, tx2, ty2], yk["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_kpts = yk["keypoints"]
                
                # If a match is found and is reasonably close
                if best_kpts is not None and best_iou > 0.3:
                    # Store the keypoints with the associated track sequence over time
                    if track_id not in skeleton_sequences:
                        skeleton_sequences[track_id] = []
                        
                    skeleton_sequences[track_id].append({
                        "frame_id": frame_id,
                        "keypoints": best_kpts
                    })
                    
                    # Visually draw the tracking state
                    if display or output_path:
                        # Draw tracking box & ID
                        cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (255, 0, 0), 2)
                        cv2.putText(frame, f"ID: {track_id}", (tx1, max(ty1 - 10, 0)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        
                        # Draw tracked skeleton
                        for point in best_kpts:
                            px, py, pconf = point
                            if pconf > 0.5: # plot confident joints
                                cv2.circle(frame, (int(px), int(py)), 4, (0, 255, 0), -1)

            if output_path:
                out.write(frame)

            if display:
                cv2.imshow("Multi-Person Pose Tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Process interrupted by user.")
                    break
            
            frame_id += 1

        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        print(f"\nFinished tracking {frame_id} frames. Total Unique IDs: {len(skeleton_sequences)}")
        return skeleton_sequences

if __name__ == "__main__":
    pose_tracker = PoseTracker("yolov8n-pose.pt")
    video_source = 0  # 0 for webcam
    
    # Process the video to construct Temporal Skeleton Sequences
    seqs = pose_tracker.process_video(video_path=video_source, display=True)
    
    if seqs and len(seqs) > 0:
        sample_id = list(seqs.keys())[0]
        print(f"\n--- Example Output for Track ID '{sample_id}' ---")
        print(f"Total tracked frames for this person: {len(seqs[sample_id])}")
        if len(seqs[sample_id]) > 0:
             print(f"Keypoints in their first tracked frame (first 3 joints):")
             print(seqs[sample_id][0]['keypoints'][:3])
