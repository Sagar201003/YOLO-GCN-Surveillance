import cv2
import torch
import torch.nn.functional as F
import numpy as np
import warnings

# Suppress minor dependency warnings internally for a cleaner terminal
warnings.filterwarnings('ignore')

from tracking_and_pose import PoseTracker
from skeleton_buffer import SkeletonBuffer
from preprocessing import GCNPreprocessor
from gcn_model import ActionRecognitionGCN

# Inference Mapping [0 = Normal, 1 = Suspicious]
CLASS_LABELS = {0: "Normal", 1: "Suspicious"}

def run_live_inference(video_source=0, sequence_length=30):
    # ---------------------------------------------------------
    # 1. Pipeline Initialization
    # ---------------------------------------------------------
    print("Loading YOLOv8-pose Engine & DeepSORT Tracker...")
    pose_tracker = PoseTracker("yolov8n-pose.pt")
    
    print(f"Initializing Multi-Person Neural Sliding Buffer (T={sequence_length})...")
    skeleton_buffer = SkeletonBuffer(max_frames=sequence_length)
    
    print("Booting Feature Engineering Preprocessor...")
    preprocessor = GCNPreprocessor(sequence_length=sequence_length)
    
    print("Loading Graph Convolutional Network into Memory...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ActionRecognitionGCN(num_classes=2, in_channels=3)
    
    # Normally, you would load your trained weights here:
    # model.load_state_dict(torch.load("pretrained_gcn_weights.pth"))
    
    model.to(device)
    model.eval() # Activate inference lockdown (shuts off dropout/grad updates)
    
    # ---------------------------------------------------------
    # 2. Video Capture Routing
    # ---------------------------------------------------------
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Fatal Error: Could not connect to visual feed.")
        return
        
    frame_id = 0
    print("\n-------------------------------------------")
    print("    🚨 SYSTEM ONLINE & MONITORING 🚨     ")
    print("-------------------------------------------")
    print("Press 'q' in the video window to shut down the feed.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # ---------------------------------------------------------
        # 3. Detect & Track (YOLO + DeepSORT)
        # ---------------------------------------------------------
        results = pose_tracker.model(frame, classes=0, verbose=False)
        yolo_detections, yolo_keypoints = [], []
        
        for result in results:
            boxes = result.boxes
            keypoints = result.keypoints
            if boxes is None or len(boxes) == 0: continue
            
            for i in range(len(boxes)):
                # Bounding Box Logic
                x1, y1, x2, y2 = map(int, boxes.xyxy[i].cpu().numpy())
                conf = float(boxes.conf[i].cpu().numpy())
                yolo_detections.append(([x1, y1, x2-x1, y2-y1], conf, 0))
                
                # Anatomy Array parsing
                kpts_list = []
                if keypoints.data is not None and len(keypoints.data) > i:
                    person_kpts = keypoints.data[i].cpu().numpy()
                    for kpt in person_kpts:
                        kpts_list.append((float(kpt[0]), float(kpt[1]), float(kpt[2])))
                yolo_keypoints.append({"bbox": [x1, y1, x2, y2], "keypoints": kpts_list})
                
        tracks = pose_tracker.tracker.update_tracks(yolo_detections, frame=frame)
        
        # ---------------------------------------------------------
        # 4. Integrate IDs & Process Neural Network Input
        # ---------------------------------------------------------
        for track in tracks:
            if not track.is_confirmed(): continue
            track_id = track.track_id
            
            tx1, ty1, tx2, ty2 = map(int, track.to_ltrb())
            best_iou, best_kpts = 0, None
            
            # Spatial Matching (Tie Bounding Box to Skeleton Structure)
            for yk in yolo_keypoints:
                box1, box2 = [tx1, ty1, tx2, ty2], yk["bbox"]
                xl, yt = max(box1[0], box2[0]), max(box1[1], box2[1])
                xr, yb = min(box1[2], box2[2]), min(box1[3], box2[3])
                
                if xr < xl or yb < yt: iou = 0.0
                else:
                    inter = (xr - xl) * (yb - yt)
                    a1 = (box1[2]-box1[0])*(box1[3]-box1[1])
                    a2 = (box2[2]-box2[0])*(box2[3]-box2[1])
                    iou = inter / float(a1 + a2 - inter)
                    
                if iou > best_iou: best_iou, best_kpts = iou, yk["keypoints"]
                    
            if best_kpts is not None and best_iou > 0.3:
                # ---------------------------------------------------------
                # 5. Extract Feature Stream & Predict Class
                # ---------------------------------------------------------
                # Slide temporal state memory mapping
                skeleton_buffer.update(track_id, best_kpts, frame_id)
                raw_tensor = skeleton_buffer.get_sequence(track_id) 
                
                # Compute Hip root transformations & broadcast target limits
                prepared_data = preprocessor.process(raw_tensor)
                
                # Transmit correctly normalized arrays to active memory map mapping
                gcn_input = torch.tensor(prepared_data['joint_data'], dtype=torch.float32).to(device)
                
                with torch.no_grad():
                    logits = model(gcn_input) # Runs entirely through Step 6 / Step 7 
                    probabilities = F.softmax(logits, dim=1)
                    confidence, predicted_class = torch.max(probabilities, 1)
                    
                class_idx = predicted_class.item()
                conf_score = confidence.item()
                activity_label = CLASS_LABELS.get(class_idx, "Unknown")
                
                # ---------------------------------------------------------
                # 6. Critical Suspicious Alert Logic Hooks
                # ---------------------------------------------------------
                # Dynamic rendering layout parameters based on AI Logic
                if class_idx == 1:
                    color = (0, 0, 255) # Warning Red
                    # Print alert out to terminal if suspicious probability breaches 75%
                    if conf_score > 0.65:
                        print(f"⚠️ ALERT Triggered: Subject ID {track_id} is flagged [Suspicious] - Model Confidence: {conf_score*100:.1f}%.")
                else:
                    color = (0, 255, 0) # Normal Safe Green 
                
                cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), color, 3)
                alert_text = f"ID:{track_id} {activity_label} ({conf_score:.2f})"
                cv2.putText(frame, alert_text, (tx1, max(ty1 - 10, 0)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        # Drop old memory allocations to preserve frame rates
        skeleton_buffer.remove_stale_tracks(frame_id)
        
        cv2.imshow("Suspicious Activity AI Core Engine", frame)
        if cv2.waitKey(1) == ord('q'): break
        frame_id += 1
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live_inference(video_source=0)
