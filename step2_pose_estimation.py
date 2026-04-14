import cv2
from ultralytics import YOLO

class PoseEstimator:
    def __init__(self, model_type="yolov8n-pose.pt"):
        """
        Initialize the YOLOv8 Pose model.
        Available models: yolov8n-pose.pt (nano), yolov8s-pose.pt (small), etc.
        """
        print(f"Loading pose model: {model_type}...")
        self.model = YOLO(model_type)
        
    def process_video(self, video_path=0, output_path=None, display=True):
        """
        Process a video to extract skeletons for detected persons directly using the pose model.
        Returns extracted keypoints in a structured format.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video source {video_path}")
            return []

        if output_path:
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            if fps == 0:  # Sometimes webcam returns 0 fps
                fps = 30
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        structured_output = []
        frame_id = 0

        print(f"Started pose processing on: {video_path}. Press 'q' to stop.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Predict pose. Default classes=0 for person is implied in pose model.
            results = self.model(frame, classes=0, verbose=False)
            
            frame_poses = []
            
            for result in results:
                boxes = result.boxes         # Bounding boxes
                keypoints = result.keypoints # Pose Keypoints
                
                # Check if there are any detections directly in the result object
                if boxes is None or keypoints is None or len(boxes) == 0:
                    continue
                    
                # keypoints.data shape is (N, 17, 3) where N = number of persons
                # Each point is [x, y, confidence]
                
                for i in range(len(boxes)):
                    # Box info
                    x1, y1, x2, y2 = map(int, boxes.xyxy[i].cpu().numpy())
                    box_conf = float(boxes.conf[i].cpu().numpy())
                    
                    # Keypoint info
                    # Check if keypoints is not completely empty for the person
                    if keypoints.data is not None and len(keypoints.data) > i:
                        person_kpts = keypoints.data[i].cpu().numpy() # Shape: (17, 3)
                        
                        # Convert to a list of (x, y, conf)
                        kpts_list = []
                        for kpt in person_kpts:
                            x, y, conf = float(kpt[0]), float(kpt[1]), float(kpt[2])
                            kpts_list.append((x, y, conf))
                        
                        frame_poses.append({
                            "bbox": (x1, y1, x2, y2),
                            "bbox_confidence": box_conf,
                            "keypoints": kpts_list 
                        })

            # Visualization
            if display or output_path:
                # The ultralytics library provides a convenient way to plot the frame with keypoints
                annotated_frame = results[0].plot(boxes=True, conf=True, kpt_line=True)
                
                if output_path:
                    out.write(annotated_frame)
                
                if display:
                    cv2.imshow("Pose Estimation (YOLOv8 Pose)", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Process interrupted by user.")
                        break
            
            # Store structured data
            structured_output.append({
                "frame_id": frame_id,
                "persons": frame_poses
            })
            
            frame_id += 1

        # Clean-up
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        print(f"Finished processing {frame_id} frames.")
        return structured_output

if __name__ == "__main__":
    estimator = PoseEstimator("yolov8n-pose.pt")
    
    # Using webcam by default. Replace with video path to test on files.
    video_source = 0  
    
    # Process the video feed.
    all_poses = estimator.process_video(video_path=video_source, display=True)
    
    if all_poses and len(all_poses) > 0:
        print("\n--- Example Output (Frame 0) ---")
        persons_in_frame = all_poses[0]['persons']
        print(f"People detected in frame 0: {len(persons_in_frame)}")
        if len(persons_in_frame) > 0:
            print("Keypoints format for first person (first 3 joints: Nose, L-Eye, R-Eye):")
            print(persons_in_frame[0]['keypoints'][:3])
