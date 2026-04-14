import cv2
from ultralytics import YOLO

class PersonDetector:
    def __init__(self, model_type="yolov8n.pt"):
        """
        Initialize the YOLOv8 model.
        Available models: yolov8n.pt (nano - fast), yolov8s.pt (small - better accuracy), etc.
        """
        print(f"Loading model: {model_type}...")
        self.model = YOLO(model_type)
        
    def process_video(self, video_path=0, output_path=None, display=True):
        """
        Process a video (or webcam link) to detect persons.
        Returns a list of dictionaries containing frame-by-frame detections.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video source {video_path}")
            return []

        # Setup video writer if output path is given
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

        print(f"Started video processing on: {video_path}. Press 'q' to stop early.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Predict only class 0 (person)
            results = self.model(frame, classes=0, verbose=False)
            
            frame_detections = []
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Extract coordinates as integers
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    # Extract confidence
                    conf = float(box.conf[0].cpu().numpy())
                    
                    frame_detections.append({
                        "bbox": (x1, y1, x2, y2),
                        "confidence": conf
                    })
                    
                    # Draw bounding box on frame
                    if display or output_path:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"Person {conf:.2f}"
                        cv2.putText(frame, label, (x1, max(y1 - 10, 0)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Store structural data for this frame
            structured_output.append({
                "frame_id": frame_id,
                "detections": frame_detections
            })
            
            if output_path:
                out.write(frame)

            if display:
                cv2.imshow("Person Detection (YOLOv8)", frame)
                # Break condition
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Process interrupted by user.")
                    break
            
            frame_id += 1

        # Clean-up
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        print(f"Finished processing {frame_id} frames.")
        return structured_output

if __name__ == "__main__":
    detector = PersonDetector("yolov8n.pt")
    
    # For testing, we use the webcam (0). 
    # Change it to "your_video.mp4" to test on a stored file.
    video_source = 0  
    
    # Process the video feed.
    # display=True shows the real-time feed with bounded boxes.
    all_detections = detector.process_video(video_path=video_source, display=True)
    
    # Show first frame results as a sample of expected structure
    if all_detections and len(all_detections) > 0:
        print("\n--- Example Output (Frame 0) ---")
        print(all_detections[0])
