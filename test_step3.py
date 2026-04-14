import cv2
import numpy as np
from step3_tracking_and_pose import PoseTracker

def test():
    # Initialize
    print("Initializing PoseTracker...")
    tracker = PoseTracker("yolov8n-pose.pt")

    # Create a small dummy video file
    print("Creating dummy video...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('dummy_test.mp4', fourcc, 30, (640, 480))

    for i in range(30):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Just writing empty frames to ensure video loop reads correctly
        out.write(frame)
    out.release()

    print("Processing dummy video without display...")
    seqs = tracker.process_video(video_path='dummy_test.mp4', display=False)

    print("Test complete. The script executes without errors.")

if __name__ == "__main__":
    test()
