"""
YOLOv8 Person Detection Script
Task 1: Run basic person detection on RTSP stream (real-time)
"""

import cv2
import os
import sys
from dotenv import load_dotenv

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import load_yolo_model

# Load environment variables
load_dotenv()


def run_person_detection(source=None, save_output=False):
    """
    Run YOLOv8 person detection on RTSP stream (real-time).
    
    Args:
        source: RTSP URL. If None, uses RTSP_URL from .env
        save_output: Whether to save the output video with detections.
    """
    
    # Determine source - prioritize RTSP for real-time detection
    if source is None:
        rtsp_url = os.getenv("RTSP_URL")
        
        if rtsp_url:
            source = rtsp_url
            print(f"[INFO] Using RTSP stream: {rtsp_url}")
        else:
            print("[ERROR] No RTSP_URL found in .env file!")
            print("   Please set RTSP_URL in your .env file")
            return
    
    # Load YOLOv8 model using the models package
    detector = load_yolo_model()
    
    # Open video source
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video source: {source}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    print(f"[INFO] Video Properties: {width}x{height} @ {fps:.1f} FPS")
    print("-" * 50)
    
    # Setup output video writer
    output_path = None
    writer = None
    
    if save_output:
        os.makedirs("output", exist_ok=True)
        output_path = "output/person_detection_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    total_detections = 0
    display = os.getenv("DISPLAY_WINDOW", "true").lower() == "true"
    
    print("[INFO] Starting person detection... (Press 'q' to quit)")
    print("-" * 50)
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("\n[INFO] End of video stream")
                break
            
            frame_count += 1
            
            # Run YOLOv8 inference using detector
            detections = detector.detect_persons(frame)
            
            # Process results
            person_count = len(detections)
            total_detections += person_count
            
            for (x1, y1, x2, y2, conf) in detections:
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"Person {conf:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - 25), (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Add info overlay
            info_text = f"Frame: {frame_count} | Persons: {person_count}"
            cv2.putText(frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Write to output video
            if writer:
                writer.write(frame)
            
            # Display frame
            if display:
                cv2.imshow("YOLOv8 Person Detection", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n[INFO] Stopped by user")
                    break
            
            # Print progress every 100 frames
            if frame_count % 100 == 0:
                print(f"   Processed {frame_count} frames...")
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
    
    # Print summary
    print("\n" + "=" * 50)
    print("Detection Summary")
    print("=" * 50)
    print(f"   Total frames processed: {frame_count}")
    print(f"   Total person detections: {total_detections}")
    print(f"   Average persons per frame: {total_detections/max(frame_count,1):.2f}")
    
    if output_path:
        print(f"   Output saved to: {output_path}")
    
    print("=" * 50)


def run_detection_on_rtsp():
    """Run person detection directly on RTSP stream (real-time)."""
    rtsp_url = os.getenv("RTSP_URL")
    if rtsp_url:
        run_person_detection(source=rtsp_url, save_output=False)
    else:
        print("[ERROR] RTSP_URL not found in .env file")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLOv8 Person Detection - Real-time RTSP")
    parser.add_argument("--source", type=str, help="Custom RTSP URL (overrides .env)")
    parser.add_argument("--save", action="store_true", help="Save output video")
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("   YOLOv8 Real-Time Person Detection")
    print("=" * 50)
    
    # Run detection on RTSP stream
    run_person_detection(source=args.source, save_output=args.save)
