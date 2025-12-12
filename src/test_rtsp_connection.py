"""
Test RTSP Connection Script
Task 1: Verify camera connectivity and stream access
"""

import cv2
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_rtsp_connection():
    """Test connection to RTSP camera stream."""
    
    rtsp_url = os.getenv("RTSP_URL")
    
    if not rtsp_url:
        print("[ERROR] RTSP_URL not found in .env file")
        return False
    
    print(f"[INFO] Attempting to connect to: {rtsp_url[:30]}...")
    print("-" * 50)
    
    # Create video capture object
    cap = cv2.VideoCapture(rtsp_url)
    
    # Set timeout (optional - helps with slow connections)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)  # 10 seconds
    
    if not cap.isOpened():
        print("[ERROR] Failed to connect to RTSP stream")
        print("   Check your credentials, IP address, and port")
        return False
    
    print("[OK] Successfully connected to RTSP stream!")
    
    # Get stream properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"[INFO] Stream Properties:")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print("-" * 50)
    
    # Try to read a frame
    ret, frame = cap.read()
    
    if ret:
        print("[OK] Successfully read frame from stream!")
        
        # Save test frame
        os.makedirs("output", exist_ok=True)
        test_frame_path = "output/test_frame.jpg"
        cv2.imwrite(test_frame_path, frame)
        print(f"[INFO] Test frame saved to: {test_frame_path}")
        
        # Display frame (optional - press 'q' to quit)
        display = os.getenv("DISPLAY_WINDOW", "true").lower() == "true"
        
        if display:
            print("\n[INFO] Displaying live stream (Press 'q' to quit)...")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[ERROR] Lost connection to stream")
                    break
                
                # Add text overlay
                cv2.putText(frame, "RTSP Test - Press 'q' to quit", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 255, 0), 2)
                
                cv2.imshow("RTSP Stream Test", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    else:
        print("[ERROR] Failed to read frame from stream")
        cap.release()
        return False
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n[OK] RTSP connection test completed successfully!")
    return True


if __name__ == "__main__":
    print("=" * 50)
    print("   RTSP Connection Test")
    print("=" * 50)
    test_rtsp_connection()
