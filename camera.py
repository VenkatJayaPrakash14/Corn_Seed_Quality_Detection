from ultralytics import YOLO
import os
from pathlib import Path
import cv2
import pandas as pd
from datetime import datetime
import torch
import time
import numpy as np
from collections import defaultdict, deque

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Load model and set up paths
model_path = r"E:\Corn_Seed_Analysis\corn11.pt"
save_dir = r"E:\Corn_Seed_Analysis\yolo11Results"

# Load model with CUDA
model = YOLO(model_path).to('cuda' if torch.cuda.is_available() else 'cpu')

# Create directories
results_dir = Path(save_dir)
results_dir.mkdir(parents=True, exist_ok=True)
(results_dir / "video_frames").mkdir(exist_ok=True)
(results_dir / "labels").mkdir(exist_ok=True)

# Initialize camera with error handling
def try_camera_index(index):
    try:
        print(f"Attempting to open camera index {index}...")
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret:
                print(f"SUCCESS: Camera index {index} opened and frame captured")
                return cap
            else:
                print(f"PARTIAL: Camera index {index} opened but couldn't read frame")
                cap.release()
                return None
        else:
            print(f"FAILED: Camera index {index} could not be opened")
            return None
    except Exception as e:
        print(f"ERROR: Exception when trying camera index {index}: {str(e)}")
        return None

# Check available cameras
print("\n--- CAMERA DETECTION ---")
print("Checking available cameras (this may take a moment)...")
available_cameras = []
for i in range(10):  # Check indices 0-9
    cap = try_camera_index(i)
    if cap is not None:
        available_cameras.append(i)
        cap.release()

print(f"\nDetected {len(available_cameras)} available camera(s): {available_cameras}")

# Select camera
camera_index = None
if 1 in available_cameras:
    camera_index = 1
    print("Selecting index 1 (likely USB camera)")
elif available_cameras:
    camera_index = available_cameras[0]
    print(f"No USB camera found at index 1, using first available camera at index {camera_index}")
else:
    print("ERROR: No cameras detected!")
    exit(1)

# Initialize camera
cap = None
try:
    print(f"\nOpening camera {camera_index} for video capture...")
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        raise Exception("Camera opened but isOpened() returned False")
        
    # Test frame reading
    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        raise Exception("Camera opened but couldn't read a frame")
        
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print(f"Camera {camera_index} successfully initialized!")
    print(f"Frame resolution: {test_frame.shape[1]}x{test_frame.shape[0]}")
    
except Exception as e:
    print(f"ERROR initializing camera {camera_index}: {str(e)}")
    if cap is not None:
        cap.release()
    print("\nPossible solutions:")
    print("1. Make sure your USB camera is properly connected")
    print("2. Check if another application is using the camera")
    print("3. Try running as administrator/with elevated privileges")
    print("4. Restart your computer and try again")
    print("5. Try a different USB port")
    exit(1)

# Video detection parameters
DETECTION_CONFIDENCE = 0.25
PROCESS_EVERY_N_FRAMES = 3  # Process every 3rd frame for better performance
SAVE_DETECTIONS = True
RECORD_VIDEO = False  # Set to True if you want to save the annotated video

# Object tracking parameters
IOU_THRESHOLD = 0.5  # Intersection over Union threshold for matching objects
MAX_TRACKING_FRAMES = 10  # Maximum frames to track an object without detection
MIN_DETECTION_FRAMES = 3  # Minimum frames an object must be detected to be considered valid
STABILITY_THRESHOLD = 0.7  # Confidence threshold for stable detections

# Initialize video recording if enabled
video_writer = None
if RECORD_VIDEO:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = results_dir / f"detection_video_{timestamp}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 20
    frame_size = (640, 480)
    video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, frame_size)
    print(f"Video recording enabled. Saving to: {video_path}")

# Colors for different classes (BGR format)
colors = {
    0: (255, 0, 0),    # Blue
    1: (0, 255, 0),    # Green
    2: (0, 0, 255),    # Red
    3: (255, 255, 0),  # Cyan
    4: (255, 0, 255),  # Magenta
    5: (0, 255, 255),  # Yellow
}

class ObjectTracker:
    """Simple object tracker to reduce duplicate detections"""
    
    def __init__(self, iou_threshold=0.5, max_frames=10, min_frames=3):
        self.iou_threshold = iou_threshold
        self.max_frames = max_frames
        self.min_frames = min_frames
        self.tracked_objects = {}
        self.next_id = 0
        self.confirmed_objects = set()
        
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def update(self, detections, frame_num):
        """Update tracker with new detections"""
        # Mark all existing objects as not seen this frame
        for obj_id in self.tracked_objects:
            self.tracked_objects[obj_id]['frames_since_seen'] += 1
        
        new_detections = []
        
        for detection in detections:
            x1, y1, x2, y2, conf, class_id, class_name = detection
            detection_box = (x1, y1, x2, y2)
            
            # Find best matching existing object
            best_match_id = None
            best_iou = 0
            
            for obj_id, obj_data in self.tracked_objects.items():
                if obj_data['class_name'] == class_name:
                    iou = self.calculate_iou(detection_box, obj_data['box'])
                    if iou > self.iou_threshold and iou > best_iou:
                        best_iou = iou
                        best_match_id = obj_id
            
            if best_match_id is not None:
                # Update existing object
                self.tracked_objects[best_match_id].update({
                    'box': detection_box,
                    'confidence': conf,
                    'frames_since_seen': 0,
                    'detection_count': self.tracked_objects[best_match_id]['detection_count'] + 1,
                    'last_seen_frame': frame_num
                })
            else:
                # Create new object
                self.tracked_objects[self.next_id] = {
                    'box': detection_box,
                    'confidence': conf,
                    'class_id': class_id,
                    'class_name': class_name,
                    'frames_since_seen': 0,
                    'detection_count': 1,
                    'first_seen_frame': frame_num,
                    'last_seen_frame': frame_num
                }
                self.next_id += 1
        
        # Remove objects that haven't been seen for too long
        to_remove = []
        for obj_id, obj_data in self.tracked_objects.items():
            if obj_data['frames_since_seen'] > self.max_frames:
                to_remove.append(obj_id)
        
        for obj_id in to_remove:
            del self.tracked_objects[obj_id]
            self.confirmed_objects.discard(obj_id)
        
        # Get stable/confirmed objects for display
        stable_objects = []
        for obj_id, obj_data in self.tracked_objects.items():
            if (obj_data['detection_count'] >= self.min_frames and 
                obj_data['frames_since_seen'] <= 2):  # Recently seen
                if obj_id not in self.confirmed_objects:
                    self.confirmed_objects.add(obj_id)
                    # This is a new confirmed detection
                    new_detections.append({
                        'id': obj_id,
                        'class_name': obj_data['class_name'],
                        'confidence': obj_data['confidence'],
                        'box': obj_data['box'],
                        'frame_num': frame_num
                    })
                stable_objects.append((obj_id, obj_data))
        
        return stable_objects, new_detections
    
    def get_current_counts(self):
        """Get current count of confirmed objects by class"""
        counts = defaultdict(int)
        for obj_id in self.confirmed_objects:
            if obj_id in self.tracked_objects:
                class_name = self.tracked_objects[obj_id]['class_name']
                counts[class_name] += 1
        return dict(counts)

def draw_detections(frame, stable_objects, tracker):
    """Draw bounding boxes and labels on frame for stable objects only"""
    detections_count = defaultdict(int)
    
    for obj_id, obj_data in stable_objects:
        if obj_data['frames_since_seen'] <= 2:  # Only draw recently seen objects
            x1, y1, x2, y2 = obj_data['box']
            confidence = obj_data['confidence']
            class_id = obj_data['class_id']
            class_name = obj_data['class_name']
            detection_count = obj_data['detection_count']
            
            # Count detections
            detections_count[class_name] += 1
            
            # Get color for this class
            color = colors.get(class_id, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Draw label with confidence and detection count
            label = f"{class_name}: {confidence:.2f} (#{obj_id})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Draw label background
            cv2.rectangle(frame, 
                        (int(x1), int(y1) - label_size[1] - 10),
                        (int(x1) + label_size[0], int(y1)), 
                        color, -1)
            
            # Draw label text
            cv2.putText(frame, label, 
                      (int(x1), int(y1) - 5), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw small indicator for tracking stability
            stability_color = (0, 255, 0) if detection_count >= MIN_DETECTION_FRAMES else (0, 255, 255)
            cv2.circle(frame, (int(x2) - 10, int(y1) + 10), 5, stability_color, -1)
    
    return dict(detections_count)

# Detection tracking variables
results_list = []
frame_count = 0
processed_frame_count = 0
detection_times = []
total_detections = defaultdict(int)
unique_detections = defaultdict(int)

# Initialize object tracker
tracker = ObjectTracker(
    iou_threshold=IOU_THRESHOLD,
    max_frames=MAX_TRACKING_FRAMES,
    min_frames=MIN_DETECTION_FRAMES
)

print("Starting real-time video detection...")
print("Controls:")
print("- Press 'q' to quit")
print("- Press 's' to save current frame")
print("- Press 'r' to toggle video recording")
print("- Close window to exit")

try:
    consecutive_failures = 0
    max_failures = 5
    
    while True:
        try:
            # Read frame from camera
            ret, frame = cap.read()
            
            # Handle frame reading issues
            if not ret or frame is None or frame.size == 0:
                consecutive_failures += 1
                print(f"Warning: Failed to grab frame (attempt {consecutive_failures}/{max_failures})")
                
                if consecutive_failures >= max_failures:
                    print("Too many consecutive failures. Attempting to reconnect camera...")
                    cap.release()
                    time.sleep(1)
                    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
                    if not cap.isOpened():
                        print("Failed to reconnect camera. Exiting.")
                        break
                    consecutive_failures = 0
                    print("Camera reconnected successfully.")
                
                time.sleep(0.1)
                continue
            
            consecutive_failures = 0
            frame_count += 1
            
            # Process frame for detection (not every frame for performance)
            current_detections = {}
            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                processed_frame_count += 1
                
                # Run YOLO detection
                start_time = time.time()
                results = model(
                    source=frame,
                    conf=DETECTION_CONFIDENCE,
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                    verbose=False  # Reduce console output
                )
                
                detection_time = time.time() - start_time
                detection_times.append(detection_time)
                
                # Extract detections for tracking
                raw_detections = []
                for r in results:
                    boxes = r.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            confidence = box.conf[0].item()
                            class_id = int(box.cls[0].item())
                            class_name = model.names[class_id]
                            
                            raw_detections.append([x1, y1, x2, y2, confidence, class_id, class_name])
                
                # Update tracker
                stable_objects, new_confirmed_detections = tracker.update(raw_detections, frame_count)
                
                # Draw stable objects only
                current_detections = draw_detections(frame, stable_objects, tracker)
                
                # Update counters for new confirmed detections only
                for new_detection in new_confirmed_detections:
                    class_name = new_detection['class_name']
                    unique_detections[class_name] += 1
                    
                    # Save detection data if enabled
                    if SAVE_DETECTIONS:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                        x1, y1, x2, y2 = new_detection['box']
                        
                        result_item = {
                            'object_id': new_detection['id'],
                            'frame_number': frame_count,
                            'timestamp': timestamp,
                            'class': class_name,
                            'confidence': round(new_detection['confidence'], 3),
                            'x1': round(x1, 2),
                            'y1': round(y1, 2),
                            'x2': round(x2, 2),
                            'y2': round(y2, 2),
                            'detection_type': 'unique_confirmed'
                        }
                        results_list.append(result_item)
            
            # Add information overlay
            height, width = frame.shape[:2]
            
            # Performance info
            if detection_times:
                avg_detection_time = np.mean(detection_times[-10:])  # Last 10 detections
                fps_estimate = 1.0 / avg_detection_time if avg_detection_time > 0 else 0
                cv2.putText(frame, f"Detection FPS: {fps_estimate:.1f}", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.putText(frame, f"Frame: {frame_count} | Processed: {processed_frame_count}", 
                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Current detections info (stable objects currently visible)
            y_offset = 90
            cv2.putText(frame, "Currently Visible:", 
                      (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 25
            for class_name, count in current_detections.items():
                cv2.putText(frame, f"  {class_name}: {count}", 
                          (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                y_offset += 20
            
            # Unique confirmed detections info
            if unique_detections:
                cv2.putText(frame, "Unique Objects Found:", 
                          (width - 220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                y_offset = 50
                for class_name, total_count in unique_detections.items():
                    cv2.putText(frame, f"  {class_name}: {total_count}", 
                              (width - 220, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    y_offset += 20
            
            # Tracking info
            total_tracked = len(tracker.confirmed_objects)
            cv2.putText(frame, f"Objects Tracked: {total_tracked}", 
                      (width - 220, y_offset + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            
            # Controls info
            cv2.putText(frame, "Press 'q' to quit, 's' to save frame, 'r' to record", 
                      (10, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, "Green dots = stable tracking, Yellow dots = new objects", 
                      (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Display the frame
            cv2.imshow('Real-time Corn Seed Detection', frame)
            
            # Save to video if recording
            if video_writer is not None:
                video_writer.write(frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or cv2.getWindowProperty('Real-time Corn Seed Detection', cv2.WND_PROP_VISIBLE) < 1:
                print("Quit command received. Exiting.")
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                img_filename = f"saved_frame_{timestamp}_{frame_count}.jpg"
                img_path = str(results_dir / "video_frames" / img_filename)
                cv2.imwrite(img_path, frame)
                print(f"Frame saved to {img_path}")
            elif key == ord('r'):
                # Toggle video recording
                if video_writer is None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    video_path = results_dir / f"detection_video_{timestamp}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(str(video_path), fourcc, 20, (width, height))
                    print(f"Started recording video to: {video_path}")
                else:
                    video_writer.release()
                    video_writer = None
                    print("Stopped recording video")
                    
        except cv2.error as e:
            print(f"OpenCV error: {str(e)}")
            time.sleep(0.1)
            continue
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            time.sleep(0.1)
            continue

except KeyboardInterrupt:
    print("Interrupted by user")
    
finally:
    # Cleanup
    if cap is not None:
        cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()
    
    # Save detection results
    if results_list:
        df = pd.DataFrame(results_list)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_path = results_dir / f"video_detection_results_{timestamp}.csv"
        df.to_csv(results_path, index=False)
        print(f"Saved detailed results to {results_path}")
        
        # Create and save summary
        summary = df.groupby('class').agg({
            'frame_number': 'count',
            'confidence': ['mean', 'min', 'max']
        }).round(3)
        summary.columns = ['detection_count', 'avg_confidence', 'min_confidence', 'max_confidence']
        
        summary_path = results_dir / f"video_detection_summary_{timestamp}.csv"
        summary.to_csv(summary_path)
        print(f"Saved summary to {summary_path}")
        
        print("\nFinal Detection Summary:")
        print("Unique confirmed objects (avoiding duplicates):")
        for class_name, count in unique_detections.items():
            print(f"  {class_name}: {count} unique objects")
        
        print(f"\nDetailed Detection Summary:")
        print(summary)
        
        # Performance statistics
        if detection_times:
            avg_detection_time = np.mean(detection_times)
            print(f"\nPerformance Statistics:")
            print(f"Average detection time: {avg_detection_time:.3f} seconds")
            print(f"Average detection FPS: {1.0/avg_detection_time:.1f}")
            print(f"Total frames processed: {processed_frame_count}")
            print(f"Total frames captured: {frame_count}")
    else:
        print("No detections were made during the video session")
    
    print("Video detection session ended.")