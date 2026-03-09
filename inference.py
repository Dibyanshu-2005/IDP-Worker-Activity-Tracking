import cv2
import math
from ultralytics import YOLO

def run_inference(video_path):
    model = YOLO("yolo26m.pt")
    cap = cv2.VideoCapture(video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    max_lost_frames = fps * 3  # Time to remember lost IDs (e.g., 3 seconds)
    distance_threshold = 300   # Max pixel distance to reconnect an ID
    
    out = cv2.VideoWriter("track5.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    # Memory for workers: {id: (center_x, center_y, last_seen_frame)}
    track_memory = {} 
    id_mapping = {}    # Format: {new_id: old_id}

    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1
        results = model.track(frame, classes=[0], persist=True, tracker="./botsort.yaml", verbose=False)
        
        current_frame_ids = []

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                
                # Apply mapping if we already linked this new ID to an old one
                current_id = id_mapping.get(track_id, track_id)
                current_frame_ids.append(current_id)
                
                # If this is a completely new ID, check if it belongs to a recently lost worker
                if track_id not in id_mapping and current_id not in track_memory:
                    best_match_id = None
                    min_dist = float('inf')
                    
                    for lost_id, (lx, ly, lost_frame) in list(track_memory.items()):
                        # Only check IDs that are NOT in the current frame
                        if lost_id not in current_frame_ids:
                            frames_passed = frame_count - lost_frame
                            dist = math.hypot(center_x - lx, center_y - ly)
                            
                            # If within time limit and distance limit, reconnect
                            if frames_passed < max_lost_frames and dist < distance_threshold:
                                if dist < min_dist:
                                    min_dist = dist
                                    best_match_id = lost_id
                                
                    if best_match_id is not None:
                        id_mapping[track_id] = best_match_id
                        current_id = best_match_id
                
                # Update the latest position of this active worker
                track_memory[current_id] = (center_x, center_y, frame_count)

                # Draw boxes and IDs
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {current_id}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Cleanup memory: Remove IDs that have been lost for longer than our threshold
        for memory_id, (lx, ly, last_frame) in list(track_memory.items()):
            if memory_id not in current_frame_ids and (frame_count - last_frame) > max_lost_frames:
                del track_memory[memory_id]

        out.write(frame)
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Tracked video saved as track5.mp4")

if __name__ == "__main__":
    run_inference("input.mp4")