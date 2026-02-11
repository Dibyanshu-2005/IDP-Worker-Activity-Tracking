from ultralytics import YOLO

def run_inference(video_path):
    model = YOLO("yolov8l.pt")  # yolov8l is solid for accuracy/speed balance
    
    model.track(  # Changed from predict()
        source=video_path,
        classes=[0],  # Person only
        persist=True,  # Key: Maintains IDs across frames
        conf=0.25,
        iou=0.7,  # Tune for overlaps
        save=True,  # Saves video + labels
        show=True,
        project="runs/detect",  # Output folder
        name="track_experiment"  # Unique run name
    )
    print("Tracked video saved in runs/detect/track_experiment/")  # Check here

if __name__ == "__main__":
    run_inference("input.mp4")
