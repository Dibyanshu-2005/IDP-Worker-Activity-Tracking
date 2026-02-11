from ultralytics import YOLO

def run_inference(video_path):
    model = YOLO("yolov8l.pt") # Trying a better version 
    
    model.predict(
        source=video_path,
        classes=[0],  # Only detect person class
        show=True,      
        save=True,
        conf=0.25
    )

if __name__ == "__main__":
    run_inference("input.mp4")
