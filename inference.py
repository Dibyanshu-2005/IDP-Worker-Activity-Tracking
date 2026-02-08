from ultralytics import YOLO

def run_inference(video_path):
    model = YOLO("yolov8l.pt") # Trying a better version 
    
    '''
    I tried yolov8 nano, medium and large, we have to select one, and understand 
    the trade-off between accuracy and speed. Would be great if you can test with yolo11 family models.
    '''

    model.predict(
        source=video_path,
        classes=[0],  # Only detect person class
        show=True,      
        save=True,
        conf=0.25
    )

if __name__ == "__main__":
    run_inference("input.mp4")
