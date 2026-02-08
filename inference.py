from ultralytics import YOLO

def run_inference(video_path):
    model = YOLO("yolov8n.pt") # yolo version 8 nano model

    model.predict(
        source=video_path,
        show=True,      
        save=True,
        conf=0.25
    )

if __name__ == "__main__":
    run_inference("input.mp4")
