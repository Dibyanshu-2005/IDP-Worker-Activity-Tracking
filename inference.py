from ultralytics import YOLO

def run_inference(video_path):
    model = YOLO("yolov8ngit i.pt")

    model.predict(
        source=video_path,
        show=True,      # 👈 LIVE DISPLAY WINDOW
        save=True,
        conf=0.25
    )

if __name__ == "__main__":
    run_inference("input.mp4")
