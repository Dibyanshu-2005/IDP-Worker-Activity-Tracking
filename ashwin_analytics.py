import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from ultralytics import YOLO

# ==========================================
# 1. SETUP, PATHS & CONSTRAINTS
# ==========================================
VIDEO_PATH = "../videos/00000001893000000_UF.mp4"
FRAME_SKIP = 6
PADDING_KERNEL = np.ones((301, 151), np.uint8)

video_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
OUTPUT_DIR = os.path.join("outputs", video_name)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Exact labels matching your phase_detection.pt model
MIN_WORKERS_PER_PHASE = {
    "mould_cleaning": 1,
    "rebar_cage_placement": 2,
    "concrete_pouring": 1,
    "surface_finishing": 1,
    "curing": 0,
    "vacuum_lifting": 2
}
DEFAULT_MIN_WORKERS = 1

# ==========================================
# 2. LOAD MODELS
# ==========================================
print("Loading AI Models...")
mould_model = YOLO("../smart_mould_detection/finetuned_multi.pt")
phase_model = YOLO("best.pt")
worker_model = YOLO("../yolo_model_weights/yolo26m.pt")

# ==========================================
# 3. INITIALIZE VIDEO & INTERACTIVE UI
# ==========================================
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)

ret, first_frame = cap.read()
if not ret:
    raise ValueError("Cannot read video file.")

# --- THE INTERACTIVE POLYLINE UI ---
points = []
first_frame_display = first_frame.copy()
temp_display = first_frame.copy()

def select_points(event, x, y, flags, param):
    global points, first_frame_display, temp_display

    # 1. LIVE PREVIEW: Draw yellow line following the mouse
    if event == cv2.EVENT_MOUSEMOVE:
        if len(points) > 0:
            temp_display = first_frame_display.copy()
            cv2.line(temp_display, points[-1], (x, y), (0, 255, 255), 2)
            cv2.imshow("Select Tracking Boundary", temp_display)

    # 2. LOCK POINT: Click to drop a dot and draw the green line
    elif event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(first_frame_display, (x, y), 5, (0, 0, 255), -1)

        if len(points) > 1:
            cv2.line(first_frame_display, points[-2], points[-1], (0, 255, 0), 2)

        temp_display = first_frame_display.copy()
        cv2.imshow("Select Tracking Boundary", temp_display)

cv2.namedWindow("Select Tracking Boundary", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Select Tracking Boundary", 1280, 720)
cv2.setMouseCallback("Select Tracking Boundary", select_points)

print("\n🛑 PAUSED: Please look at the popup window.")
print("1. Click multiple points from Left to Right to draw your boundary.")
print("2. A yellow preview line will follow your mouse. Clicks lock it in green.")
print("3. Only workers visually BELOW this line will be tracked.")
print("4. Press 'Enter' to confirm and start video processing.")

while True:
    cv2.imshow("Select Tracking Boundary", temp_display)
    key = cv2.waitKey(1) & 0xFF
    if key == 13 and len(points) >= 2: # 13 is Enter key
        break
    elif key == ord('q'):
        print("Canceled by user.")
        exit()

cv2.destroyWindow("Select Tracking Boundary")

# Mathematical helper using pure NumPy to evaluate the connected line segments
def get_boundary_y(cx, pts_list):
    # Sort points by X coordinate so the math flows correctly from left to right
    sorted_pts = sorted(pts_list, key=lambda p: p[0])
    xs = [p[0] for p in sorted_pts]
    ys = [p[1] for p in sorted_pts]
    # np.interp connects the dots with straight lines and flatlines the edges
    return np.interp(cx, xs, ys)

# Tracking Variables
heatmap_data = np.zeros((first_frame.shape[0], first_frame.shape[1]), dtype=np.float32)
timeline_history = {}
frame_count = 0

window_name = "Productivity Tracker Live"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1280, 720)

print("\n🚀 Starting Video Processing...")

# ==========================================
# 4. MAIN PROCESSING LOOP
# ==========================================
while cap.isOpened():
    ret, frame = cap.read()
    frame_count += 1
    if not ret: break

    if frame_count % FRAME_SKIP != 0:
        continue

    annotated_frame = frame.copy()

    # Draw the polyline boundary on the live feed
    for i in range(1, len(points)):
        cv2.line(annotated_frame, points[i-1], points[i], (0, 255, 0), 2)

    # --- A. MOULD ZONES ---
    mould_results = mould_model.predict(frame, imgsz=640, verbose=False)
    exclusive_zones = []

    if mould_results[0].masks is not None:
        masks = mould_results[0].masks.data.cpu().numpy()
        boxes = mould_results[0].boxes.xyxy.cpu().numpy()

        sorted_indices = np.argsort(boxes[:, 0])
        masks = masks[sorted_indices]

        raw_dilated_masks = []
        for mask in masks:
            mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
            dilated = cv2.dilate(mask_resized, PADDING_KERNEL, iterations=1)
            raw_dilated_masks.append(dilated)

        sum_mask = np.sum(raw_dilated_masks, axis=0)
        overlap_areas = (sum_mask > 1).astype(np.uint8)

        for i, dilated in enumerate(raw_dilated_masks):
            exclusive_mask = cv2.bitwise_and(dilated, dilated, mask=cv2.bitwise_not(overlap_areas))
            exclusive_zones.append(exclusive_mask)

            if i not in timeline_history:
                timeline_history[i] = []

            blue_tint = np.array([255, 150, 0], dtype=np.float32)
            annotated_frame[exclusive_mask > 0] = (annotated_frame[exclusive_mask > 0].astype(np.float32) * 0.7 + blue_tint * 0.3).astype(np.uint8)

    # --- B. PHASE DETECTION (BACKGROUND ONLY) ---
    current_mould_phases = {i: "idle" for i in range(len(exclusive_zones))}

    phase_results = phase_model.predict(frame, imgsz=640, verbose=False)

    if phase_results[0].boxes is not None:
        phase_boxes = phase_results[0].boxes.xyxy.cpu().numpy()
        phase_classes = phase_results[0].boxes.cls.cpu().numpy()
        names = phase_model.names

        for box, cls_id in zip(phase_boxes, phase_classes):
            phase_name = names[int(cls_id)]
            px1, py1, px2, py2 = map(int, box)
            center_x, center_y = int((px1 + px2) / 2), int((py1 + py2) / 2)

            cv2.rectangle(annotated_frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"PHASE: {phase_name}", (px1, max(20, py1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            for i, zone in enumerate(exclusive_zones):
                if zone[center_y, center_x] > 0:
                    current_mould_phases[i] = phase_name
                    break

    # --- C. WORKER TRACKING & COUNTING ---
    workers_in_zone_count = {i: 0 for i in range(len(exclusive_zones))}

    worker_results = worker_model.track(frame, classes=[0], persist=True, imgsz=640,conf=0.5, verbose=False)

    if worker_results[0].boxes.id is not None:
        boxes = worker_results[0].boxes.xyxy.cpu().numpy()
        ids = worker_results[0].boxes.id.cpu().numpy()

        for box, w_id in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)

            box_width = x2 - x1
            box_height = y2 - y1
            if box_width > 100 or box_height > 200:
                continue

            centroid_x, centroid_y = int((x1 + x2) / 2), int((y1 + y2) / 2)

            # ✅ THE POLYLINE BOUNDARY CHECK
            boundary_y = get_boundary_y(centroid_x, points)
            if centroid_y < boundary_y:
                continue # Visually above the line, ignore them

            if 0 <= centroid_y < heatmap_data.shape[0] and 0 <= centroid_x < heatmap_data.shape[1]:
                heatmap_data[centroid_y, centroid_x] += 1

            worker_status = "idle"
            for i, zone in enumerate(exclusive_zones):
                if zone[centroid_y, centroid_x] > 0:
                    workers_in_zone_count[i] += 1
                    worker_status = current_mould_phases[i]
                    break

            color = (0, 165, 255) if worker_status != "idle" else (0, 0, 255)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID: {int(w_id)} [{worker_status}]"
            cv2.putText(annotated_frame, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # --- D. THE LABOR GATING LOGIC ---
    for i in range(len(exclusive_zones)):
        detected_phase = current_mould_phases.get(i, "idle")
        worker_count = workers_in_zone_count[i]

        required_workers = MIN_WORKERS_PER_PHASE.get(detected_phase, DEFAULT_MIN_WORKERS)

        if detected_phase == "curing":
            final_timeline_state = "curing"
        elif worker_count >= required_workers:
            final_timeline_state = detected_phase if detected_phase != "idle" else "Active (Unclassified Phase)"
        else:
            final_timeline_state = "idle"

        timeline_history[i].append(final_timeline_state)

    # --- E. SHOW LIVE FEED ---
    cv2.imshow(window_name, annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("\nPlayback stopped by user.")
        break

cap.release()
cv2.destroyAllWindows()

# ==========================================
# 5. GENERATE HEATMAP
# ==========================================
print("\n📊 Generating Dashboards...")

heatmap_blurred = cv2.GaussianBlur(heatmap_data, (99, 99), 0)
heatmap_norm = cv2.normalize(heatmap_blurred, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

colored_heatmap = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
heatmap_mask = (heatmap_norm > 5).astype(np.float32)[:, :, np.newaxis]

overlay = first_frame.copy()
overlay = (overlay * (1 - heatmap_mask * 0.6) + colored_heatmap * (heatmap_mask * 0.6)).astype(np.uint8)

cv2.imwrite(os.path.join(OUTPUT_DIR, "motion_heatmap_overlay.jpg"), overlay)


# ==========================================
# 6. GENERATE VERTICAL CREW BALANCE CHART
# ==========================================
fig, ax = plt.subplots(figsize=(max(8, len(timeline_history) * 2), 10))

phase_colors = {
    "idle": "#d3d3d3",
    "mould_cleaning": "#1f77b4",
    "rebar_cage_placement": "#ff7f0e",
    "concrete_pouring": "#2ca02c",
    "surface_finishing": "#9467bd",
    "curing": "#8c564b",
    "vacuum_lifting": "#e377c2",
    "Active (Unclassified Phase)": "#ffff00"
}

for mould_id, history in timeline_history.items():
    if len(history) == 0: continue

    current_phase = history[0]
    start_frame = 0

    for current_frame, phase in enumerate(history):
        if phase != current_phase or current_frame == len(history) - 1:
            start_sec = (start_frame * FRAME_SKIP) / fps
            duration_sec = ((current_frame - start_frame) * FRAME_SKIP) / fps

            color = phase_colors.get(current_phase, "#000000")
            ax.bar(x=mould_id, height=duration_sec, bottom=start_sec, width=0.6, color=color, edgecolor='none')

            current_phase = phase
            start_frame = current_frame

ax.set_xlim(-0.5, len(timeline_history) - 0.5)

exact_video_length_seconds = frame_count / fps
ax.set_ylim(0, exact_video_length_seconds)
ax.invert_yaxis()

ax.set_ylabel('Time (Seconds)', fontsize=12)
ax.set_xlabel('Mould ID', fontsize=12)
ax.set_xticks(range(len(timeline_history)))
ax.set_xticklabels([f'Mould {i}' for i in range(len(timeline_history))])
ax.grid(True, axis='y', linestyle='--', alpha=0.7)
ax.set_title("True Crew Balance Timeline (Labor Constrained)", fontsize=14, pad=20)

legend_patches = [mpatches.Patch(color=color, label=phase) for phase, color in phase_colors.items()]
plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "crew_balance_chart.png"))
plt.close()

print(f"✅ Success! Your Labor-Gated Gantt Chart and Heatmap are saved in: {OUTPUT_DIR}")