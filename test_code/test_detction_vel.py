import cv2
import time
from ultralytics import YOLO

# -----------------------------
# Configuration
# -----------------------------
# MODEL_PATH = "/home/ubuntu/Documents/sandy_files/ANPR_web3/backend/models/vechile_plate_yolov8s.pt"
MODEL_PATH = "//home/facetag4/Documents/sandy_files/anpr_backend/models/vechile_plate_yolov8s.pt"
VIDEO_PATH = "/home/facetag4/Documents/sandy_files/anpr_backend/sample_video.mp4"

CONF_THRESHOLD = 0.25
FRAME_INTERVAL = 1

# -----------------------------
# Load Model
# -----------------------------
model = YOLO(MODEL_PATH)

print("\nClasses in the model:")
for class_id, class_name in model.names.items():
    print(f"{class_id}: {class_name}")
print()

# -----------------------------
# Generate a unique color for each class
# -----------------------------
CLASS_COLORS = {}

for class_id in model.names.keys():
    CLASS_COLORS[class_id] = (
        (37 * class_id) % 256,   # Blue
        (17 * class_id) % 256,   # Green
        (97 * class_id) % 256    # Red
    )

print("Class Colors:")
for class_id, class_name in model.names.items():
    print(f"{class_name} -> {CLASS_COLORS[class_id]}")
print()

# -----------------------------
# Open Video
# -----------------------------
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error opening video")
    exit()

frame_count = 0
prev_time = time.time()

cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO Detection", 960, 540)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    detection_count = 0
    inference_time = 0

    # -----------------------------------------
    # Run inference every FRAME_INTERVAL frames
    # -----------------------------------------
    if frame_count % FRAME_INTERVAL == 0:

        start = time.perf_counter()

        results = model(
            frame,
            conf=CONF_THRESHOLD,
            verbose=False
        )

        inference_time = (time.perf_counter() - start) * 1000

        for result in results:

            for box in result.boxes:

                detection_count += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                confidence = float(box.conf[0])

                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                # Get class color
                color = CLASS_COLORS[class_id]

                label = f"{class_name} {confidence:.2f}"

                # Draw bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                # Draw filled label background
                (tw, th), baseline = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    2
                )

                cv2.rectangle(
                    frame,
                    (x1, y1 - th - 10),
                    (x1 + tw + 5, y1),
                    color,
                    -1
                )

                # Draw label text
                cv2.putText(
                    frame,
                    label,
                    (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

    # -----------------------------
    # FPS Calculation
    # -----------------------------
    current_time = time.time()
    fps = 1.0 / (current_time - prev_time)
    prev_time = current_time

    # -----------------------------
    # Display Information
    # -----------------------------
    cv2.putText(
        frame,
        f"FPS : {fps:.2f}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Inference : {inference_time:.2f} ms",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Detections : {detection_count}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Frame : {frame_count}",
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Frame Interval : {FRAME_INTERVAL}",
        (20, 175),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 255),
        2
    )

    status = (
        "PROCESSED"
        if frame_count % FRAME_INTERVAL == 0
        else "SKIPPED"
    )

    status_color = (
        (0, 255, 0)
        if frame_count % FRAME_INTERVAL == 0
        else (0, 0, 255)
    )

    cv2.putText(
        frame,
        status,
        (20, 210),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        status_color,
        2
    )

    cv2.imshow("YOLO Detection", frame)

    key = cv2.waitKey(1)

    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()