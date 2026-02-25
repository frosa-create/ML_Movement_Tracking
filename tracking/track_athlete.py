import cv2
import numpy as np
from ultralytics import YOLO

VIDEO_PATH = "/Users/ryanrosa/Downloads/AI_DS_Project_Repos/ML_Movement_Tracking/J2_vs_Mem_Martins_1.mov"
MODEL_PATH = 'templates/yolov8n-pose.pt'

IOU_KEEP_THRESH = 0.05 # if best IOU drops below this the track is lost

model = YOLO(MODEL_PATH)

clicked_pt = {"xy": None}

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_pt["xy"] = (x, y)

def iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter <= 0:
        return 0.0
    areaA = (a[2] - a[0]) * (a[3] - a[1])
    areaB = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (areaA + areaB - inter + 1e-9)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

win = "Select athlete (click), press 'n' next frame, ESC quit"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(win, on_mouse)

target_box = None #xyxy float
target_kpts = None

#Selection

while True:
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Reached end of video before selecting a target.")

    res = model.predict(
        frame,
        classes = [0],
        imgsz = 960,
        conf = 0.2,
        verbose = False
    )[0]  # person only
    xyxy = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4), dtype=float)

    vis = frame.copy()
    for b in xyxy.astype(int):
        cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 255, 255), 2)

    cv2.putText(vis, "Click athlete, then press any key. 'n' skips frame.",
                (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow(win, vis)

    key = cv2.waitKey(0) & 0xFF
    if key == 27:
        cap.release()
        cv2.destroyAllWindows()
        raise SystemExit

    if key == ord("n"):
        clicked_pt["xy"] = None
        continue

    if clicked_pt["xy"] is None:
        continue

    x, y = clicked_pt["xy"]
    clicked_pt["xy"] = None

    if len(xyxy) == 0:
        continue

    # choose the detection whose box center is closest to the click
    centers = np.column_stack((
        (xyxy[:, 0] + xyxy[:, 2]) / 2.0,
        (xyxy[:, 1] + xyxy[:, 3]) / 2.0
    ))
    click_pt = np.array([x, y], dtype=float)
    dists = np.linalg.norm(centers - click_pt, axis=1)

    best_idx = int(np.argmin(dists))
    max_click_dist = 50.0  # max distance in pixels to accept a selection
    if dists[best_idx] > max_click_dist:
        continue

    target_box = xyxy[best_idx]
    print("Selected target box:", target_box)
    break

# Tracking Phase
while True:
    ret, frame = cap.read()
    if not ret:
        break

    res = model.predict(
        frame,
        classes = [0],
        imgsz = 960,
        conf = 0.2,
        verbose = False
    )[0]
    xyxy = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4), dtype=float)

    vis = frame.copy()

    if len(xyxy) == 0 or target_box is None:
        cv2.putText(vis, "No detections / target lost. Press 'r' to reselect",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    else:
        ious = np.array([iou(target_box, b) for b in xyxy], dtype=float)
        best_idx = int(np.argmax(ious))
        best_iou = float(ious[best_idx])

        tx1, ty1, tx2, ty2 = target_box
        t_cx, t_cy = (tx1 + tx2) / 2.0, (ty1 + ty2) / 2.0
        cx = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
        cy = (xyxy[:, 1] + xyxy[:, 3]) / 2.0
        centers = np.column_stack((cx, cy))
        t_center = np.array([t_cx, t_cy], dtype = float)
        dists = np.linalg.norm(centers - t_center, axis = 1)
        best_dist = float(dists[best_idx])

        CENTER_THRESH = 80.0  #pixel amount

        if best_iou >= IOU_KEEP_THRESH or best_dist <= CENTER_THRESH:
            # smooth box update so tracking doesn't break when it's closer
            new_box = xyxy[best_idx]
            target_box = 0.7 * target_box + 0.3 * new_box

            b = target_box.astype(int)
            cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 3)
            cv2.putText(vis, f"Athlete (IoU {best_iou:.2f}, d {best_dist:.1f})",
                        (b[0], max(20, b[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        else:
            cv2.putText(vis, f"Target lost (best IoU {best_iou:.2f}, d {best_dist:.1f}). Press 'r' to reselect.",
            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow(win, vis)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    if key == ord("r"):
        target_box = None
        while target_box is None:
            res = model.predict(frame, classes=[0], verbose=False)[0]
            xyxy = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4), dtype=float)

            vis2 = frame.copy()
            for b in xyxy.astype(int):
                cv2.rectangle(vis2, (b[0], b[1]), (b[2], b[3]), (0, 255, 255), 2)
            cv2.putText(vis2, "Reselect: click athlete, press any key. 'n' cancels reselect.",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow(win, vis2)

            k2 = cv2.waitKey(0) & 0xFF
            if k2 == ord("n") or k2 == 27:
                break
            if clicked_pt["xy"] is None:
                continue

            x, y = clicked_pt["xy"]
            clicked_pt["xy"] = None
            inside = [i for i, (x1, y1, x2, y2) in enumerate(xyxy) if x1 <= x <= x2 and y1 <= y <= y2]
            if inside:
                areas = [(xyxy[i][2] - xyxy[i][0]) * (xyxy[i][3] - xyxy[i][1]) for i in inside]
                idx = inside[int(np.argmin(areas))]
                target_box = xyxy[idx]


cap.release()
cv2.destroyAllWindows()

