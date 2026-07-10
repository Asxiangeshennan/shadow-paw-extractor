import os
import cv2
import numpy as np
from PIL import Image
from rembg import remove, new_session
from ultralytics import YOLO
import torch
import torchvision
import torchvision.transforms as T

INPUT_DIR = "input_images"
OUTPUT_DIR = "output_masks"

ANIMAL_IDS = set(range(16, 26))
yolo = YOLO("yolov8n.pt")
yolo_seg = YOLO("yolov8n-seg.pt")
session = new_session("isnet-general-use")
session_u2 = new_session("u2net")

device = "cuda" if torch.cuda.is_available() else "cpu"
deeplab = torchvision.models.segmentation.deeplabv3_resnet50(weights="DEFAULT")
deeplab.eval().to(device)
VOC_ANIMALS = {3: "bird", 8: "cat", 10: "cow", 12: "dog", 13: "horse", 17: "sheep"}

transform = T.Compose([
    T.ToPILImage(),
    T.Resize(520),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def detect_animal_bboxes(img):
    results = yolo(img, verbose=False)
    boxes = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls in ANIMAL_IDS:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append((conf, x1, y1, x2, y2))
    return boxes


def merge_boxes(boxes):
    if not boxes:
        return None
    x1 = min(b[1] for b in boxes)
    y1 = min(b[2] for b in boxes)
    x2 = max(b[3] for b in boxes)
    y2 = max(b[4] for b in boxes)
    return (x1, y1, x2, y2)


def keep_largest(mask):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask
    sizes = stats[1:, -1]
    if len(sizes) == 0:
        return mask
    return np.where(labels == (np.argmax(sizes) + 1), 255, 0).astype(np.uint8)


def convex_hull_fill(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    all_pts = np.vstack(contours)
    hull = cv2.convexHull(all_pts)
    hull_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.fillPoly(hull_mask, [hull], 255)
    return hull_mask


def get_rembg_mask(img_rgb):
    best_mask = None
    best_verts = 0
    for use_alpha in [False, True]:
        params = dict(alpha_matting=use_alpha)
        if use_alpha:
            params.update(alpha_matting_foreground_threshold=240,
                          alpha_matting_background_threshold=10,
                          alpha_matting_erode_size=2)
        res = remove(Image.fromarray(img_rgb), session=session, **params)
        m = np.array(res)[:, :, 3]
        _, m = cv2.threshold(m, 10, 255, cv2.THRESH_BINARY)
        m = keep_largest(m)
        c, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if c:
            verts = sum(len(ci) for ci in c)
            if verts > best_verts:
                best_verts = verts
                best_mask = m

    res = remove(Image.fromarray(img_rgb), session=session_u2, alpha_matting=False)
    m = np.array(res)[:, :, 3]
    _, m = cv2.threshold(m, 10, 255, cv2.THRESH_BINARY)
    m = keep_largest(m)
    c, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if c:
        verts = sum(len(ci) for ci in c)
        if verts > best_verts:
            best_verts = verts
            best_mask = m

    return best_mask


def get_deeplab_pred(img_rgb):
    h, w = img_rgb.shape[:2]
    input_tensor = transform(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        output = deeplab(input_tensor)["out"][0]
    pred = output.argmax(0).cpu().numpy()
    pred = cv2.resize(pred.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return pred


def get_deeplab_mask(img_rgb):
    pred = get_deeplab_pred(img_rgb)
    mask = np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.uint8)
    for aid in VOC_ANIMALS:
        mask[pred == aid] = 255
    if np.sum(mask == 255) < 100:
        return None
    return mask


def get_yolo_seg_mask(img):
    results = yolo_seg(img, verbose=False)
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for r in results:
        if r.masks is None:
            continue
        for i, box in enumerate(r.boxes):
            cls = int(box.cls[0])
            if cls not in ANIMAL_IDS:
                continue
            seg_mask = r.masks.data[i].cpu().numpy()
            seg_mask = cv2.resize(seg_mask, (img.shape[1], img.shape[0]))
            mask = cv2.bitwise_or(mask, (seg_mask > 0.5).astype(np.uint8) * 255)
    mask = keep_largest(mask)
    if np.sum(mask == 255) < 100:
        return None
    return mask


def refine_mask(mask, gray_img, use_hull=False):
    kernel = np.ones((3, 3), np.uint8)

    if use_hull:
        mask = convex_hull_fill(mask)

    gx = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, strong_edge = cv2.threshold(mag, 10, 255, cv2.THRESH_BINARY)

    dilated = cv2.dilate(mask, kernel, iterations=6)
    boundary = cv2.bitwise_xor(dilated, mask)
    keep = cv2.bitwise_and(boundary, strong_edge)
    result = cv2.bitwise_or(mask, keep)
    result = keep_largest(result)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=2)
    return result


def crop_and_rembg(img, gray, merged_box, box_ratio):
    mb_x1, mb_y1, mb_x2, mb_y2 = merged_box
    box_w, box_h = mb_x2 - mb_x1, mb_y2 - mb_y1
    use_hull = box_ratio > 0.7

    if use_hull:
        cropped_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cropped_gray = gray
        crop_x1, crop_y1 = 0, 0
        crop_x2, crop_y2 = img.shape[1], img.shape[0]
    else:
        margin_factor = 0.15 if box_ratio > 0.4 else 0.3
        margin_x = int(box_w * margin_factor)
        margin_y = int(box_h * margin_factor)
        crop_x1 = max(0, mb_x1 - margin_x)
        crop_y1 = max(0, mb_y1 - margin_y)
        crop_x2 = min(img.shape[1], mb_x2 + margin_x)
        crop_y2 = min(img.shape[0], mb_y2 + margin_y)
        cropped = img[crop_y1:crop_y2, crop_x1:crop_x2]
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    crop_mask = get_rembg_mask(cropped_rgb)
    if crop_mask is not None:
        crop_mask = refine_mask(crop_mask, cropped_gray, use_hull)

    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    if crop_mask is not None:
        ch = cropped_rgb.shape[0]
        cw = cropped_rgb.shape[1]
        mask[crop_y1:crop_y1 + ch, crop_x1:crop_x1 + cw] = crop_mask

    # Constrain to crop area
    box_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    box_mask[crop_y1:crop_y2, crop_x1:crop_x2] = 255
    mask = cv2.bitwise_and(mask, box_mask)
    mask = keep_largest(mask)
    return mask


def process_image(input_path, output_path):
    img = cv2.imread(input_path)
    if img is None:
        print(f"  Failed to read")
        return

    h_full, w_full = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    yolo_boxes = detect_animal_bboxes(img)
    merged_box = merge_boxes(yolo_boxes) if yolo_boxes else None
    if merged_box:
        mb_x1, mb_y1, mb_x2, mb_y2 = merged_box
        print(f"  YOLO: {len(yolo_boxes)} detections, box=({mb_x1},{mb_y1},{mb_x2},{mb_y2})")

    # --- Priority 1: YOLOv8n-seg (most accurate) ---
    mask = get_yolo_seg_mask(img)
    if mask is not None:
        cov = np.sum(mask == 255) / mask.size
        print(f"  YOLOv8n-seg: ({cov*100:.0f}%)")

    # --- Priority 2: DeepLabV3 (VOC animals, min 10% coverage) ---
    if mask is None:
        deeplab_mask = get_deeplab_mask(img_rgb)
        if deeplab_mask is not None:
            deeplab_pred = get_deeplab_pred(img_rgb)
            present = [VOC_ANIMALS[c] for c in set(deeplab_pred.ravel()) if c in VOC_ANIMALS]
            cov = np.sum(deeplab_mask == 255) / deeplab_mask.size
            print(f"  DeepLabV3: {','.join(present) if present else 'animal'} ({cov*100:.0f}%)")
            if cov > 0.10:
                mask = deeplab_mask
                if merged_box:
                    box_mask = np.zeros((h_full, w_full), dtype=np.uint8)
                    box_mask[mb_y1:mb_y2, mb_x1:mb_x2] = 255
                    mask = cv2.bitwise_and(mask, box_mask)
                mask = keep_largest(mask)

    # --- Priority 3: YOLO detection + rembg crop ---
    if mask is None and merged_box is not None:
        box_w, box_h = mb_x2 - mb_x1, mb_y2 - mb_y1
        box_ratio = box_w * box_h / (w_full * h_full)
        mask = crop_and_rembg(img, gray, merged_box, box_ratio)
        print(f"  Crop+rembg ({np.sum(mask==255)/mask.size*100:.0f}%)")

    # --- Priority 4: Full-image rembg ---
    if mask is None:
        mask = get_rembg_mask(img_rgb)
        if mask is not None:
            mask = refine_mask(mask, gray, False)
            print(f"  Full rembg ({np.sum(mask==255)/mask.size*100:.0f}%)")

    if mask is None:
        mask = np.zeros((h_full, w_full), dtype=np.uint8)

    mask = keep_largest(mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        print(f"  No animal contour")
        size = max(h_full, w_full)
        Image.fromarray(np.ones((size, size), dtype=np.uint8) * 255, mode="L").save(output_path)
        return

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    size = max(w, h)
    cx, cy = x + w // 2, y + h // 2

    padding = int(size * 0.08) + 10
    square_size = size + 2 * padding
    if square_size % 2 != 0:
        square_size += 1

    half = square_size // 2

    x1 = cx - half
    y1 = cy - half
    x2 = cx + half
    y2 = cy + half

    square_mask = np.zeros((square_size, square_size), dtype=np.uint8)

    src_x1 = max(0, x1)
    src_y1 = max(0, y1)
    src_x2 = min(mask.shape[1], x2)
    src_y2 = min(mask.shape[0], y2)

    dst_x1 = max(0, -x1)
    dst_y1 = max(0, -y1)

    if src_x2 > src_x1 and src_y2 > src_y1:
        sh = src_y2 - src_y1
        sw = src_x2 - src_x1
        square_mask[dst_y1:dst_y1 + sh, dst_x1:dst_x1 + sw] = mask[src_y1:src_y2, src_x1:src_x2]

    result = 255 - square_mask

    black = 255 - result
    black = keep_largest(black)
    result = 255 - black

    Image.fromarray(result, mode="L").save(output_path)
    anim_pct = np.sum(result == 0) / result.size * 100
    print(f"  Saved ({anim_pct:.0f}% animal)")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    files = [f for f in os.listdir(INPUT_DIR)
             if os.path.splitext(f)[1].lower() in exts]
    files.sort()
    print(f"Found {len(files)} images\n")
    for i, fn in enumerate(files, 1):
        ip = os.path.join(INPUT_DIR, fn)
        op = os.path.join(OUTPUT_DIR, os.path.splitext(fn)[0] + ".png")
        print(f"[{i}/{len(files)}] {fn}")
        process_image(ip, op)
    print("\nDone!")


if __name__ == "__main__":
    main()
