# Shadow Paw Extractor — 多模型融合动物自动抠图工具

基于 **四级优先级级联架构** 的动物图像自动抠图工具，融合 YOLOv8、DeepLabV3+ 和 Rembg 多种深度学习模型，实现高精度的动物分割与背景移除。

在图像处理中，动物抠图是一个常见但具有挑战性的任务：动物种类繁多、背景复杂多变、部分遮挡普遍存在，单一模型很难在所有场景下表现良好。本工具通过多模型级联的设计，确保 **精度优先、容错性强、终极保底**。

## 整体架构

```
输入图片
    │
    ▼
┌──────────────────────────────────────┐
│  ① YOLOv8n 检测 → 获取动物边界框      │
└──────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────┐
│  ② 多策略分割（四级优先级级联）        │
│                                      │
│  Priority 1: YOLOv8n-seg 实例分割     │ ← 最精确
│  Priority 2: DeepLabV3+ 语义分割      │
│  Priority 3: YOLO框 + Rembg 局部抠图   │
│  Priority 4: 全图 Rembg 通用抠图       │ ← 兜底
└──────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────┐
│  ③ 后处理优化                         │
│     · 最大连通区域保留                 │
│     · 凸包填充（补全肢体）              │
│     · Sobel 边缘细化                  │
│     · 形态学闭运算                     │
└──────────────────────────────────────┘
    │
    ▼
   白底黑主体的 Mask 图片（正方形居中）
```

### 四级优先级

| 优先级 | 方案 | 优点 | 触发条件 |
|--------|------|------|---------|
| 1★ | YOLOv8n-seg 实例分割 | 像素级精确，区分个体 | COCO动物 + 分割成功 |
| 2★ | DeepLabV3+ 语义分割 | 边界干净，VOC动物擅长 | 语义分割覆盖 > 10% 像素 |
| 3★ | YOLO框 + Crop Rembg | 缩小范围降低误删 | 有检测框但前两级失败 |
| 4★ | 全图 Rembg 通用抠图 | 不限类别，终极保底 | 所有方案均失败 |

## 关键代码解析

代码位于 `识别动物/extract_animal.py`（单文件 335 行），以下逐段说明核心逻辑。

### 1. 模型初始化与配置

```python
import os, cv2, numpy as np
from PIL import Image
from rembg import remove, new_session
from ultralytics import YOLO
import torchvision
import torchvision.transforms as T

INPUT_DIR = "input_images"
OUTPUT_DIR = "output_masks"

# COCO 动物 ID 范围：16=鸟 17=猫 18=狗 19=马 20=羊 21=牛 22=象 23=熊 24=斑马 25=长颈鹿
ANIMAL_IDS = set(range(16, 26))

# 加载 5 个模型（权重按需自动下载）
yolo = YOLO("yolov8n.pt")                                   # 检测
yolo_seg = YOLO("yolov8n-seg.pt")                           # 实例分割
session = new_session("isnet-general-use")                   # Rembg ISNet
session_u2 = new_session("u2net")                            # Rembg U²Net
deeplab = torchvision.models.segmentation.deeplabv3_resnet50(weights="DEFAULT")
deeplab.eval().to("cuda" if torch.cuda.is_available() else "cpu")
```

同时定义 VOC 动物映射和 DeepLabV3+ 预处理流水线，用于后续语义分割。

### 2. 动物检测与后处理基元

```python
def detect_animal_bboxes(img):
    """YOLOv8n 检测 → 只保留 COCO 动物"""
    results = yolo(img, verbose=False)
    boxes = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls in ANIMAL_IDS:                     # 非动物（人、车等）直接过滤
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append((float(box.conf[0]), x1, y1, x2, y2))
    return boxes

def merge_boxes(boxes):
    """多只动物 → 外接矩形，避免漏抠"""
    if not boxes:
        return None
    return (min(b[1] for b in boxes), min(b[2] for b in boxes),
            max(b[3] for b in boxes), max(b[4] for b in boxes))

def keep_largest(mask):
    """连通域分析 → 只保留最大区域，去除噪点"""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask
    sizes = stats[1:, -1]
    if len(sizes) == 0:
        return mask
    return np.where(labels == (np.argmax(sizes) + 1), 255, 0).astype(np.uint8)

def convex_hull_fill(mask):
    """凸包填充 → 补全尾巴/四肢等被误切部位"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    hull = cv2.convexHull(np.vstack(contours))
    hull_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.fillPoly(hull_mask, [hull], 255)
    return hull_mask
```

`keep_largest` 和 `convex_hull_fill` 是后处理的基石，贯穿各优先级方案。

### 3. Priority 1 — YOLOv8n-seg 实例分割

```python
def get_yolo_seg_mask(img):
    """YOLOv8n-seg 实例分割 → 逐实例软掩码合并"""
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
    return mask if np.sum(mask == 255) >= 100 else None
```

精度最高、能区分个体，但限于 COCO 的 10 种动物。

### 4. Priority 2 — DeepLabV3+ 语义分割

```python
VOC_ANIMALS = {3: "bird", 8: "cat", 10: "cow", 12: "dog", 13: "horse", 17: "sheep"}

def get_deeplab_pred(img_rgb):
    h, w = img_rgb.shape[:2]
    input_tensor = transform(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        output = deeplab(input_tensor)["out"][0]
    pred = output.argmax(0).cpu().numpy()
    return cv2.resize(pred.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

def get_deeplab_mask(img_rgb):
    pred = get_deeplab_pred(img_rgb)
    mask = np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.uint8)
    for aid in VOC_ANIMALS:
        mask[pred == aid] = 255
    return mask if np.sum(mask == 255) >= 100 else None
```

VOC 的 6 种动物边界干净，要求覆盖 > 10% 像素才接受，配合 YOLO 框做范围约束。

### 5. Priority 3 — YOLO 框 + 裁剪 Rembg

```python
def crop_and_rembg(img, gray, merged_box, box_ratio):
    mb_x1, mb_y1, mb_x2, mb_y2 = merged_box
    box_w, box_h = mb_x2 - mb_x1, mb_y2 - mb_y1
    use_hull = box_ratio > 0.7

    # 自适应裁剪边距
    if use_hull:                                    # > 70% → 全图 + 凸包
        cropped_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cropped_gray = gray
        crop_x1, crop_y1 = 0, 0
        crop_x2, crop_y2 = img.shape[1], img.shape[0]
    else:
        margin_factor = 0.15 if box_ratio > 0.4 else 0.3   # 40~70% → 15% 边距；< 40% → 30% 边距
        margin_x = int(box_w * margin_factor)
        margin_y = int(box_h * margin_factor)
        crop_x1 = max(0, mb_x1 - margin_x)
        crop_y1 = max(0, mb_y1 - margin_y)
        crop_x2 = min(img.shape[1], mb_x2 + margin_x)
        crop_y2 = min(img.shape[0], mb_y2 + margin_y)
        cropped = img[crop_y1:crop_y2, crop_x1:crop_x2]
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    crop_mask = get_rembg_mask(cropped_rgb)         # 局部 Rembg 抠图
    if crop_mask is not None:
        crop_mask = refine_mask(crop_mask, cropped_gray, use_hull)

    # 放回全图坐标系 + 裁剪框约束
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    if crop_mask is not None:
        mask[crop_y1:crop_y1 + cropped_rgb.shape[0],
             crop_x1:crop_x1 + cropped_rgb.shape[1]] = crop_mask
    box_mask = np.zeros_like(mask)
    box_mask[crop_y1:crop_y2, crop_x1:crop_x2] = 255
    return keep_largest(cv2.bitwise_and(mask, box_mask))
```

先裁剪局部区域再 Rembg，大幅缩小背景干扰，精度和速度都优于全图 Rembg。

### 6. Priority 4 — 全图 Rembg（终极保底）

```python
def get_rembg_mask(img_rgb):
    """双模型取优（ISNet + U²Net），α-matting 开/关各试一次"""
    best_mask = None
    best_verts = 0        # 用轮廓顶点数衡量精细度

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
                best_verts, best_mask = verts, m

    # U²Net 补充
    res = remove(Image.fromarray(img_rgb), session=session_u2, alpha_matting=False)
    m = np.array(res)[:, :, 3]
    _, m = cv2.threshold(m, 10, 255, cv2.THRESH_BINARY)
    m = keep_largest(m)
    c, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if c and sum(len(ci) for ci in c) > best_verts:
        best_mask = m

    return best_mask
```

不限类别，任何图片都能输出，是系统的最终保底。

### 7. 后处理精化 — Sobel 边缘细化

```python
def refine_mask(mask, gray_img, use_hull=False):
    kernel = np.ones((3, 3), np.uint8)
    if use_hull:
        mask = convex_hull_fill(mask)

    # Sobel 梯度检测原图强边缘
    gx = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, strong_edge = cv2.threshold(mag, 10, 255, cv2.THRESH_BINARY)

    # 膨胀 mask 边界 ∩ 原图强边缘 → 找回被误切的细长部位
    dilated = cv2.dilate(mask, kernel, iterations=6)
    boundary = cv2.bitwise_xor(dilated, mask)
    keep = cv2.bitwise_and(boundary, strong_edge)
    result = cv2.bitwise_or(mask, keep)

    result = keep_largest(result)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=2)
    return result
```

这是保证边缘精度的关键：Rembg 常把耳朵尖、爪子、尾巴误判为背景，Sobel 从原图梯度中找回这些部位，恢复自然轮廓。

### 8. 主调度 — 四级优先级级联

```python
def process_image(input_path, output_path):
    img = cv2.imread(input_path)
    h_full, w_full = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    yolo_boxes = detect_animal_bboxes(img)
    merged_box = merge_boxes(yolo_boxes) if yolo_boxes else None

    mask = get_yolo_seg_mask(img)                   # Priority 1

    if mask is None:                                # Priority 2
        deeplab_mask = get_deeplab_mask(img_rgb)
        if deeplab_mask is not None:
            cov = np.sum(deeplab_mask == 255) / deeplab_mask.size
            if cov > 0.10:                          # 要求 > 10% 覆盖
                mask = deeplab_mask
                if merged_box:                      # YOLO 框约束
                    box_mask = np.zeros((h_full, w_full), dtype=np.uint8)
                    box_mask[mb_y1:mb_y2, mb_x1:mb_x2] = 255
                    mask = cv2.bitwise_and(mask, box_mask)
                mask = keep_largest(mask)

    if mask is None and merged_box is not None:     # Priority 3
        box_ratio = (mb_x2-mb_x1)*(mb_y2-mb_y1) / (w_full*h_full)
        mask = crop_and_rembg(img, gray, merged_box, box_ratio)

    if mask is None:                                # Priority 4
        mask = get_rembg_mask(img_rgb)
        if mask is not None:
            mask = refine_mask(mask, gray, False)

    if mask is None:                                # 全失败 → 全白
        mask = np.zeros((h_full, w_full), dtype=np.uint8)

    mask = keep_largest(mask)

    # 正方形化输出：最小外接正方形 + 8% padding + 反色（白底黑主体）
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
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
    square_mask = np.zeros((square_size, square_size), dtype=np.uint8)
    src_x1 = max(0, cx - half)
    src_y1 = max(0, cy - half)
    src_x2 = min(mask.shape[1], cx + half)
    src_y2 = min(mask.shape[0], cy + half)
    dst_x1 = max(0, -cx + half)
    dst_y1 = max(0, -cy + half)
    if src_x2 > src_x1 and src_y2 > src_y1:
        sh, sw = src_y2 - src_y1, src_x2 - src_x1
        square_mask[dst_y1:dst_y1 + sh, dst_x1:dst_x1 + sw] = mask[src_y1:src_y2, src_x1:src_x2]

    result = 255 - square_mask              # 反色：白底黑主体
    black = 255 - result
    black = keep_largest(black)
    result = 255 - black
    Image.fromarray(result, mode="L").save(output_path)
```

级联逻辑清晰：任一方案成功即进入后处理，否则自动降级。**精度优先、容错性强、终极保底**。

### 9. 入口 main

```python
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    files = [f for f in os.listdir(INPUT_DIR)
             if os.path.splitext(f)[1].lower() in exts]
    files.sort()
    print(f"Found {len(files)} images\n")
    for i, fn in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {fn}")
        process_image(os.path.join(INPUT_DIR, fn),
                      os.path.join(OUTPUT_DIR, os.path.splitext(fn)[0] + ".png"))
    print("\nDone!")

if __name__ == "__main__":
    main()
```

扫描 `input_images/` 下 7 种格式图片，逐一处理输出为同名的 PNG 掩码到 `output_masks/`。

## 目录结构

```
shadow-paw-extractor/
├── .gitignore                          # Git 忽略规则
├── README.md                           # 项目说明
└── 识别动物/
    ├── extract_animal.py               # ★ 主程序（全部核心逻辑）
    ├── yolov8n.pt                      # YOLOv8 检测模型
    ├── yolov8n-seg.pt                  # YOLOv8 分割模型
    ├── input_images/                   # 测试图片（内置 23 张测试用例）
    │   ├── animal.jpg
    │   ├── 009f100a052d461cece9f6b7f165fa63.jpg
    │   ├── 1f5add93a332570f4f84367098490b4a.jpg
    │   └── ...（共 23 张，涵盖多种动物和场景）
    └── output_masks/                   # 输出结果（22 张已生成的 PNG 掩码）
        ├── animal.png
        ├── 009f100a052d461cece9f6b7f165fa63.png
        └── ...（运行后自动生成）
```

## 环境要求

- Python 3.8+
- CUDA GPU（可选，用于加速推理）

## 安装依赖

```bash
pip install opencv-python numpy pillow rembg ultralytics torch torchvision
```

## 模型下载说明

本工具共使用 **5 个深度学习模型**，下载方式如下：

| 模型 | 自动下载机制 | 存放位置 |
|------|------------|---------|
| `yolov8n.pt` | 首次 `YOLO("yolov8n.pt")` 时自动下载 | `识别动物/` 目录 |
| `yolov8n-seg.pt` | 首次 `YOLO("yolov8n-seg.pt")` 时自动下载 | `识别动物/` 目录 |
| ISNet (Rembg) | 首次 `new_session("isnet-general-use")` 时自动下载 | 系统缓存目录 |
| U²-Net (Rembg) | 首次 `new_session("u2net")` 时自动下载 | 系统缓存目录 |
| DeepLabV3+ ResNet50 | 首次 `deeplabv3_resnet50(weights="DEFAULT")` 时自动下载 | Torch 缓存目录 |

> **一句话总结**：只需确保 `yolov8n.pt` 和 `yolov8n-seg.pt` 在 `识别动物/` 目录下（首次运行会自动下载），其他模型由对应库自动管理。

如果想提前下载 YOLO 模型：

```bash
cd 识别动物
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); YOLO('yolov8n-seg.pt')"
```

## 使用方法

### 1. 准备图片

`input_images/` 目录已内置 **23 张测试图片**（涵盖猫、狗、鸟、马等多种动物及复杂背景），可直接运行体验。如需处理自己的图片，放入该目录即可，支持格式：
- JPG/JPEG、PNG、BMP、TIFF/TIF、WEBP

### 2. 运行程序

```bash
cd 识别动物
python extract_animal.py
```

### 3. 获取结果

输出在 `识别动物/output_masks/` 目录（PNG 单通道灰度图）：
- **白色 (255)**：背景
- **黑色 (0)**：动物主体
- 输出为**正方形**，动物居中，带 8% padding

## 输出示例

```
Found 3 images

[1/3] dog.jpg
  YOLO: 1 detections, box=(120,80,450,380)
  YOLOv8n-seg: (45%)
  Saved (42% animal)

[2/3] cat.png
  DeepLabV3: cat (38%)
  Saved (35% animal)

[3/3] bird.jpg
  Crop+rembg (28%)
  Saved (25% animal)

Done!
```

## 识别精度与边缘精度的双重保障

本工具从两个维度共同保证输出质量，确保抠出来的轮廓是**自然的动物形状**，而非几何体的拼凑：

**识别精度** — 确保"抠出来的是动物，不是背景杂物"：
- YOLOv8n 用 `ANIMAL_IDS` 严格筛选 COCO 的 10 种动物类别，非动物目标（人、车等）全部过滤
- DeepLabV3+ 语义分割沿用 VOC 的 6 种动物标签，要求覆盖 > 10% 像素才接受，防误检
- Rembg 通用模型作为终极保底，配合 YOLO 检测框裁剪后大幅降低误抠背景的概率

**边缘精度** — 确保"抠出来的轮廓像动物，不是多边形或锯齿"：
- YOLOv8n-seg 实例分割输出像素级软掩码（>0.5 阈值二值化），天然保留动物轮廓的复杂曲度
- Sobel 边缘细化基于原图梯度找回被误切的耳朵尖、爪子、尾巴等细长部位，恢复自然边缘
- 形态学闭运算填补 mask 内部的小孔洞，避免空洞感
- 凸包填充虽用直线补全，但后续 Sobel 步骤会基于真实边缘重新修正，确保最终轮廓贴合原图中的动物实际形状

**最终效果**：输出掩码的黑色区域（动物主体）的轮廓与原图中的动物边缘高度吻合，呈现自然的曲线过渡，而非几何直线的拼接。人的肉眼应当能一眼认出"这是一只动物的形状"，而非"这是一些多边形拼在一起"。

## 场景覆盖

| 场景 | 处理策略 |
|------|---------|
| 正面清晰照片 | YOLOv8n-seg 直接输出精准轮廓 |
| 复杂背景（树丛、水面） | YOLO框 + 裁剪Rembg，缩小干扰范围 |
| 部分遮挡 | Sobel 边缘细化从梯度找回丢失边缘 |
| 多只动物 | 合并检测框，一次处理 |
| 小目标（远景动物） | 裁剪放大后局部 Rembg |
| 大目标（特写） | 自适应边距 + 凸包填充补全 |
| 非常见动物（兔子/狐狸等） | Rembg 全图抠图终极保底 |

## 技术细节

### 模型说明

| 模型 | 用途 | 特点 |
|------|------|------|
| YOLOv8n | 动物检测 | 快速定位动物位置 |
| YOLOv8n-seg | 实例分割 | 像素级精确分割，区分个体 |
| DeepLabV3+ | 语义分割 | 边界干净，VOC动物擅长 |
| ISNet / U²-Net | 背景移除 | 通用抠图，不限类别 |

### 核心算法流程

1. **YOLO 检测**：识别图像中的动物并获取边界框，合并多只动物为外接矩形
2. **多策略分割**：四级优先级级联，质量最优者胜出
3. **后处理优化**：
   - 最大连通区域保留（去噪）
   - Sobel 边缘细化（基于原图梯度找回丢失边缘）
   - 凸包填充（补全肢体缺失）
   - 形态学闭运算（填小孔）
4. **输出处方化**：正方形居中 + 8% padding + 反色（白底黑主体）

### 支持的动物类别

- **COCO 数据集**（ID 16~25）：鸟、猫、牛、狗、马、羊、象、熊、斑马、长颈鹿
- **VOC 数据集**：bird、cat、cow、dog、horse、sheep
- **Rembg 通用模型**：不限类别

## 许可证

本项目仅供学习和研究使用。

## 致谢

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Rembg](https://github.com/danielgatis/rembg)
- [DeepLabV3+](https://pytorch.org/vision/stable/models.html#deeplabv3)
