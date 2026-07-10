# Shadow Paw Extractor - 动物抠图工具

这是一个用于动物图像自动抠图的 Python 工具，结合多种深度学习模型实现高精度的动物分割和背景移除。

## 功能特性

- **广泛适用性**：本代码适用于大部分动物图像场景，无论是清晰照片、复杂背景还是部分遮挡情况，都能提供可靠的抠图效果
- **高效处理**：每张图片的识别和处理时间不到 5 秒，快速响应用户需求
- **多模型融合**：集成 YOLOv8、DeepLabV3+ 和 Rembg 三种分割方案
- **智能优先级**：自动选择最优分割结果
  1. YOLOv8n-seg（实例分割，最精确）
  2. DeepLabV3+（VOC 数据集动物类别）
  3. YOLO 检测框 + Rembg 局部抠图
  4. 全图 Rembg 背景移除
- **支持的动物类别**：
  - COCO 数据集：鸟、猫、牛、狗、马、羊等 10 种动物
  - VOC 数据集：bird、cat、cow、dog、horse、sheep
- **后处理优化**：
  - 最大连通区域保留
  - 凸包填充
  - 边缘细化（基于 Sobel 梯度）
  - 形态学操作

## 目录结构

```
shadow-paw-extractor/
├── README.md                 # 项目说明文档
└── 识别动物/
    ├── extract_animal.py     # 主程序
    ├── yolov8n.pt            # YOLOv8 检测模型
    ├── yolov8n-seg.pt        # YOLOv8 分割模型
    ├── input_images/         # 输入图片目录
    └── output_masks/         # 输出掩码目录
```

## 环境要求

- Python 3.8+
- CUDA GPU（可选，用于加速推理）

## 安装依赖

```bash
pip install opencv-python numpy pillow rembg ultralytics torch torchvision
```

## 使用方法

### 1. 准备图片

将需要处理的图片放入 `识别动物/input_images/` 目录，支持的格式：
- JPG/JPEG
- PNG
- BMP
- TIFF/TIF
- WEBP

### 2. 运行程序

```bash
cd 识别动物
python extract_animal.py
```

### 3. 获取结果

处理后的掩码图片将保存在 `识别动物/output_masks/` 目录：
- 输出格式：PNG（单通道灰度图）
- 白色区域 (255)：背景
- 黑色区域 (0)：动物主体

## 输出示例

程序运行时会显示每张图的处理信息：
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

## 技术细节

### 模型说明

| 模型 | 用途 | 特点 |
|------|------|------|
| YOLOv8n | 动物检测 | 快速定位动物位置 |
| YOLOv8n-seg | 实例分割 | 像素级精确分割 |
| DeepLabV3+ | 语义分割 | 擅长 VOC 动物类别 |
| ISNet-General-Use | 背景移除 | 通用抠图模型 |
| U^2-Net | 背景移除 | 补充抠图方案 |

### 核心算法流程

1. **YOLO 检测**：识别图像中的动物并获取边界框
2. **多策略分割**：按优先级尝试不同分割方法
3. **掩码优化**：
   - 保留最大连通区域
   - 边缘细化（Sobel 梯度约束）
   - 凸包填充（针对大面积检测框）
4. **输出处方化**：
   - 提取动物区域的最小外接正方形
   - 添加 8%  padding
   - 反转为白底黑主体

## 注意事项

- 首次运行时会自动下载预训练模型
- 建议使用 GPU 以获得更快的处理速度
- 输出掩码为正方形，动物居中显示
- 如未检测到动物，将输出全白图片

## 许可证

本项目仅供学习和研究使用。

## 致谢

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Rembg](https://github.com/danielgatis/rembg)
- [DeepLabV3+](https://pytorch.org/vision/stable/models.html#deeplabv3)
