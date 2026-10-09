如何使用深度学习模型训练使用 天宫遥感图像自然场景智能识别数据集 进行识别分类？![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c3b773163dc747c69bb040edc9cc282e.png)天宫遥感图像自然场景智能识别数据集
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a35a54a743cf4a20a2c87662c3765e73.png)

背景描述
“天宫二号”空间实验室于2016年9月成功发射，2019年7月结束数据服务，在轨期间运行稳定、各载荷状态良好。其中，天宫二号搭载的宽波段成像仪获取了一大批高质量的数据产品，在国土资源、海洋及海岸带、内陆水体、生态环境、农业、林业等各个领域具有重要的应用潜力。 本赛题旨在基于“天宫二号”宽波段成像仪数据的大尺度进行典型场景智能识别，场景类型包括海滩、圆形农田、云、荒漠、林地、山脉、矩形农田、建筑区、河流和雪山共10类，共20000张图像，其中2000张为训练集，18000张为测试集。要求参赛选手利用机器学习等智能算法自动识别出所规定的测试集图像对应的场景类型。![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/27e944bd377e4a83bf2f40a4a198310f.png)


数据说明
数据集名称：NaSC-TG2 （天宫二号遥感图像自然场景分类数据集）
图像格式： RGB三波段真彩色图像(jpg)![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8edec9c6c9a0452180f39032ef470cda.png)

场景类别：海滩、圆形农田、云、荒漠、林地、山脉、矩形农田、建筑区、河流和雪山
图像尺寸：128×128![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/98749d5fafea401e94b7074b45fa6ec9.png)

图像数量：每种类别有2000张，共计20000张
train：包含2000张，用于算法训练![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d4514287035e4b1cbcf28c5c6f54c5e0.png)

test：包含18000张，作为独立样本进行算法精度验证，不可参与训练![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0a4529d29a894062a575df25414c632d.png)
/![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/28a47e2d162c4a0bb092629ef42cc389.png)
以下仅供参考

使用YOLOv8来进行天宫遥感图像自然场景智能识别任务。由于这是一个分类任务而不是检测任务，我们将使用YOLOv8的分类版本 `yolov8n-cls.pt`。

以下是详细的步骤：

1. **环境准备**：
   - 安装必要的库。
   - 下载并组织数据集。

2. **数据预处理**：
   - 创建一个包含类别信息的 CSV 文件作为标签。
   - 将图像路径和类别标签映射到 CSV 文件中。

3. **模型定义与训练**：
   - 使用YOLOv8进行图像分类任务。

4. **评估与可视化**：
   - 评估模型性能。
   - 可视化结果。

### 环境准备

首先，我们需要安装必要的库。您可以使用以下命令来设置环境：

```bash
pip install ultralytics pandas opencv-python-headless scikit-learn
```

### 数据组织

假设您的数据集已经按照类别进行了组织，并且每个类别的图像存储在单独的文件夹中。以下是文件夹结构示例：

```
datasets/nasc_tg2/
├── train/
│   ├── beach/
│   │   ├── image0.jpg
│   │   ├── image1.jpg
│   │   └── ...
│   ├── circular_farmland/
│   │   ├── image0.jpg
│   │   ├── image1.jpg
│   │   └── ...
│   ├── cloud/
│   │   ├── image0.jpg
│   │   ├── image1.jpg
│   │   └── ...
│   ├── desert/
│   │   ├── image0.jpg
│   │   ├── image1.jpg
│   │   └── ...
│   ├── forest/
│   │   ├── image0.jpg
│   │   ├── image1.jpg
│   │   └── ...
│   ├── mountain/
│   │   ├── image0.jpg
│   │   ├── image1.jpg
│   │   └── ...
│   ├── rectangular_farmland/
│   │   ├── image0.jpg
│   │   ├── image1.jpg
│   │   └── ...
│   ├── residential_area/
│   │   ├── image0.jpg
│   │   ├── image1.jpg
│   │   └── ...
│   ├── river/
│   │   ├── image0.jpg
│   │   ├── image1.jpg
│   │   └── ...
│   └── snow_mountain/
│       ├── image0.jpg
│       ├── image1.jpg
│       └── ...
└── test/
    ├── beach/
    │   ├── image0.jpg
    │   ├── image1.jpg
    │   └── ...
    ├── circular_farmland/
    │   ├── image0.jpg
    │   ├── image1.jpg
    │   └── ...
    ├── cloud/
    │   ├── image0.jpg
    │   ├── image1.jpg
    │   └── ...
    ├── desert/
    │   ├── image0.jpg
    │   ├── image1.jpg
    │   └── ...
    ├── forest/
    │   ├── image0.jpg
    │   ├── image1.jpg
    │   └── ...
    ├── mountain/
    │   ├── image0.jpg
    │   ├── image1.jpg
    │   └── ...
    ├── rectangular_farmland/
    │   ├── image0.jpg
    │   ├── image1.jpg
    │   └── ...
    ├── residential_area/
    │   ├── image0.jpg
    │   ├── image1.jpg
    │   └── ...
    ├── river/
    │   ├── image0.jpg
    │   ├── image1.jpg
    │   └── ...
    └── snow_mountain/
        ├── image0.jpg
        ├── image1.jpg
        └── ...
```

### 创建标签文件

我们需要创建一个包含图像路径和类别标签的CSV文件。以下是生成标签文件的脚本 `create_labels_csv.py`:

```python
[<title="Create Labels CSV for NaSC-TG2 Dataset">]
import os
import pandas as pd
from pathlib import Path

# Define paths
base_path = Path('datasets/nasc_tg2')
output_train_csv = base_path / 'train.csv'
output_test_csv = base_path / 'test.csv'

# Class names and their corresponding IDs
class_names = {
    'beach': 0,
    'circular_farmland': 1,
    'cloud': 2,
    'desert': 3,
    'forest': 4,
    'mountain': 5,
    'rectangular_farmland': 6,
    'residential_area': 7,
    'river': 8,
    'snow_mountain': 9
}

# Initialize lists to store file paths and labels
train_file_paths = []
train_labels = []
test_file_paths = []
test_labels = []

# Iterate over each class directory in train set
for class_name, class_id in class_names.items():
    class_dir = base_path / 'train' / class_name
    for image_file in class_dir.glob('*.jpg'):
        train_file_paths.append(str(image_file))
        train_labels.append(class_id)

# Iterate over each class directory in test set
for class_name, class_id in class_names.items():
    class_dir = base_path / 'test' / class_name
    for image_file in class_dir.glob('*.jpg'):
        test_file_paths.append(str(image_file))
        test_labels.append(class_id)

# Create DataFrames
train_df = pd.DataFrame({
    'file_path': train_file_paths,
    'label': train_labels
})

test_df = pd.DataFrame({
    'file_path': test_file_paths,
    'label': test_labels
})

# Save to CSV
train_df.to_csv(output_train_csv, index=False)
test_df.to_csv(output_test_csv, index=False)

print(f"Train labels CSV saved to {output_train_csv}")
print(f"Test labels CSV saved to {output_test_csv}")
```

### 创建YAML配置文件

确保您有一个正确的YAML配置文件 `nasc_tg2.yaml` 来描述数据集。以下是示例配置文件：

```yaml
[<title="YOLOv8 Configuration File for NaSC-TG2 Image Classification">]
train: ../datasets/nasc_tg2/train.csv
val: ../datasets/nasc_tg2/test.csv  # Using the test set for validation as no separate validation set is provided

nc: 10
names: ['beach', 'circular_farmland', 'cloud', 'desert', 'forest', 'mountain', 'rectangular_farmland', 'residential_area', 'river', 'snow_mountain']
```

将上述内容保存为 `datasets/nasc_tg2/nasc_tg2.yaml`。

### 模型定义与训练

我们将使用YOLOv8进行分类任务。以下是训练脚本 `train_classification.py`:

```python
[<title="Training Script for NaSC-TG2 Image Classification using YOLOv8">]
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-cls.pt')  # load a pretrained classification model (recommended for training)

# Train the model
results = model.train(
    data='../datasets/nasc_tg2/nasc_tg2.yaml',
    epochs=50,
    imgsz=128,
    batch=64,
    project='../runs/train',
    name='nasc_tg2_classification'
)

# Evaluate the model
metrics = model.val()
results = model.export(format='onnx')  # export the trained model to ONNX format
```

### 评估与可视化

使用YOLOv8自带的评估脚本来评估分类模型。

```python
[<title="Evaluation Script for NaSC-TG2 Image Classification using YOLOv8">]
from ultralytics import YOLO

# Load the best model
best_model = YOLO('../runs/train/nasc_tg2_classification/weights/best.pt')

# Evaluate the model on the validation dataset
metrics = best_model.val(data='../datasets/nasc_tg2/nasc_tg2.yaml', conf=0.5)
print(metrics)
```

### 用户界面

我们将使用 PyQt5 创建一个简单的 GUI 来加载和运行模型进行实时预测。以下是用户界面脚本 `ui.py`:

```python
[<title="PyQt5 Main Window for NaSC-TG2 Image Classification">]
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from ultralytics import YOLO

# Load model
classification_model = YOLO('../runs/train/nasc_tg2_classification/weights/best.pt')

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("天宫二号遥感图像自然场景分类系统")
        self.setGeometry(100, 100, 800, 600)

        self.initUI()

    def initUI(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)

        self.load_image_button = QPushButton("加载图像", self)
        self.load_image_button.clicked.connect(self.load_image)
        self.layout.addWidget(self.load_image_button)

        self.predict_button = QPushButton("开始预测", self)
        self.predict_button.clicked.connect(self.start_prediction)
        self.layout.addWidget(self.predict_button)

        self.central_widget.setLayout(self.layout)

        self.image_path = None

    def load_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "选择图像文件", "", "Images (*.png *.jpg *.jpeg);;All Files (*)", options=options)
        if file_name:
            self.image_path = file_name
            self.display_image(file_name)

    def display_image(self, path):
        pixmap = QPixmap(path)
        scaled_pixmap = pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
        self.image_label.setPixmap(scaled_pixmap)

    def start_prediction(self):
        if self.image_path is not None:
            original_image = cv2.imread(self.image_path)
            image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

            # Prediction
            results = classification_model.predict(image_rgb, size=128)[0]

            # Map class ID to name
            class_names = ['海滩', '圆形农田', '云', '荒漠', '林地', '山脉', '矩形农田', '建筑区', '河流', '雪山']
            class_id = int(results.probs.argmax())
            class_name = class_names[class_id]
            confidence = float(results.probs.max())

            # Display result
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f'{class_name} ({confidence:.2f})'
            cv2.putText(image_rgb, text, (10, 50), font, 1, (0, 255, 0), 2)

            h, w, ch = image_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
            self.image_label.setPixmap(scaled_pixmap)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
```

请确保将路径替换为您实际的路径。

### 使用说明

1. **配置路径**：
   - 确保 `datasets/nasc_tg2` 目录结构正确，并且包含 `train.csv`, `test.csv` 文件。
   - 确保 `runs/train/nasc_tg2_classification/weights/best.pt` 是训练好的 YOLOv8 分类模型权重路径。

2. **运行脚本**：
   - 在终端中运行 `create_labels_csv.py` 脚本来生成标签CSV文件。
   - 在终端中运行 `check_dataset.py` 脚本来检查数据集的有效性。
   - 在终端中运行 `train_classification.py` 脚本来训练分类模型。
   - 在终端中运行 `evaluate_classification.py` 来评估分类模型性能。
   - 在终端中运行 `ui.py` 来启动 GUI 应用程序。

3. **注意事项**：
   - 确保所有必要的工具箱已安装，特别是 `ultralytics` 和 `PyQt5`。
   - 根据需要调整参数，如 `epochs` 和 `batch_size`。

### 示例

假设您的数据文件夹结构如下：
```
datasets/
└── nasc_tg2/
    ├── train/
    │   ├── beach/
    │   │   ├── image0.jpg
    │   │   ├── image1.jpg
    │   │   └── ...
    │   ├── circular_farmland/
    │   │   ├── image0.jpg
    │   │   ├── image1.jpg
    │   │   └── ...
    │   ├── cloud/
    │   │   ├── image0.jpg
    │   │   ├── image1.jpg
    │   │   └── ...
    │   ├── desert/
    │   │   ├── image0.jpg
    │   │   ├── image1.jpg
    │   │   └── ...
    │   ├── forest/
    │   │   ├── image0.jpg
    │   │   ├── image1.jpg
    │   │   └── ...
    │   ├── mountain/
    │   │   ├── image0.jpg
    │   │   ├── image1.jpg
    │   │   └── ...
    │   ├── rectangular_farmland/
    │   │   ├── image0.jpg
    │   │   ├── image1.jpg
    │   │   └── ...
    │   ├── residential_area/
    │   │   ├── image0.jpg
    │   │   ├── image1.jpg
    │   │   └── ...
    │   ├── river/
    │   │   ├── image0.jpg
    │   │   ├── image1.jpg
    │   │   └── ...
    │   └── snow_mountain/
    │       ├── image0.jpg
    │       ├── image1.jpg
    │       └── ...
    ├── test/
    │   ├── beach/
    │   │   ├── image0.jpg
    │   │   ├── image1.jpg
    │   │   └── ...
    │   ├── circular_farmland/
    │   │   ├── image0.jpg
    │   │   ├── image1.jpg
    │   │   └── ...
    │   ├── cloud/
    │   │   ├── image0.jpg
    │   │   ├── image1.jpg
    │   │   └── ...
    │   ├── desert/
    │   │   ├── image0.jpg
    │   │   ├── image1.jpg
    │   │   └── ...
    │   ├── forest/
    │   │   ├── image0.jpg
    │   │   ├── image1.jpg
    │   │   └── ...
    │   ├── mountain/
    │   │   ├── image0.jpg
    │   │   ├── image1.jpg
    │   │   └── ...
    │   ├── rectangular_farmland/
    │   │   ├── image0.jpg
    │   │   ├── image1.jpg
    │   │   └── ...
    │   ├── residential_area/
    │   │   ├── image0.jpg
    │   │   ├── image1.jpg
    │   │   └── ...
    │   ├── river/
    │   │   ├── image0.jpg
    │   │   ├── image1.jpg
    │   │   └── ...
    │   └── snow_mountain/
    │       ├── image0.jpg
    │       ├── image1.jpg
    │       └── ...
    ├── train.csv
    └── test.csv
```

并且每个数据集中包含相应的图像和标签文件。运行 `ui.py` 后，您可以点击按钮来加载图像并进行天宫二号遥感图像自然场景分类。

### 总结

通过上述步骤，构建一个全面的基于深度学习的天宫二号遥感图像自然场景分类系统，包括数据集准备、环境部署、数据预处理、模型定义、训练、评估、结果分析、可视化以及用户界面开发。以下是所有相关的代码文件：

1. **创建标签文件** (`create_labels_csv.py`)
2. **数据检查** (`check_dataset.py`)
3. **分类训练脚本** (`train_classification.py`)
4. **YOLOv8 配置文件** (`nasc_tg2.yaml`)
5. **分类评估脚本** (`evaluate_classification.py`)
6. **用户界面** (`ui.py`)


