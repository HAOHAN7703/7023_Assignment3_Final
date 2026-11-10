
from pathlib import Path
import jetson.inference
import jetson.utils

# 1. 创建结果文件夹
Path("results").mkdir(exist_ok=True)

# 2. 两张测试图片
images = ["image1.jpg", "image2.jpg"]

# 3. 两个置信度阈值
thresholds = [0.3, 0.7]

# 4. 分别在两个阈值下运行模型
for threshold in thresholds:

    net = jetson.inference.detectNet(
        "ssd-mobilenet-v2",
        threshold=threshold
    )

    for filename in images:

        # 读取图片
        img = jetson.utils.loadImage(filename)

        # 进行目标检测（同时绘制检测框）
        detections = net.Detect(img)

        # 输出检测结果
        print(f"\nImage: {filename}")
        print(f"Threshold: {threshold}")
        print(f"Detected objects: {len(detections)}")

        for i, detection in enumerate(detections, 1):
            print(f"\nObject {i}")
            print(detection)

        # 保存检测后的图片
        output = f"results/{Path(filename).stem}_t{threshold}.jpg"
        jetson.utils.cudaDeviceSynchronize()
        jetson.utils.saveImage(output, img)

        print(f"Saved: {output}")
