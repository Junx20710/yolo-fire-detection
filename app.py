import io
import torch
from flask import Flask, request, jsonify
from PIL import Image
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from pathlib import Path

app = Flask(__name__)

# --- 1. 全局加载模型 (避免每次请求都重新加载，太慢) ---
# 这里我们用之前测试过的逻辑，自动下载 yolov5s.pt
WEIGHTS = Path('yolov5s.pt')
if not WEIGHTS.exists():
    torch.hub.download_url_to_file(
        'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt',
        str(WEIGHTS)
    )

print("正在加载模型...")
device = torch.device('cpu') # 强制用 CPU，保证兼容性
model = DetectMultiBackend(WEIGHTS, device=device)
print("✅ 模型加载完毕！")

# --- 2. 定义接口 (Endpoint) ---
@app.route('/predict', methods=['POST'])
def predict():
    """
    接口功能：接收图片文件，返回检测结果
    请求方式：POST
    参数：file (图片文件)
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # A. 读取图片
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        
        # B. 预处理 (Resize to 640x640)
        img_resized = img.resize((640, 640))
        
        # 转为 Tensor: [C, H, W], 归一化
        import numpy as np
        img_np = np.array(img_resized)
        # 如果是 RGBA (4通道)，转 RGB
        if img_np.shape[-1] == 4:
            img_np = img_np[..., :3]
            
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device) # Add batch dim

        # C. 推理
        pred = model(img_tensor)
        pred = non_max_suppression(pred, 0.25, 0.45) # NMS 后处理

        # D. 格式化结果
        results = []
        for det in pred:
            if len(det):
                for *xyxy, conf, cls in det:
                    results.append({
                        'class': int(cls),
                        'confidence': float(conf),
                        'box': [float(x) for x in xyxy]
                    })

        return jsonify({'message': 'Success', 'detections': results}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- 3. 启动服务 ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)