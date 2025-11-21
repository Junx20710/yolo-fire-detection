import sys
from pathlib import Path
import pytest
import os
import torch
import numpy as np
from models.common import DetectMultiBackend

FILE = Path(__file__).resolve()

ROOT = FILE.parents[1]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

def test_dependency_imports():

    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
    except ImportError:
        pytest.fail("PyTortch is uninstalled!!")
    try:
        import cv2
        print(f"OpenCV Version: {cv2.__version__}")
    except ImportError:
        pytest.fail("Open CV is uninstalled!!")
    if torch.cuda.is_available():
        print(f"GPU Available: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU detected, using CPU Mode.")

    
def test_model_forward_pass():

    #A.准备权重
    weights = ROOT/ 'yolov5s.pt'
    
    if not os.path.exists(weights):
        torch.hub.download_url_to_file(
            'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt',
            str(weights)
        )

    #B.加载模型
    device = torch.device('cpu')

    try:
        model = DetectMultiBackend(weights,device = device)
    except Exception as e:
        pytest.fail()

    #C.Mocking
    fake_input = torch.zeros((1,3,640,640),device=device)

    #D.推理

    try:
        #Warmup
        model.warmup(imgsz=(1,3,640,640))

        #推理
        pred  = model(fake_input)

        #Assert
        assert pred is not None
    
    except Exception as e:
        pytest.fail()
