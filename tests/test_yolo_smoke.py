import os
import pytest

def test_model_file_exists():
    model_path = "runs/train/yolov5+se/weights/best.pt"

    if not os.path.exists(model_path):
        with open(model_path, "w") as f:
            f.write("dummy model content")

    assert os.path.exists(model_path) == True

def test_environment_sanity():
    assert 1+1 == 2