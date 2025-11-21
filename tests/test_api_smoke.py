import pytest
import io
from app import app # 导入刚才写的 Flask app

# 这是 Pytest 的 Fixture，相当于每次测试前先造一个虚拟的客户端
@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_check(client):
    """
    测试 1: 随便访问一个不存在的页面，看看是不是返回 404
    目的: 验证服务基础功能正常
    """
    response = client.get('/')
    # 因为我们没写首页，所以应该 404，但不能崩 (500)
    assert response.status_code == 404

def test_predict_no_file(client):
    """
    测试 2: 异常场景测试 - 不传图片
    预期: 返回 400 错误
    """
    response = client.post('/predict')
    assert response.status_code == 400
    assert b'No file part' in response.data

def test_predict_flow(client):
    """
    测试 3: 正常流程测试 - 上传一张假图片
    预期: 返回 200 成功，且 JSON 里包含 detections 字段
    """
    # 1. 造一张假图片 (纯黑色)
    # 这里我们需要用 PIL 在内存里画一张图，模拟文件上传
    from PIL import Image
    
    # 创建一个字节流缓冲区
    img_byte_arr = io.BytesIO()
    # 创建一张 100x100 的红图
    image = Image.new('RGB', (100, 100), color='red')
    # 保存到缓冲区，格式为 JPEG
    image.save(img_byte_arr, format='JPEG')
    # 把指针重置到开头
    img_byte_arr.seek(0)

    # 2. 发送 POST 请求
    data = {'file': (img_byte_arr, 'test.jpg')}
    response = client.post('/predict', data=data, content_type='multipart/form-data')

    # 3. 断言 (验证结果)
    assert response.status_code == 200
    json_data = response.get_json()
    
    assert json_data['message'] == 'Success'
    assert 'detections' in json_data
    # 即使是空列表也算对 (因为纯色图可能识别不出东西)，只要格式对就行
    assert isinstance(json_data['detections'], list)