# 1. 基础镜像
FROM python:3.9-slim

# 2. 设置工作目录
WORKDIR /app

# 3. 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. 复制依赖文件并安装
COPY requirements-ci.txt .

# 强制安装 CPU 版 Torch + 其他依赖
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==1.13.1+cpu torchvision==0.14.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements-ci.txt

# 5. 复制项目所有代码
COPY . .

# 6. 设置默认运行命令
CMD ["python", "detect.py", "--source", "data/images/bus.jpg", "--device", "cpu"]