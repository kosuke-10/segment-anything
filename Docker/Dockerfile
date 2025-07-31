FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo

# 基本ツール・Python・PyTorch
RUN apt-get update && \
    apt-get install -y tzdata git wget curl unzip nano build-essential \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y python3 python3-pip python3-venv nodejs npm && rm -rf /var/lib/apt/lists/*
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install torch==2.2.2+cu118 torchvision==0.17.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
RUN pip install opencv-python pycocotools matplotlib onnxruntime onnx jupyter

RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs

RUN npm install -g yarn

# Webデモ用の依存関係インストール
WORKDIR /segment-anything/demo
COPY demo/package.json ./
RUN yarn install

WORKDIR /segment-anything
COPY . /segment-anything
RUN pip install -e .

RUN pip install pandas scikit-learn scikit-image tqdm

# ポート公開
EXPOSE 8888 8081

CMD ["/bin/bash"]