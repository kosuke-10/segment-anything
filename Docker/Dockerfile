FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo

RUN apt-get update && \
    apt-get install -y tzdata git wget curl unzip nano build-essential \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y python3 python3-pip python3-venv && rm -rf /var/lib/apt/lists/*
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN python -m pip install --upgrade pip setuptools wheel

RUN pip install torch==2.2.2+cu118 torchvision==0.17.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

RUN pip install opencv-python pycocotools matplotlib onnxruntime onnx jupyter

EXPOSE 8888

CMD ["/bin/bash"]