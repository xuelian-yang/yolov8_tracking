FROM nvcr.io/nvidia/cuda:11.6.0-cudnn8-runtime-ubuntu20.04

ARG CUDA_VERSION=11.6.0


ARG PYTHON_VERSION=3.8

# Needed for string substitution
SHELL ["/bin/bash", "-c"]

# https://techoverflow.net/2019/05/18/how-to-fix-configuring-tzdata-interactive-input-when-building-docker-images/
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=CST-8

ENV LD_LIBRARY_PATH /usr/local/cuda/compat:$LD_LIBRARY_PATH


RUN sed -i "s/archive.ubuntu.com/mirrors.aliyun.com/g" /etc/apt/sources.list
RUN apt-get update && apt-get install -yq --no-install-recommends \
    software-properties-common  && \
    apt-get install -yq --no-install-recommends \
    python${PYTHON_VERSION} \
    python3-pip \
    python3-dev \
    git \
    gcc \
    g++ \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx && \
    apt-get clean &&   \
    rm -rf /var/lib/apt/lists/*




ENV PYTHONPATH="/usr/lib/python${PYTHONVERSION}/site−packages:/usr/local/lib/python${PYTHON_VERSION}/site-packages"

RUN CUDA_PATH=(/usr/local/cuda-*) && \
    CUDA=`basename $CUDA_PATH` && \
    echo "$CUDA_PATH/compat" >> /etc/ld.so.conf.d/${CUDA/./-}.conf && \
    ldconfig



# Install gstreamer
# RUN apt-get install --no-install-recommends -y \
#     libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev \
#     gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
#     gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-doc gstreamer1.0-tools \
#     gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 \
#     gstreamer1.0-qt5 gstreamer1.0-pulseaudio




