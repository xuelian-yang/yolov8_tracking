# WARNING: CUDA Minor Version Compatibility mode ENABLED.
#   Using driver version 510.108.03 which has support for CUDA 11.6.  This container
#   was built with CUDA 11.8 and will be run in Minor Version Compatibility mode.
#   CUDA Forward Compatibility is preferred over Minor Version Compatibility for use
#   with this container but was unavailable:
#   [[Forward compatibility was attempted on non supported HW (CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE) cuInit()=804]]
#   See https://docs.nvidia.com/deploy/cuda-compatibility/ for details.

# NOTE: The SHMEM allocation limit is set to the default of 64MB.  This may be
#    insufficient for PyTorch.  NVIDIA recommends the use of the following flags:
#    docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 ...

FROM cuda:11.6.0-cudnn8-runtime-ubuntu20.04-python3.8

# Install pip packages
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip setuptools wheel  numpy -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
# RUN pip install --no-cache torch>=1.7.0  torchvision>=0.8.1 --index-url  https://mirrors.aliyun.com/pytorch-wheels/cu116 -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

#  阿里云的源下载torch太慢，或者版本不全, 无法下载想要的cuda版本
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu116
RUN pip install --no-cache -r requirements.txt -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

# Create working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# Clone with submodules
# RUN git clone --recurse-submodules https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet.git /usr/src/app



CMD jupyter lab  --notebook-dir=/usr/src/app --ip=0.0.0.0 --no-browser --allow-root --port=8888 --NotebookApp.token='' 
