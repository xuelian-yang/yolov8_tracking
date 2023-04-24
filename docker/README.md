# yolov8 tracking  docker环境


## 1. 搭建过程
-  step 1
从NGC下载`nvcr.io/nvidia/cuda:11.6.0-cudnn8-runtime-ubuntu20.04`镜像

- step2
  基于step 1中的镜像，利用[cuda-base-python3.Dockerfile](./cuda-base-python3.Dockerfile)build 包含一些系统级工具和基本python3环境，后续将此作为基础镜像，每次更新就不用从源头开始

- step3
  基于step 2中的镜像，利用[yolov8_tracking.Dockerfile](./yolov8_tracking.Dockerfile) build真正的应用

## 2. 使用
- 目前通过jupyter去使用该环境，只需在网页访问`http://10.10.132.11:8888/lab`即可
- WORKSPACE 为`/usr/src/app`，该目录下的数据均会被持久化到磁盘
- `mount`下的目录均为宿主机目录`/mnt/d16archive/data/alaco_video_archive/multi-stream-reocrds`，主要用来存放截取的视频数据
