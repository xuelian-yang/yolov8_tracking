#!/bin/bash

conda activate dev

# Windows 下使用 ^ 连接多行，Linux 下使用 \

# single RTSP
python track.py --source rtsp://admin:hik12345=@10.10.145.231:554/Streaming/Channels/101

# multi RTSPs
python track.py --source configs/rtsp.txt


# single mp4 file
python track.py --source D:/alaco_video_archive/multi-stream-reocrds/W91_2023-04-18_09_45_32.mp4

# multi mp4 files
python track.py --source configs/mp4.txt

# dailin
python track.py --source mp4.txt

# <!-- End of File -->
