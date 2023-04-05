#!/bin/bash

conda activate dev

# Windows 下使用 ^ 连接多行，Linux 下使用 \

python track.py ^
  --source rtsp://admin:hik12345=@10.10.145.231:554/Streaming/Channels/101 ^
  --show-vid

python track.py ^
  --source D:/Video_via_PotPlayer/20230403/20230403_134613.mp4 ^
  --show-vid


# <!-- End of File -->
