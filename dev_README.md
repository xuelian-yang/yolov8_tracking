
# github to do.geely.com

```bash
D:
mkdir temp
cd temp

git clone --recursive git@github.com:ultralytics/ultralytics.git
cd ultralytics
git config user.name "xuelian"
git config user.email "xuelian@"
git remote add geely ssh://git@coding.geely.com:3721/0355879/ultralytics.git
git push -u geely --all
git push -u geely --tags

cd ..
git clone --recursive git@github.com:mikel-brostrom/yolov8_tracking.git
cd yolov8_tracking
git config user.name "xuelian"
git config user.email "xuelian@"
git remote add geely ssh://git@coding.geely.com:3721/0355879/yolov8_tracking.git
git push -u geely --all
git push -u geely --tags
git branch -b dev

# 参考 https://blog.csdn.net/qq_25100723/article/details/116193146 修改 submodule
git submodule
# 修改 .gitmodules
#   https://github.com/ultralytics/ultralytics -> https://do.geely.com/codingRoot/0355879/ultralytics/
git submodule sync
git submodule init
git submodule update
```
