# -*- coding: utf-8 -*-
import cv2
from distinctipy import distinctipy
import json
import math
import numpy as np
import os
import os.path as osp
from termcolor import colored
import time

"""
要求:
  1. 代码可读性
     1.1 必要的中文注释
     1.2 变量意义一目了然
  2. numpy 格式的注释规范 见 https://numpydoc.readthedocs.io/en/v1.1.0/format.html
  3. 统计算法运行所需时间
  4. 代码上传到 鲁班
"""

"""
# BFS求解(复杂度太高)
def boundary_fill(x, y, filled_color, boundary_color):
    global mask
    width = mask.shape[1]
    height = mask.shape[0]
    list = [[x, y]]
    while list:
        point = list.pop()
        if mask[point[0]][point[1]] != boundary_color and mask[point[0]][point[1]] != filled_color and 0 <= point[0] < width and 0 <= point[1] < height:
            mask[point[1]][point[0]] = filled_color
            list.append((point[0]+1, point[1]))
            list.append((point[0]-1, point[1]))
            list.append((point[0], point[1]+1))
            list.append((point[0], point[1]-1))
    #
    # if 0 <= x < width and 0 <= y < height:
    #     if mask[y][x] != filled_color and mask[y][x] != boundary_color:
    #         mask[y][x] = filled_color
    #         boundary_fill(x + 1, y, filled_color, boundary_color, index+1)
    #         boundary_fill(x - 1, y, filled_color, boundary_color, index+1)
    #         boundary_fill(x, y + 1, filled_color, boundary_color, index+1)
    #         boundary_fill(x, y - 1, filled_color, boundary_color, index+1)
    return mask
# 寻找一个内部点
def find_seed(x, y, mask):
    y_mid = int((min(y) + max(y)) / 2)
    x_min = min(x)
    x_max = max(x)
    for x in range(x_min, x_max):
        flag = mask[y_mid][x] == 1
        if flag == 1:
            if mask[y_mid][x+1] == 0:
                return y_mid, x+1
            else:
                y_mid += 1
    return -1, -1"""


colors = {'road_left_main': [255, 0, 0],            # 定义各个类别的颜色
          'road_horizontal_main': [0, 255, 0],
          'road_left_sub': [0, 0, 255],
          'road_right_main': [128, 128, 0]
          }

def d_print(text):
    print(colored(text, 'cyan'))

# 计算线条, bresenham 算法 
def bresenham_line(x1, y1, x2, y2):
    dx, dy = abs(x2 - x1), abs(y2 - y1)
    s1 = 1 if x2 > x1 else -1
    s2 = 1 if y2 > y1 else -1
    interchange = False
    if dy > dx:                 # 根据斜率交换迭代方向
        tmp = dx
        dx = dy
        dy = tmp
        interchange = True
    e = 2 * dy - dx
    x, y = x1, y1
    # 绘点并将矩阵对应的值设置为2
    rr = []
    cc = []
    for i in range(0, int(abs(dx) + 1)):
        rr.append(y)
        cc.append(x)
        if e >= 0:
            if not interchange:
                y += s2
            else:
                x += s1
            e -= 2 * dx
        if not interchange:
            x += s1
        else:
            y += s2
        e += 2 * dy

    return rr, cc

# 创建一个新边表
def creat_Net(point, y_max):
    f_eps = 1e-6
    Net = [([] * y_max) for i in range(y_max)]
    point_count = point.shape[0]
    for j in range(0, point_count):
        x = np.zeros(3)
        line_y = int(min(point[(j + 1) % point_count][1], point[j][1])) # 线段的最低点作为初始点

        div_0 = point[(j + 1) % point_count][0] - point[j][0]
        if math.fabs(div_0) < f_eps:
            div_0 += f_eps
        div_1 = (point[(j + 1) % point_count][1] - point[j][1]) / div_0
        if math.fabs(div_1) < f_eps:
            div_1 += f_eps

        # x[1] = 1 / ((point[(j + 1) % point_count][1] - point[j][1]) / (
        #            point[(j + 1) % point_count][0] - point[j][0]))     # x 的增量
        x[1] = 1.0 / div_1
        x[2] = max(point[(j + 1) % point_count][1], point[j][1])        # y_max
        if (point[(j + 1) % point_count][1] < point[j][1]):             # x_y_max
            x[0] = point[(j + 1) % point_count][0]
        else:
            x[0] = point[j][0]
        Net[line_y].append(x)
    return Net

# 扫描算法填充
def polygon_fill(point, mask):
    y_min = np.min(point[:, 1])
    y_max = np.max(point[:, 1])
    Net = creat_Net(point, y_max+1)
    x_sort = []
    for i in range(y_min, y_max):   # 扫描线开始进行扫描
        x = Net[i]                  # 当前是否新增边
        drop = []
        if (len(x) != 0):
            for k in x:
                x_sort.append(k)    # 若新增，放入x_sort中
                if k[2] == i:
                    drop.append(k[0])   # 出现水平线，标记横坐标
        x_image = []
        for cell in x_sort:
            x_image.append(cell[0])
        x_image.sort()              # 对当前交点x坐标排序

        if len(drop) != 0:
            for a in drop:
                x_image.remove(a)
        
        for odd, item in enumerate(x_image):    # 使用奇偶原则判断区域内外部
            if odd % 2 != 0:
                mask[i, int(previous): int(item)] = 2   # 画线
            else:
                previous = item

        filter = []                 # 判断下次是否会进入交界状态，若未进入，预测下一次x的位置，若进入，删除先前线段
        for cell in x_sort:
            if cell[2] > i+1:
                cell[0] += cell[1]
                filter.append(cell)
        x_sort = filter[:]

# 使用bresenham算法计算边缘像素
def poly_perimeter(points, mask):
    rr1 = []
    cc1 = []
    for odd, point in enumerate(points):
        if odd != 0:
            line_r, line_c = bresenham_line(previous[0], previous[1], point[0], point[1])
            previous = point
            rr1.extend(line_r)
            cc1.extend(line_c)
        else:
            previous = point
    line_r, line_c = bresenham_line(previous[0], previous[1], points[0][0], points[0][1])
    rr1.extend(line_r)
    cc1.extend(line_c)
    rr1 = np.round(rr1).astype(int)
    cc1 = np.round(cc1).astype(int)
    mask[rr1, cc1] = 1

def polygons_to_mask(json_path, img_path):
    """
    计算多边形对应分割 mask

    Parameters
    ----------
    json_path: str
        labelme 格式的多边形 label 文件
    img_path: str
        图像路径

    Returns
    -------
    mask_dict: {str: numpy.ndarray}
               {多边形名称: 每个像素对应的语义信息}
                            0: 背景
                            1: 多边形边缘像素
                            2: 多边形内部像素
    """
    print(colored(f'polygons_to_mask( {json_path}, {img_path} )', 'yellow'))
    start = time.time()
    annotations = json.load(open(json_path, 'r'))  # 加载.json文件
    img_name = annotations["imagePath"]
    labels = annotations['shapes']
    img_h = annotations['imageHeight']
    img_w = annotations['imageWidth']               # 读取json信息
    mask_dict = {}                                  # 创建结果保存字典
    for label in labels:
        mask = np.zeros((img_h, img_w), dtype=np.uint8)    # 创建mask
        class_name = label['label']
        annotations_point = label['points']
        annotations_point = np.round(annotations_point).astype(int)
        annotations_point[:, 0] = np.clip(annotations_point[:, 0], 0, img_w-1)
        annotations_point[:, 1] = np.clip(annotations_point[:, 1], 0, img_h-1)

        polygon_fill(annotations_point, mask)

        poly_perimeter(annotations_point, mask)

        mask_dict[class_name] = mask
    print(f"Time spent: {time.time() - start}s")

    return mask_dict

def demo_of_polygons_to_mask(json_path, img_in_path, img_out_path):
    """
    测试 demo，合成分割图像

    Parameters
    ----------
    json_path: str
        同上
    img_in_path: str
        同上
    img_out_path: str
        demo 图保存路径

    Returns
    -------
    NULL
    """
    res = polygons_to_mask(json_path, img_in_path)
    if not osp.exists(img_in_path):
        raise ValueError(f'path not exist: {img_in_path}')

    # todo: 合成可视化图像
    img = cv2.imread(img_in_path)
    color_col = distinctipy.get_colors(len(res))
    img_gray = np.zeros((img.shape[:2]), dtype=np.uint8)

    # for key, value in res.items():  # 遍历mask
    for idx, key in enumerate(res):
        value = res[key]
        perimeter = value == 1  # 边缘像素
        mask = value == 2  # 内部像素
        img[perimeter] = [255, 255, 255]
        _color = [int(f * 255.0) for f in color_col[idx]]
        # img[mask] = 0.5 * img[mask] + 0.5 * np.array(colors[key]) # 叠加mask到原图
        img[mask] = 0.5 * img[mask] + 0.5 * np.array(_color)
        img_gray[perimeter] = 255
        img_gray[mask] = 255

    # cv2.imshow('test', img)
    # cv2.waitKey(0)

    dir_name = osp.abspath(osp.dirname(img_out_path))
    if not osp.exists(dir_name):
        os.makedirs(dir_name)
        print(colored(f'os.makedirs( {dir_name} )', 'yellow'))

    cv2.imwrite(img_out_path, img)
    print(colored(f'save {img_out_path}', 'cyan'))

    def rename_gray(path):
        _name, _ext = osp.splitext(path)
        return f'{_name}_gray{_ext}'

    cv2.imwrite(rename_gray(img_out_path), img_gray)

if __name__ == "__main__":
    # var_json_path = "./42_02.json"
    # var_img_in_path = "./42_02.png"
    # var_img_out_path = "./temp/42_02_demo.png"

    # demo_of_polygons_to_mask(var_json_path, var_img_in_path, var_img_out_path)

    path = './vision_map'
    jsons = ['road_src_A_W_145.231_W8.1.json',
             'road_src_B_E_145.232_W9.json',
        'road_src_C_S_145.233_New.json',
        'road_src_D_N_145.234_New.json'
    ]
    imgs = ['road_src_A_W_145.231_W8.1.png',
             'road_src_B_E_145.232_W9.png',
        'road_src_C_S_145.233_New.png',
        'road_src_D_N_145.234_New.png'
    ]
    for item_json, item_img in zip(jsons, imgs):
        demo_of_polygons_to_mask(osp.join(path, item_json), osp.join(path, item_img), osp.join('./temp', item_img))
