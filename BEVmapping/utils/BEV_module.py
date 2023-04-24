import json
import logging
import numpy as np
import cv2

class BEV_module:
    '''
    Alaco BEV mapping module
    '''
    def __init__(self, im_shape, mask_shape, ROI_dict, mapping, BEV_jsonpath=None):
        '''

        Parameters
        ----------
        im_shape:
        mask_shape: 入模size
        ROI_dict: ROI区域字典
        mapping: 单应性矩阵或者映射表
        BEV_jsonpath: 若为映射表, 存放BEV底图生成参数
        '''
        anno_w = 1514
        anno_h = 798
        self.im_shape = im_shape

        self.isHomo = BEV_jsonpath is None                  # 若未加入BEV_jsonpath，则认定为是单应性矩阵
        self.mapping = mapping
        if not self.isHomo:
            self.BEV_para = json.load(open(BEV_jsonpath, 'r'))  # 加载.json文件
            self.scale_w = self.im_shape[1] / mask_shape[1]
            self.scale_h = self.im_shape[0] / mask_shape[0]
        else:
            self.scale_w = anno_w/mask_shape[1]
            self.scale_h = anno_h/mask_shape[0]

        self.ROI = None
        if ROI_dict is not None:
            self.ROI = np.zeros((anno_h, anno_w)) if self.isHomo else np.zeros((im_shape[0], im_shape[1]))
            for value in ROI_dict.values():
                self.ROI = np.logical_or(self.ROI, value)

    def to_BEV(self, det, masks, retina_masks=False, gnd=False, debug=False):
        '''
        Convert FOV point to BEV perspection

        Parameters
        ----------
        det: xyxy
        mapping: homography or mapping matrix
        masks: masks after NMS
        retina_masks: bool value

        Returns
        -------
        bev_points: list of coord
        '''
        det_cpu = det.astype(np.int16)
        # print(np.min(det_cpu)
        masks_cpu = masks.cpu().numpy()
        x1 = det_cpu[:, 0]
        y1 = det_cpu[:, 1]
        x2 = det_cpu[:, 2]
        y2 = det_cpu[:, 3]
        indexes = np.argsort(x1)

        num = 1
        current_items = {indexes[0]}
        border = [[] for i in range(len(det_cpu))]
        points = [[] for i in range(len(det_cpu))]
        for i in range(np.min(x1), np.max(x2 + 1)):
            for item in current_items.copy():
                if x2[item] == i:
                    current_items.discard(item)
                    # 结束之后计算接地点
                    if len(border[item]) != 0:
                        points[item] = np.average(np.array(border[item]), axis=0)
                        if not retina_masks:
                            points[item][0] *= self.scale_w
                            points[item][1] *= self.scale_h
                    else:
                        points[item] = [-1, -1]
            if num < len(det_cpu) and x1[indexes[num]] <= i:
                current_items.add(indexes[num])
                num += 1
            if len(current_items) == 0:
                continue
            for item in current_items:
                column = masks_cpu[item][y1[item]:y2[item], i]
                column_index = np.where(column == 1)
                if len(column_index[0]) > 0:
                    j = np.max(column_index)
                    border[item].append([i, j + y1[item]])
                    # self.im[int((j + y1[item])*self.im.shape[1]/im_gpu.shape[1]), int(i*self.im.shape[1]/im_gpu.shape[1])] = [255, 255, 0] # 绘制border
        try:
            points = np.array(points).astype(np.int16)
        except ValueError:
            print(f"points: {points}\nborder: {border}\ndet: {det_cpu}")
            for i, mask in enumerate(masks_cpu):
                cv2.imwrite(f'{i}.jpg', mask * 255)

        # 计算BEV映射点
        bev_points = [[] for i in range(len(det_cpu))]
        for i, point in enumerate(points):
            if point[0] < 0:
                continue
            if self.ROI is not None:
                if self.ROI[point[1], point[0]] == False:
                    continue
            # cv2.circle(self.im, (point[0], point[1]), 10, [0, 255, 255], -1)
            if self.isHomo:                                                 # 使用homography映射
                x_bev, y_bev, _ = self.mapping @ np.transpose([point[0], point[1], 1])
                x_bev = int(x_bev/_)
                y_bev = int(y_bev/_)
            else:
                x_bev, y_bev = self.mapping[point[0]][point[1]]
                if x_bev > self.BEV_para['max_x'] or x_bev < self.BEV_para['min_x']:
                    continue
                if y_bev > self.BEV_para['max_y'] or y_bev < self.BEV_para['min_y']:
                    continue
                x_bev = int((x_bev - self.BEV_para['min_x']) * self.BEV_para['res'])
                y_bev = int((y_bev - self.BEV_para['min_y']) * self.BEV_para['res'])
            bev_points[i] = [x_bev, y_bev]
        if gnd:
            if debug:
                return points, bev_points, border, self.scale_w, self.scale_h
            return points, bev_points
        return bev_points

    def trajectory_bev(self, qs, bev):
        for j, q in enumerate(qs):
            if len(q) == 0:
                continue
            for i, p in enumerate(q):
                thickness = int(np.sqrt(float (i + 1)) * 1.5)
                if p[0] == 'observationupdate':
                    cv2.circle(bev, p[1], 2, color=[128, 200, 255], thickness=thickness)
                else:
                    cv2.circle(bev, p[1], 2, color=(255, 255, 255), thickness=thickness)

class BEV_parameters:
    '''
    保存各个摄像头的参数
    '''
    def __init__(self, bev, to_bev, direct_kf, direct_list, ROI_dict, BEV_jsonpath, mapping):
        self.bev = bev                   # np.ndarray 由路侧前视图像 warping 得到的俯视图
        self.to_bev = to_bev             # bool
        self.direct_kf = direct_kf       # bool
        self.direct_list = direct_list   # None ?
        self.ROI_dict = ROI_dict         # dict of {str: np.ndarray}  道路 ROI
        self.BEV_jsonpath = BEV_jsonpath # str 经纬度转俯视图的参数
        self.mapping = mapping           # np.ndarray 由单应性矩阵计算的像素映射表
        self.bev_copy = self.bev.copy()
        self.direct_list = []
        self.q_bev = []
        self.bev_points = []
        self.img = []

    def add_q_bev(self, q_bev):
        self.q_bev = q_bev

    def add_bev_points(self, bev_points):
        self.bev_points = bev_points

    def update_img(self, img):
        self.img = img

    def refresh_bev(self):
        self.bev_copy = self.bev.copy()

    def __str__(self):
        return f'BEV_parameters:' \
               f'\n\tbev:          {self.bev.shape}' \
               f'\n\tto_bev:       {self.to_bev}' \
               f'\n\tdirect_kf:    {self.direct_kf}' \
               f'\n\tdirect_list:  {self.direct_list}' \
               f'\n\tROI_dict:     {self.ROI_dict.keys()}' \
               f'\n\tBEV_jsonpath: {self.BEV_jsonpath}' \
               f'\n\tmapping:      {self.mapping.shape}'

    def __repr__(self):
        return self.__str__()
