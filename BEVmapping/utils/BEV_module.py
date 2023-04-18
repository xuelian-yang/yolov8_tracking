import json
import numpy as np
import cv2

class BEV_module:
    '''
    Alaco BEV mapping module
    '''
    def __init__(self, im, mask_shape,  bev, ROI_dict, mapping, BEV_jsonpath=None):
        anno_w = 1514
        anno_h = 798
        assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to Annotator() input images.'
        self.im = im
        self.bev = bev
        self.bev_copy = bev.copy()

        self.isHomo = BEV_jsonpath is None                  # 若未加入BEV_jsonpath，则认定为是单应性矩阵
        self.mapping = mapping
        if not self.isHomo:
            self.BEV_para = json.load(open(BEV_jsonpath, 'r'))  # 加载.json文件
            self.scale_w = self.im.shape[1] / mask_shape[1]
            self.scale_h = self.im.shape[0] / mask_shape[0]
        else:
            self.scale_w = anno_w/mask_shape[1]
            self.scale_h = anno_h/mask_shape[0]

        self.ROI = None
        if ROI_dict is not None:
            self.ROI = np.zeros((anno_h, anno_w)) if self.isHomo else np.zeros((im.shape[0], im.shape[1]))
            for value in ROI_dict.values():
                self.ROI = np.logical_or(self.ROI, value)
        # cv2.imshow('ROI', self.ROI.astype(np.uint8)*255)
        # cv2.waitKey(0)

    def to_BEV(self, det, mapping, masks, retina_masks=False):
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
        bev_points:

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
                x_bev, y_bev, _ = mapping @ np.transpose([point[0], point[1], 1])
                x_bev = int(x_bev/_)
                y_bev = int(y_bev/_)
            else:
                x_bev, y_bev = mapping[point[0]][point[1]]
                if x_bev > self.BEV_para['max_x'] or x_bev < self.BEV_para['min_x']:
                    continue
                if y_bev > self.BEV_para['max_y'] or y_bev < self.BEV_para['min_y']:
                    continue
                x_bev = int((x_bev - self.BEV_para['min_x']) * self.BEV_para['res'])
                y_bev = int((y_bev - self.BEV_para['min_y']) * self.BEV_para['res'])
            cv2.circle(self.bev_copy, (x_bev, y_bev), 10, [0, 255, 255], -1)
            bev_points[i] = [x_bev, y_bev]
        return bev_points

    def trajectory_bev(self, qs):
        for j, q in enumerate(qs):
            if len(q) == 0:
                continue
            for i, p in enumerate(q):
                thickness = int(np.sqrt(float (i + 1)) * 1.5)
                if p[0] == 'observationupdate':
                    cv2.circle(self.bev_copy, p[1], 2, color=[128, 200, 255], thickness=thickness)
                else:
                    cv2.circle(self.bev_copy, p[1], 2, color=(255, 255, 255), thickness=thickness)

