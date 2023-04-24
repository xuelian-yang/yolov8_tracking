import argparse
import copy
import cv2
from distinctipy import distinctipy
import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import time
import PIL
import platform
import numpy as np
from pathlib import Path
import torch
from distinctipy import distinctipy
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov8') not in sys.path:
    sys.path.append(str(ROOT / 'yolov8'))  # add yolov5 ROOT to PATH
if str(ROOT / 'trackers' / 'strongsort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strongsort'))  # add strong_sort ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import logging
from yolov8.ultralytics.nn.autobackend import AutoBackend
from yolov8.ultralytics.yolo.data.dataloaders.stream_loaders import LoadImages, LoadStreams
from yolov8.ultralytics.yolo.data.utils import IMG_FORMATS, VID_FORMATS
from yolov8.ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, SETTINGS, callbacks, colorstr, ops
from yolov8.ultralytics.yolo.utils.checks import check_file, check_imgsz, check_imshow, print_args, check_requirements
from yolov8.ultralytics.yolo.utils.files import increment_path
from yolov8.ultralytics.yolo.utils.torch_utils import select_device
from yolov8.ultralytics.yolo.utils.ops import Profile, non_max_suppression, scale_boxes, process_mask, process_mask_native
from yolov8.ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from yolov8.ultralytics.yolo.utils.torch_utils import strip_optimizer

from trackers.multi_tracker_zoo import create_tracker

from mask_generation import mask_generation
from BEVmapping.utils.BEV_module import BEV_module, BEV_parameters

from termcolor import colored
import os.path as osp

import tkinter  # pip install pytk

from common.util import setup_log, d_print, d_print_b, d_print_g, d_print_r, d_print_y, get_name
from configs.workspace import WorkSpace

logger = logging.getLogger(__name__)
g_enable_debug = True


def get_sat_map():
    """
    获取卫星底图.
    """
    path_sat = 'BEVmapping/BEVimages/google_sat_img.png'
    path_roi = 'BEVmapping/BEVimages/google_sat_roi_v1.png'
    if not osp.exists(path_sat) or not osp.exists(path_roi):
        raise ValueError(f'missing map.')
    img_sat = cv2.imread(path_sat)
    img_roi = cv2.imread(path_roi)
    sat_map = cv2.addWeighted(img_sat, 0.85, img_roi, 0.15, 0)  # 卫星图与 roi 叠加
    # return sat_map
    return img_sat


def load_bev_config(source):
    rtspdict = {
        'rtsp://admin:hik12345=@10.10.145.231/Streaming/Channels/101': 'src_A_W_145.231_W8.1',
        'rtsp://admin:hik12345=@10.10.145.232/Streaming/Channels/101': 'src_B_E_145.232_W9',
        'rtsp://admin:hik12345=@10.10.145.233/Streaming/Channels/101': 'src_C_S_145.233_New',
        'rtsp://admin:hik12345=@10.10.145.234/Streaming/Channels/101': 'src_D_N_145.234_New',
    }
    mp4dict = {
        'D:/alaco_video_archive/multi-stream-reocrds/W91_2023-04-18_09_45_32.mp4': 'src_A_W_145.231_W8.1',
        'D:/alaco_video_archive/multi-stream-reocrds/W92_2023-04-18_09_45_32.mp4': 'src_B_E_145.232_W9',
        'D:/alaco_video_archive/multi-stream-reocrds/W93_2023-04-18_09_45_32.mp4': 'src_C_S_145.233_New',
        './videos/W91_2023-04-19_18_18_24.mp4': 'src_A_W_145.231_W8.1',
        './videos/W92_2023-04-19_18_18_24.mp4': 'src_B_E_145.232_W9',
        './videos/W93_2023-04-19_18_18_24.mp4': 'src_C_S_145.233_New',
        './videos/W94_2023-04-19_18_18_24.mp4': 'src_D_N_145.234_New',
    }

    bev_parameters = None

    if source.endswith('.txt'):                 # 若多路视频，分别建立各自参数
        with open(source, 'r') as f:
            lines = f.readlines()
            bev_parameters = []         # 每个摄像头保存一个parameter对象：BEV_module.BEV_parameters
            for str_line in lines:
                str_line = str_line.strip()
                if str_line.startswith('#'):  # 忽略注释掉的路径
                    continue
                # 根据每行的前后缀判断是 rtsp 还是 mp4 地址
                is_rtsp = str_line.startswith('rtsp://')
                if is_rtsp:
                    if str_line in rtspdict.keys():
                        camera = rtspdict[str_line]
                    else:
                        raise ValueError(f'unknown parameters for {str_line}')
                elif str_line.endswith('.mp4'):
                    if str_line in mp4dict.keys():
                        camera = mp4dict[str_line]
                    else:
                        raise ValueError(f'unknown parameters for {str_line}')

                to_bev = True
                direct_kf = True
                USING_ROI = True

                direct_list = None
                ROI_dict = None  # 道路 ROI
                bev = cv2.imread(os.path.join('BEVmapping', 'BEVimages', 'perspective_' + camera + '.png'))  # 由路侧前视图像 warping 得到的俯视图
                if os.path.exists(os.path.join('BEVmapping', 'mapping_matrix', camera + '.json')):
                    BEV_jsonpath = os.path.join('BEVmapping', 'mapping_matrix', camera + '.json')  # 经纬度转俯视图的参数
                    mapping = np.load(os.path.join('BEVmapping', 'mapping_matrix', 'mapping_' + camera + '.npy'))  # 由内外参计算的像素映射表
                else:
                    BEV_jsonpath = None
                    mapping = np.load(os.path.join('BEVmapping', 'homography', 'homo_' + camera + '.npy'))  # 由单应性矩阵计算的像素映射表
                if USING_ROI:
                    ROI_json_path = os.path.join('BEVmapping', 'json', 'road_' + camera + '.json')  # labelme 标注的道路区域
                    src_path = os.path.join('BEVmapping', 'images', camera + '.png')  # 玉海东路兴慈二路口四相机图 1514x798 的分辨率
                    ROI_dict = mask_generation.polygons_to_mask(ROI_json_path, src_path)  # 道路 ROI

                # 添加多相机参数
                bev_parameters.append(
                    BEV_parameters(bev, to_bev, direct_kf, direct_list, ROI_dict, BEV_jsonpath, mapping)
                )
    else: # source.endswith('.txt'):
        camera = rtspdict[source]
        # camera = 'W42'
        to_bev = True
        direct_kf = True
        USING_ROI = True

        direct_list = None
        ROI_dict = None
        bev = cv2.imread(os.path.join('BEVmapping', 'BEVimages', 'perspective_'+camera+'.png'))
        if os.path.exists(os.path.join('BEVmapping', 'mapping_matrix', camera+'.json')):
            BEV_jsonpath = os.path.join('BEVmapping', 'mapping_matrix', camera+'.json')
            mapping = np.load(os.path.join('BEVmapping', 'mapping_matrix', 'mapping_'+camera+'.npy'))
        else:
            BEV_jsonpath = None
            mapping = np.load(os.path.join('BEVmapping', 'homography', 'homo_'+camera+'.npy'))
        if USING_ROI:
            ROI_json_path = os.path.join('BEVmapping', 'json', 'road_'+camera+'.json')
            src_path = os.path.join('BEVmapping', 'images', camera+'.png')
            ROI_dict = mask_generation.polygons_to_mask(ROI_json_path, src_path)

        # 添加单相机进入BEV_parameters
        bev_parameters = [BEV_parameters(bev, to_bev, direct_kf, direct_list, ROI_dict, BEV_jsonpath, mapping)]
        #############################################

    return bev_parameters


def check_source_type(source, nosave):
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download
    return webcam


def make_save_dir(yolo_weights, reid_weights, name, project, exist_ok, save_txt):
    if not isinstance(yolo_weights, list):  # single yolo model
        exp_name = yolo_weights.stem
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = Path(yolo_weights[0]).stem
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name if name else exp_name + "_" + reid_weights.stem
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    return save_dir


def load_model(device, yolo_weights, dnn, half, imgsz):
    my_time_lap = time.time()
    is_seg = '-seg' in str(yolo_weights)
    model = AutoBackend(yolo_weights, device=device, dnn=dnn, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_imgsz(imgsz, stride=stride)  # check image size

    if g_enable_debug:
        logger.warning(f'@@@==> load model elapsed {time.time() - my_time_lap:.3f} seconds')
        logger.info(f'model: \n\ttype: {type(model)}, \n\tstride: {stride}, \n\tnames: {names}, \n\tpt: {type(pt)}')
        logger.info(f'selected device: {device}')
        logger.info(f'is_seg:          {is_seg}')
        logger.info(f'imgsz:           {imgsz}')
    return is_seg, model, stride, names, pt, imgsz


def load_dataset(bs, webcam, source, imgsz, stride, pt, model, vid_stride):
    my_time_lap = time.time()
    if webcam:
        show_vid = check_imshow(warn=True)
        dataset = LoadStreams(
            source,
            imgsz=imgsz,
            stride=stride,
            auto=pt,
            transforms=getattr(model.model, 'transforms', None),
            vid_stride=vid_stride
        )
        bs = len(dataset)
        if g_enable_debug:
            logger.info(f'webcam mode: {bs}')
    else:
        dataset = LoadImages(
            source,
            imgsz=imgsz,
            stride=stride,
            auto=pt,
            transforms=getattr(model.model, 'transforms', None),
            vid_stride=vid_stride
        )
        if g_enable_debug:
            logger.info(f'ImageList mode: {bs}')

    if g_enable_debug:
        logger.warning(f'@@@==> load data elapsed {time.time() - my_time_lap:.3f} seconds')
    return bs, dataset


def create_trackers(dataset, bs, bev_parameters, tracking_method, tracking_config, reid_weights, device, half):
    my_time_lap = time.time()
    tracker_list = []
    bev_modules = []
    for batch in dataset:
        path, im, im0s, vid_cap, s = batch
        break
    for i in range(bs):
        bev_module = BEV_module(im0s[i], im[i].shape[1:], bev_parameters[i].ROI_dict, bev_parameters[i].mapping, bev_parameters[i].BEV_jsonpath,)
        tracker = create_tracker(tracking_method, tracking_config, reid_weights, device, half)
        tracker_list.append(tracker, )
        bev_modules.append(bev_module)
        if hasattr(tracker_list[i], 'model'):
            if hasattr(tracker_list[i].model, 'warmup'):
                tracker_list[i].model.warmup()
    if g_enable_debug:
        logger.warning(f'@@@==> create trackers elapsed {time.time() - my_time_lap:.3f} seconds')

    return tracker_list, bev_modules


@torch.no_grad()
def run(
        source='0',
        yolo_weights=WEIGHTS / 'yolov5m.pt',  # model.pt path(s),
        reid_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        tracking_method='strongsort',
        tracking_config=None,
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_single_vid=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_trajectories=False,  # save trajectories for each track
        save_vid=False,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs' / 'track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        retina_masks=False,
):
    global g_enable_debug
    enable_heading_angle = True

    # 1. 打印函数参数
    if g_enable_debug:
        _all_variables = dir()
        _text = 'run('
        for _idx, _name in enumerate(sorted(_all_variables)):
            if not _name.startswith('__'):
                _val = eval(_name)
                _text += f'\n\tparam[{_idx:02d}]: {_name:<20s} = [{str(type(_val)):<30s}] {_val}'
        _text += ')'
        logger.info(_text)
        logger.warning(f'@@@==> exec to run(..) elapsed {time.time() - time_beg:.3f} seconds')
        my_time_lap = time.time()

    # 2. 根据视频源加载 BEV 映射参数
    bev_parameters = load_bev_config(source)
    if g_enable_debug:  # 打印参数
        logger.warning(f'@@@==> init param elapsed {time.time() - my_time_lap:.3f} seconds')
        my_time_lap = time.time()
        for idx, item_bev_param in enumerate(bev_parameters):
            logger.info(f'BEV config [{idx}]:\n{item_bev_param}')

    bev_merge = cv2.imread('./BEVmapping/BEVimages/bev_weighted_v1_scale-1.0.png')
    show_bev_vid = True

    source = str(source)
    # 3. 检测数据源类型
    webcam = check_source_type(source, nosave)

    # 4. 创建必要的保存目录
    save_dir = make_save_dir(yolo_weights, reid_weights, name, project, exist_ok, save_txt)

    # 5. 加载模型
    device = select_device(device)
    is_seg, model, stride, names, pt, imgsz = load_model(device, yolo_weights, dnn, half, imgsz)

    # 6. 加载数据源
    bs = 1
    bs, dataset = load_dataset(bs, webcam, source, imgsz, stride, pt, model, vid_stride)

    vid_path, vid_writer, txt_path = [None] * bs, [None] * bs, [None] * bs
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup

    # 7. 创建 trackers (Create as many strong sort instances as there are video sources)
    tracker_list, bev_modules = create_trackers(dataset, bs, bev_parameters, tracking_method, tracking_config, reid_weights, device, half)  # 为每个类添加 BEV_module

    outputs, q_bev, matches = [None] * bs, [None] * bs, [None] * bs
    color_col = distinctipy.get_colors(bs)   # 为每个ID分配颜色，用不同颜色表示位置和速度
    for idx in range(bs):
        _color = [int(f * 255.0) for f in color_col[idx]]
        color_col[idx] = _color

    inited_output_files = []  # 标记 xxx.mp4.txt 文件以 'at' 还是 'wt' 形式打开.
    ws = WorkSpace()  # 存储中间结果的相对路径

    # Run tracking
    #model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile(), Profile())
    time_fps_elapsed = time.time()
    for frame_idx, batch in enumerate(dataset):
        ts = time.time()
        path, im, im0s, vid_cap, s = batch
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False

        with dt[0]:  # 图像载入显存
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255.0  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            preds = model(im, augment=augment, visualize=visualize)
        if g_enable_debug: logger.info(f'preds: {type(preds)}')

        # Apply NMS
        with dt[2]:
            if is_seg:
                p = non_max_suppression(preds[0], conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)
                proto = preds[1][-1]
                if g_enable_debug: logger.info(f'p: {type(p)} {len(p)},'
                                               f'\n\tp[0]: {type(p[0])} {p[0].size()} {p[0].is_cuda} {p[0].device} {p[0].ndim},'
                                               f'\n\tproto: {type(proto)} {proto.size()}')
            else:
                p = non_max_suppression(preds, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                if g_enable_debug: logger.info(f'p: {type(p)}')

        # Process detections
        masks = [[] for i in range(bs)]
        bbox_all = [[] for i in range(bs)]
        st = ''
        for i, det in enumerate(p):  # detections per image
            start11 = time.time()
            seen += 1

            if webcam:  # bs >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                p = Path(p)  # to Path
                s += f'{i}: '
                txt_file_name = p.name
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path

                # video file
                if source.endswith(VID_FORMATS):
                    txt_file_name = p.stem
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                # folder with imgs
                else:
                    txt_file_name = p.parent.name  # get folder name containing current img
                    save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...
            # curr_frames[i] = im0

            txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop

            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            st += f'{i},stage11:{time.time()-start11},'
            start12 = time.time()
            # if hasattr(tracker_list[i], 'tracker') and hasattr(tracker_list[i].tracker, 'camera_update'):
            #     if prev_frames[i] is not None and curr_frames[i] is not None:  # camera motion compensation
            #         tracker_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])
            st += f'stage12:{time.time()-start12},'

            if det is not None and len(det):
                start2 = time.time()
                bbox_all[i] = copy.deepcopy(det[:, :4].cpu().numpy())
                if is_seg:
                    shape = im0.shape
                    # scale bbox first the crop masks
                    if retina_masks:
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], shape).round()  # rescale boxes to im0 size
                        masks[i] = process_mask_native(proto[i], det[:, 6:], det[:, :4], im0.shape[:2])  # HWC
                    else:
                        masks[i] = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], shape).round()  # rescale boxes to im0 size
                else:
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

                # Print results - 单帧检测结果统计
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                st += f'stage2:{time.time()-start2},'

                # pass detections to strongsort
                with dt[3]:
                    # ======================================================== #
                    # < FOV 点映射至 BEV >
                    # ======================================================== #
                    if bev_parameters[i].to_bev:
                        try:
                            # 图像坐标映射到俯视图坐标
                            fov_points, bev_points, border, _scale_w, _scale_h = bev_modules[i].to_BEV(
                                masks=masks[i],
                                det=bbox_all[i],
                                gnd=True,
                                debug=True
                            )
                            bev_parameters[i].add_bev_points(bev_points)
                        except Exception as e:
                            print(masks, len(masks), bbox_all)
                            print(repr(e))
                            quick_exit()
                            exit(0)

                    # outputs[i], q_bev[i] = tracker_list[i].update(det.cpu(), im0, bev_points)  # ITTI 扩展的 tracker
                    outputs[i], q_bev[i], matches[i] = tracker_list[i].update(det.cpu(), im0, bev_points, match=True)
                    bev_parameters[i].add_q_bev(q_bev[i])

                    # draw boxes for visualization
                    if len(outputs[i]) > 0:
                        if is_seg:
                            # Mask plotting
                            annotator.masks(
                                masks[i],
                                colors=[colors(x, True) for x in det[:, 5]],
                                im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(device).permute(2, 0, 1).flip(0).contiguous() /
                                255 if retina_masks else im[i]
                            )
                        
                        for j, (output) in enumerate(outputs[i]):
                            
                            bbox = output[0:4]
                            id = output[4]
                            cls = output[5]
                            conf = output[6]

                            if save_txt:
                                # to MOT format
                                bbox_left = output[0]
                                bbox_top = output[1]
                                bbox_w = output[2] - output[0]
                                bbox_h = output[3] - output[1]
                                # Write MOT compliant results to file
                                with open(txt_path + '.txt', 'a') as f:
                                    f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                                bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                            if save_vid or save_crop:  # Add bbox/seg to image
                                c = int(cls)  # integer class
                                id = int(id)  # integer id
                                label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                                    (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                                color = colors(c, True)
                                annotator.box_label(bbox, label, color=color)
                                
                                if save_trajectories and tracking_method == 'strongsort':
                                    q = output[7]
                                    tracker_list[i].trajectory(im0, q, color=color)
                                if save_crop:
                                    txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                    save_one_box(np.array(bbox, dtype=np.int16), imc, file=save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)
                if enable_heading_angle:
                    im2_ = cv2.resize(im0s[i], (1514, 798))
                    for _t in border:
                        for _t_t in _t:
                            _x, _y = _t_t
                            _x = int(_x * _scale_w)
                            _y = int(_y * _scale_h)
                            im2_[_y-1:_y+1, _x-1:_x+1, :] = (0, 0, 255)
                    cv2.imwrite(osp.join(ws.get_temp_dir(), f'{get_name(path[i])}_2_{frame_idx:04d}.png'), im2_)
                start3 = time.time()
                ################ 保存Result
                _outfile_permission = 'at'
                # _outfile_name = path[i]+'.txt'
                _outfile_name = osp.join(ws.get_temp_dir(), get_name(path[i]) + '.txt')
                if _outfile_name not in inited_output_files:
                    _outfile_permission = 'wt'
                    inited_output_files.append(_outfile_name)

                # logger.info(f'with open({_outfile_name}, {_outfile_permission}) as f')
                with open(_outfile_name, _outfile_permission) as f:
                    result = []
                    for t, d_id in matches[i]:
                        if len(bev_points[d_id]) != 0:                      # 过滤掉ROI区域外的点
                            if bev_parameters[i].direct_kf:                 # 计算BEV下卡尔曼预测值
                                bev_parameters[i].direct_list = []
                                x, y = t.mean_bev[0:2]
                                dx, dy = t.mean_bev[4:6]
                                x2, y2 = [x + dx * 20, y + dy*20]
                                bev_parameters[i].direct_list.append([x, y, x2, y2])
                                result.append([{'ts': ts, 'ID': t.track_id, 'FOVxy': fov_points[d_id].tolist(), 'BEVxy': bev_points[d_id],
                                                'confidence': det[d_id][4].cpu().item(), 'direct': [dx, dy]}])
                                f.writelines(
                                    f'ts:{ts},ID:{t.track_id},FOVxy:{fov_points[d_id].tolist()},BEVxy:{bev_points[d_id]},confidence:{det[d_id][4].cpu().item()},direct:{[dx, dy]}\n')
                            else:
                                result.append([{'ts': ts, 'ID': t.track_id, 'FOVxy': fov_points[d_id].tolist(), 'BEVxy': bev_points[d_id],
                                                'confidence': det[d_id][4].cpu().item()}])
                                f.writelines(
                                    f'ts:{ts},ID:{t.track_id},FOVxy:{fov_points[d_id].tolist()},BEVxy:{bev_points[d_id]},confidence:{det[d_id][4].cpu().item()}\n')
            else:
                bev_parameters[i].add_bev_points([])
                bev_parameters[i].add_q_bev([])
                #tracker_list[i].tracker.pred_n_update_all_tracks()

            start4 = time.time()
            bev_parameters[i].update_img(annotator.result())

        if show_bev_vid:
            bev_merge_copy = bev_merge.copy()
            for i in range(bs):
                # 画BEV点
                for bev_point in bev_parameters[i].bev_points:
                    if len(bev_point) == 0:
                        continue
                    x_bev, y_bev = bev_point
                    cv2.circle(bev_merge_copy, (x_bev, y_bev), 10, color_col[i], -1)
                # 画航向角
                if bev_parameters[i].direct_kf and bev_parameters[i].direct_list:
                    for x, y, x2, y2 in bev_parameters[i].direct_list:
                        cv2.line(bev_merge_copy, (int(x), int(y)), (int(x2), int(y2)), color_col[i], 10)
                # 画轨迹
                if bev_parameters[i].to_bev:
                    bev_modules[i].trajectory_bev(bev_parameters[i].q_bev, bev_merge_copy)

            if platform.system() == 'Linux' and p not in windows:
                windows.append(p)
                cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                # cv2.namedWindow('BEV', 0)

            if platform.system() == 'Windows' and p not in windows:
                windows.append(p)
                cv2.namedWindow(str(p), cv2.WINDOW_NORMAL)
                tk_win = tkinter.Tk()  # 获取显示器分辨率
                win_w = tk_win.winfo_screenwidth()
                win_h = tk_win.winfo_screenheight()
                logger.info(f'screen: {win_w} x {win_h}')
                cv2.resizeWindow(str(p), win_w, win_h-100)
                cv2.moveWindow(str(p), 0, 0)
                _win_sz = cv2.getWindowImageRect(str(p))
                logger.info(f'WinSz of {str(p)}: {_win_sz}')

            show_w = int(bev_parameters[0].img.shape[1] * 2)
            bev_merge_copy = cv2.resize(bev_merge_copy, (show_w, int(bev_merge_copy.shape[0]*(show_w/bev_merge_copy.shape[1]))))
            assert(len(bev_parameters) > 0)
            _tmp_ims = []
            for _item in bev_parameters:
                _tmp_ims.append(_item.img)
            for _ in range(len(bev_parameters), 5):
                _tmp_ims.append(np.zeros_like(_tmp_ims[0]))
            im0 = np.hstack((_tmp_ims[0], _tmp_ims[1]))
            im1 = np.hstack((_tmp_ims[2], _tmp_ims[3]))
            htich = np.vstack((im0, im1, bev_merge_copy))
            htich = cv2.resize(htich, (int(htich.shape[1]/4), int(htich.shape[0]/4)))

            cv2.imshow(str(p), htich)

            if cv2.waitKey(1) == ord('q'):  # 1 millisecond
                quick_exit()
                exit()
            # prev_frames[i] = curr_frames[i]
            st += f'stage4:{time.time() - start4},'
            st += f"model:{sum([dt.dt for dt in dt if hasattr(dt, 'dt')]) * 1E3:.1f}ms,"
        # print(time.time() - ts)
        logger.info(f'frame {frame_idx} elapsed {time.time() - ts:.3f} seconds.')

        # Print total time (preprocessing + inference + NMS + tracking)
        LOGGER.info(f"frame:{frame_idx} {s}{'' if len(det) else '(no detections), '}{sum([dt.dt for dt in dt if hasattr(dt, 'dt')]) * 1E3:.1f}ms")

        if frame_idx > 0 and frame_idx % 10 == 0:
            logger.info(f'fps: {(time.time() - time_fps_elapsed)*1000.0 / (1.0 * frame_idx):.3f} ms per frame')

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms {tracking_method} update per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list((save_dir / 'tracks').glob('*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=Path, default='yolov8n-seg.pt', help='model.pt path(s)')
    parser.add_argument('--reid-weights', type=Path, default='osnet_x0_25_msmt17.pt')
    parser.add_argument('--tracking-method', type=str, default='strongsort', help='strongsort, ocsort, bytetrack')
    parser.add_argument('--tracking-config', type=Path, default=None)
    # parser.add_argument('--source', type=str, default='rtsp://10.10.132.6:554/openUrl/yH3gzrq', help='file/dir/URL/glob, 0 for webcam')
    # parser.add_argument('--source', type=str, default='rtsp://admin:hik12345=@10.10.145.231/Streaming/Channels/101', help='file/dir/URL/glob, 0 for webcam')
    # parser.add_argument('--source', type=str, default=r'/home/itti/Downloads/W_4.1_chan2_20230316_130436.mp4', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--source', type=str, default='./mp4.txt', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.2, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.4, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-single-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-trajectories', action='store_true', help='save trajectories for each track')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', default=[0,1,2,3,5,7], type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs' / 'track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--retina-masks', action='store_true', help='whether to plot masks in native resolution')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    opt.tracking_config = ROOT / 'trackers' / opt.tracking_method / 'configs' / (opt.tracking_method + '.yaml')
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


def quick_exit():
    time_end = time.time()
    logger.warning(f'{this_filename} elapsed {time_end - time_beg:.3f} seconds')
    print(colored(f'{this_filename} elapsed {time_end - time_beg:.3f} seconds', 'yellow'))


if __name__ == "__main__":
    time_beg = time.time()
    this_filename = osp.basename(__file__)
    setup_log(this_filename)
    logger.info(f'torch.__version__: {torch.__version__}')

    opt = parse_opt()
    main(opt)

    time_end = time.time()
    logger.warning(f'{this_filename} elapsed {time_end - time_beg:.3f} seconds')
    print(colored(f'{this_filename} elapsed {time_end - time_beg:.3f} seconds', 'yellow'))
