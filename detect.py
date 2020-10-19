import argparse
import os
import platform
import datetime
import shutil
import time
import socket
import pathlib
from pathlib import Path
from time import sleep

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized
from websocket import create_connection
import logging

test=0
picture=0
today1=0

def socket1():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s2:
                         s2.connect(('127.0.0.1', 50007))
                         BUFFER_SIZE=1024
                         data1='1'
                         s2.send(data1.encode())
                         print(s2.recv(BUFFER_SIZE).decode())

def socket2():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s2:
                         s2.connect(('127.0.0.1', 50007))
                         BUFFER_SIZE=1024
                         data1='2'
                         s2.send(data1.encode())
                         print(s2.recv(BUFFER_SIZE).decode())

def socket3():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(' %(module)s -  %(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    ws = create_connection("ws://127.0.0.1:12345")
    ws.send("1")
    result = ws.recv()
    logger.info("Received '{}'".format(result))
    ws.close()
    logger.info("Close")

def socket4():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(' %(module)s -  %(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    ws = create_connection("ws://127.0.0.1:12345")
    ws.send("2")
    result = ws.recv()
    logger.info("Received '{}'".format(result))
    ws.close()
    logger.info("Close")



def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')

    #初期化
    set_logging()
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  #出力フォルダーを削除します
    os.makedirs(out)  #新しい出力フォルダーを作成します
    d_today = datetime.date.today()
    f = pathlib.Path('daystext/'+ str(d_today) +'.txt')

    f.touch()
    half = device.type != 'cpu'  #精度はCUDAでのみサポートされます

    model = attempt_load(weights, map_location=device)  # FP32モデルをロードします
    imgsz = check_img_size(imgsz, s=model.stride.max())  # img_sizeを確認
    if half:
        model.half() 

    # 第2段階の分類子
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # データローダーを設定する
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # Trueを設定して、一定の画像サイズの推論を高速化
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # 名前と色を取得
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # 推論を実行する
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # 推論
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # NMSを適用
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # 分類子を適用する分類する場合
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # プロセス検出
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
               
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # 結果を印刷
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string
                    label1 = str((names[int(c)]))  # add to string
                    if label1=="cell phone":
                     print("スマホを発見しました")
                     cv2.imwrite('sumaho.jpg',im0)
                     img = cv2.imread('sumaho.jpg')
                     cv2.imshow('sumaho', img)

                     socket1()
                     socket3()
                     with open('daystext/'+str(d_today)+'.txt', 'a') as f:
                         dt_now = datetime.datetime.now()
                         f.write(str(dt_now)+"スマホを発見しました"+"\n")
                    if label1=="book":
                     print("本を発見しました")
                     cv2.imwrite('book.jpg',im0)
                     img = cv2.imread('book.jpg')
                     cv2.imshow('book', img)
                     socket2()
                     socket4()
                     with open('daystext/'+str(d_today)+'.txt', 'a') as f:
                         dt_now = datetime.datetime.now()
                         f.write(str(dt_now)+"本を発見しました"+"\n")
#==================================================================================================#
#                    if label1=="book":
#                     print("本を発見しました")
#                     cv2.imwrite('book.jpg',im0)
#                     img = cv2.imread('book.jpg')
#                     cv2.imshow('book', img)
#                     socket2()
#label1に学習させてあるモデルのモデル名を入れることで,モデル検知をすることができる. label1=="book"
#認識したものを別ウインドーにて可視化する為にその認識した画像を保存する.img = cv2.imread('book.jpg')
#認識したものを可視化する.cv2.imshow('book', img)
#==================================================================================================#
                # 結果を書く
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            #印刷時間（推論+ NMS）
            #print('%sDone. (%.3fs)' % (s, t2 - t1))
            

            # 結果のストリーミング
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                   
                    raise StopIteration

            # 結果を保存する（検出された画像）
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        video = cv2.VideoWriter('video.mp4', fourcc, fps)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % Path(out))
        if platform.system() == 'Darwin' and not opt.update:  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  #すべてのモデルを更新（SourceChangeWarningを修正するため）
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
