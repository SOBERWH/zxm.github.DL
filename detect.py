# -*-coding:utf-8-*-
# date:2021-04-5
# Author: Eric.Lee & 改进
# function: 完整的手势检测 - 支持图片、视频、摄像头
#
import os
import argparse
import torch
import numpy as np
import cv2
from models.resnet import resnet18, resnet34, resnet50, resnet101
from models.squeezenet import squeezenet1_1, squeezenet1_0
from models.shufflenetv2 import ShuffleNetV2
from models.shufflenet import ShuffleNet
from models.mobilenetv2 import MobileNetV2
from torchvision.models import shufflenet_v2_x1_5, shufflenet_v2_x1_0, shufflenet_v2_x2_0
from models.rexnetv1 import ReXNetV1
from hand_data_iter.datasets import draw_bd_handpose


class HandPoseDetector:
    def __init__(self, model_path, model_type='ReXNetV1', img_size=(256, 256), device='cuda'):
        self.img_size = img_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model(model_type, 42, img_size[0])
        self._load_model(model_path)

    def _build_model(self, model_type, num_classes, img_size):
        if model_type == 'resnet_50':
            model = resnet50(num_classes=num_classes, img_size=img_size)
        elif model_type == 'resnet_18':
            model = resnet18(num_classes=num_classes, img_size=img_size)
        elif model_type == 'resnet_34':
            model = resnet34(num_classes=num_classes, img_size=img_size)
        elif model_type == 'resnet_101':
            model = resnet101(num_classes=num_classes, img_size=img_size)
        elif model_type == "squeezenet1_0":
            model = squeezenet1_0(num_classes=num_classes)
        elif model_type == "squeezenet1_1":
            model = squeezenet1_1(num_classes=num_classes)
        elif model_type == "shufflenetv2":
            model = ShuffleNetV2(ratio=1., num_classes=num_classes)
        elif model_type == "shufflenet_v2_x1_5":
            model = shufflenet_v2_x1_5(pretrained=False, num_classes=num_classes)
        elif model_type == "shufflenet_v2_x1_0":
            model = shufflenet_v2_x1_0(pretrained=False, num_classes=num_classes)
        elif model_type == "shufflenet_v2_x2_0":
            model = shufflenet_v2_x2_0(pretrained=False, num_classes=num_classes)
        elif model_type == "shufflenet":
            model = ShuffleNet(num_blocks=[2,4,2], num_classes=num_classes, groups=3)
        elif model_type == "mobilenetv2":
            model = MobileNetV2(num_classes=num_classes)
        elif model_type == "ReXNetV1":
            model = ReXNetV1(num_classes=num_classes)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        return model.to(self.device)

    def _load_model(self, model_path):
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            self.model.eval()
            print(f'模型加载成功: {model_path}')
        else:
            raise FileNotFoundError(f'模型文件不存在: {model_path}')

    def preprocess_hand(self, img, hand_bbox=None):
        if hand_bbox is None:
            h, w = img.shape[:2]
            hand_bbox = [0, 0, w, h]
        
        x_min, y_min, x_max, y_max = hand_bbox
        
        w_ = max(abs(x_max - x_min), abs(y_max - y_min))
        w_ = w_ * 1.1
        x_mid = (x_max + x_min) / 2
        y_mid = (y_max + y_min) / 2
        
        x1, y1 = int(x_mid - w_ / 2), int(y_mid - w_ / 2)
        x2, y2 = int(x_mid + w_ / 2), int(y_mid + w_ / 2)
        
        x1 = np.clip(x1, 0, img.shape[1] - 1)
        x2 = np.clip(x2, 0, img.shape[1] - 1)
        y1 = np.clip(y1, 0, img.shape[0] - 1)
        y2 = np.clip(y2, 0, img.shape[0] - 1)
        
        hand_img = img[y1:y2, x1:x2, :]
        
        img_ = cv2.resize(hand_img, self.img_size, interpolation=cv2.INTER_CUBIC)
        img_ = img_.astype(np.float32)
        img_ = (img_ - 128.) / 256.
        img_ = img_.transpose(2, 0, 1)
        img_ = torch.from_numpy(img_).unsqueeze_(0).to(self.device)
        
        return img_, (x1, y1, x2, y2), hand_img.shape[:2]

    def predict(self, img, hand_bbox=None):
        img_tensor, crop_bbox, crop_shape = self.preprocess_hand(img, hand_bbox)
        
        with torch.no_grad():
            output = self.model(img_tensor.float())
            output = output.cpu().detach().numpy()
            output = np.squeeze(output)
        
        pts_hand = {}
        x1, y1, x2, y2 = crop_bbox
        w, h = x2 - x1, y2 - y1
        
        for i in range(int(output.shape[0] / 2)):
            x = output[i * 2 + 0] * float(w) + x1
            y = output[i * 2 + 1] * float(h) + y1
            pts_hand[str(i)] = {"x": x, "y": y}
        
        return pts_hand

    def visualize(self, img, pts_hand, bbox=None):
        vis_img = img.copy()
        
        if bbox is not None:
            cv2.rectangle(vis_img, (int(bbox[0]), int(bbox[1])), 
                         (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        
        draw_bd_handpose(vis_img, pts_hand, 0, 0)
        
        for i in range(21):
            x = int(pts_hand[str(i)]["x"])
            y = int(pts_hand[str(i)]["y"])
            cv2.circle(vis_img, (x, y), 3, (255, 50, 60), -1)
            cv2.circle(vis_img, (x, y), 1, (255, 150, 180), -1)
        
        return vis_img


class SimpleHandDetector:
    """简单的手部检测器 - 基于皮肤颜色"""
    def __init__(self):
        pass
    
    def detect(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        mask1 = cv2.inRange(hsv, lower_skin, upper_skin)
        
        lower_skin = np.array([170, 20, 70], dtype=np.uint8)
        upper_skin = np.array([180, 255, 255], dtype=np.uint8)
        
        mask2 = cv2.inRange(hsv, lower_skin, upper_skin)
        
        mask = cv2.bitwise_or(mask1, mask2)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            return [x, y, x + w, y + h]
        
        return None


def process_image(detector, img_path, hand_detector=None, output_path=None):
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图片: {img_path}")
        return
    
    hand_bbox = None
    if hand_detector is not None:
        hand_bbox = hand_detector.detect(img)
    
    pts = detector.predict(img, hand_bbox)
    vis_img = detector.visualize(img, pts, hand_bbox)
    
    if output_path:
        cv2.imwrite(output_path, vis_img)
        print(f"结果已保存: {output_path}")
    
    return vis_img


def process_video(detector, video_path, hand_detector=None, output_path=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    video_writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print("按 'q' 键退出")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        hand_bbox = None
        if hand_detector is not None:
            hand_bbox = hand_detector.detect(frame)
        
        pts = detector.predict(frame, hand_bbox)
        vis_frame = detector.visualize(frame, pts, hand_bbox)
        
        cv2.imshow('Hand Pose Detection', vis_frame)
        
        if video_writer:
            video_writer.write(vis_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    if video_writer:
        video_writer.release()
        print(f"视频已保存: {output_path}")
    cv2.destroyAllWindows()


def process_webcam(detector, camera_id=0, hand_detector=None):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"无法打开摄像头: {camera_id}")
        return
    
    print("摄像头已打开，按 'q' 键退出")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        hand_bbox = None
        if hand_detector is not None:
            hand_bbox = hand_detector.detect(frame)
        
        pts = detector.predict(frame, hand_bbox)
        vis_frame = detector.visualize(frame, pts, hand_bbox)
        
        cv2.imshow('Hand Pose Detection - Webcam', vis_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='手部关键点检测 - 支持图片/视频/摄像头')
    parser.add_argument('--model_path', type=str, default='D:\\pretrain\\ReXNetV1-size-256-wingloss102-0.122.pth', help='模型路径')
    parser.add_argument('--model', type=str, default='ReXNetV1', help='模型类型')
    parser.add_argument('--img_size', type=tuple, default=(256, 256), help='输入图片尺寸')
    parser.add_argument('--mode', type=str, default='image', choices=['image', 'video', 'webcam'], help='检测模式: image/video/webcam')
    parser.add_argument('--input', type=str, default='./image/', help='输入路径 (图片/视频文件夹或文件)')
    parser.add_argument('--output', type=str, default=None, help='输出路径')
    parser.add_argument('--use_hand_detector', action='store_true', help='使用皮肤颜色手部检测器')
    parser.add_argument('--camera_id', type=int, default=0, help='摄像头ID')
    
    args = parser.parse_args()
    
    print('='*60)
    print('手部关键点检测系统')
    print('='*60)
    print(f'模式: {args.mode}')
    print(f'模型: {args.model}')
    print(f'输入: {args.input}')
    if args.output:
        print(f'输出: {args.output}')
    print('='*60)
    
    detector = HandPoseDetector(
        model_path=args.model_path,
        model_type=args.model,
        img_size=args.img_size
    )
    
    hand_detector = SimpleHandDetector() if args.use_hand_detector else None
    
    if args.mode == 'image':
        if os.path.isdir(args.input):
            image_files = [f for f in os.listdir(args.input) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            for img_file in image_files:
                img_path = os.path.join(args.input, img_file)
                print(f'\n处理: {img_file}')
                
                output_path = None
                if args.output:
                    os.makedirs(args.output, exist_ok=True)
                    output_path = os.path.join(args.output, f'result_{img_file}')
                
                vis_img = process_image(detector, img_path, hand_detector, output_path)
                
                if vis_img is not None:
                    cv2.imshow('Result', vis_img)
                    if cv2.waitKey(0) & 0xFF == ord('q'):
                        break
            cv2.destroyAllWindows()
        else:
            vis_img = process_image(detector, args.input, hand_detector, args.output)
            if vis_img is not None:
                cv2.imshow('Result', vis_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    
    elif args.mode == 'video':
        process_video(detector, args.input, hand_detector, args.output)
    
    elif args.mode == 'webcam':
        process_webcam(detector, args.camera_id, hand_detector)
    
    print('\n检测完成!')
