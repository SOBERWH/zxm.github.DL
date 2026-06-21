# -*-coding:utf-8-*-
# 批量测试脚本 - 不显示界面，直接保存结果
import os
import torch
import numpy as np
import cv2
from models.rexnetv1 import ReXNetV1
from hand_data_iter.datasets import draw_bd_handpose


class HandPoseDetector:
    def __init__(self, model_path, model_type='ReXNetV1', img_size=(256, 256), device='cpu'):
        self.img_size = img_size
        self.device = torch.device(device)
        self.model = ReXNetV1(num_classes=42)
        self._load_model(model_path)

    def _load_model(self, model_path):
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            self.model.eval()
            print(f'模型加载成功: {model_path}')
        else:
            raise FileNotFoundError(f'模型文件不存在: {model_path}')

    def predict(self, img):
        h, w = img.shape[:2]
        
        img_ = cv2.resize(img, self.img_size, interpolation=cv2.INTER_CUBIC)
        img_ = img_.astype(np.float32)
        img_ = (img_ - 128.) / 256.
        img_ = img_.transpose(2, 0, 1)
        img_ = torch.from_numpy(img_).unsqueeze_(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(img_.float())
            output = output.cpu().detach().numpy()
            output = np.squeeze(output)
        
        pts_hand = {}
        for i in range(int(output.shape[0] / 2)):
            x = output[i * 2 + 0] * float(w)
            y = output[i * 2 + 1] * float(h)
            pts_hand[str(i)] = {"x": x, "y": y}
        
        return pts_hand

    def visualize(self, img, pts_hand):
        vis_img = img.copy()
        draw_bd_handpose(vis_img, pts_hand, 0, 0)
        
        for i in range(21):
            x = int(pts_hand[str(i)]["x"])
            y = int(pts_hand[str(i)]["y"])
            cv2.circle(vis_img, (x, y), 3, (255, 50, 60), -1)
            cv2.circle(vis_img, (x, y), 1, (255, 150, 180), -1)
        
        return vis_img


def main():
    print('='*60)
    print('手部关键点检测 - 批量测试')
    print('='*60)
    
    model_path = 'D:\\pretrain\\ReXNetV1-size-256-wingloss102-0.122.pth'
    input_dir = './image/'
    output_dir = './results/'
    
    os.makedirs(output_dir, exist_ok=True)
    
    detector = HandPoseDetector(model_path)
    
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    print(f'\n找到 {len(image_files)} 张图片')
    print('开始处理...\n')
    
    for idx, img_file in enumerate(image_files, 1):
        img_path = os.path.join(input_dir, img_file)
        print(f'[{idx}/{len(image_files)}] 处理: {img_file}')
        
        img = cv2.imread(img_path)
        if img is None:
            print(f'  无法读取: {img_file}')
            continue
        
        pts = detector.predict(img)
        vis_img = detector.visualize(img, pts)
        
        output_path = os.path.join(output_dir, f'result_{img_file}')
        cv2.imwrite(output_path, vis_img)
        print(f'  保存到: {output_path}')
    
    print('\n' + '='*60)
    print(f'处理完成! 结果保存在: {output_dir}')
    print('='*60)


if __name__ == '__main__':
    main()
