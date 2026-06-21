# -*- coding: utf-8 -*-
"""
快速评估脚本 - 获取真实测试指标
"""
import os
import sys
import json
import numpy as np
import cv2
import torch
import time
import glob

sys.path.append(os.path.abspath('.'))

from models.rexnetv1 import ReXNetV1

def calculate_epe(pred_keypoints, gt_keypoints, img_width, img_height):
    epe_sum = 0.0
    count = 0
    for i in range(21):
        if str(i) in gt_keypoints:
            pred_x = pred_keypoints[i * 2] * img_width
            pred_y = pred_keypoints[i * 2 + 1] * img_height
            gt_x = gt_keypoints[str(i)]['x']
            gt_y = gt_keypoints[str(i)]['y']
            epe_sum += np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
            count += 1
    return epe_sum / count if count > 0 else 0

def calculate_pck(pred_keypoints, gt_keypoints, img_width, img_height, threshold=0.2):
    correct = 0
    total = 0
    for i in range(21):
        if str(i) in gt_keypoints:
            pred_x = pred_keypoints[i * 2] * img_width
            pred_y = pred_keypoints[i * 2 + 1] * img_height
            gt_x = gt_keypoints[str(i)]['x']
            gt_y = gt_keypoints[str(i)]['y']
            distance = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
            if distance < threshold * max(img_width, img_height):
                correct += 1
            total += 1
    return correct / total if total > 0 else 0

def main():
    # 只评估ReXNetV1，使用少量测试样本
    data_path = 'D:/data1/handpose_datasets_v1-2021-01-31/handpose_datasets_v1/'
    test_images = glob.glob(os.path.join(data_path, '*.jpg'))[:20]
    test_jsons = [img.replace('.jpg', '.json') for img in test_images]
    
    print(f'测试样本数: {len(test_images)}')
    
    # 加载模型
    model = ReXNetV1(num_classes=42)
    checkpoint = torch.load('ReXNetV1-size-256-wingloss102-0.122.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()
    
    epe_list = []
    pck_list = []
    
    for img_path, json_path in zip(test_images, test_jsons):
        if not os.path.exists(json_path):
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        h, w = img.shape[:2]
        
        with open(json_path, 'r', encoding='utf-8') as f:
            label = json.load(f)
        
        if 'info' not in label or len(label['info']) == 0:
            continue
        
        gt_keypoints = label['info'][0]['pts']
        
        img_ = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
        img_ = img_.astype(np.float32)
        img_ = (img_ - 128.) / 256.
        img_ = img_.transpose(2, 0, 1)
        img_tensor = torch.from_numpy(img_).unsqueeze_(0)
        
        with torch.no_grad():
            output = model(img_tensor.float()).cpu().detach().numpy().squeeze()
        
        epe = calculate_epe(output, gt_keypoints, w, h)
        pck = calculate_pck(output, gt_keypoints, w, h)
        
        epe_list.append(epe)
        pck_list.append(pck)
    
    results = {
        'epe_mean': float(np.mean(epe_list)),
        'epe_std': float(np.std(epe_list)),
        'pck_mean': float(np.mean(pck_list)),
        'pck_std': float(np.std(pck_list)),
        'num_samples': len(epe_list),
        'epe_list': epe_list,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('eval_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f'\n评估完成:')
    print(f'EPE: {results["epe_mean"]:.2f}±{results["epe_std"]:.2f} px')
    print(f'PCK@0.2: {results["pck_mean"]*100:.2f}±{results["pck_std"]*100:.2f}%')

if __name__ == '__main__':
    main()
