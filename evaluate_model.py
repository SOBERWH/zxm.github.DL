# -*- coding: utf-8 -*-
"""
模型评估脚本 - 计算真实的EPE和PCK指标
"""
import os
import sys
import json
import numpy as np
import cv2
import torch
import time
import glob

# 添加项目路径
sys.path.append(os.path.abspath('.'))

from models.rexnetv1 import ReXNetV1
from models.resnet import resnet50
from models.squeezenet import squeezenet1_1

def calculate_epe(pred_keypoints, gt_keypoints, img_width, img_height):
    """计算端点误差(EPE)"""
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
    """计算PCK@0.2"""
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

def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters()) / 1e6

def evaluate_model(model, test_images, test_jsons, model_name, device='cpu'):
    """评估模型"""
    print(f'\n正在评估 {model_name}...')
    model.eval()
    model.to(device)
    
    epe_list = []
    pck_list = []
    fps_list = []
    
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
        
        # 预处理
        img_ = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
        img_ = img_.astype(np.float32)
        img_ = (img_ - 128.) / 256.
        img_ = img_.transpose(2, 0, 1)
        img_tensor = torch.from_numpy(img_).unsqueeze_(0).to(device)
        
        # 推理计时
        start_time = time.time()
        with torch.no_grad():
            output = model(img_tensor.float()).cpu().detach().numpy().squeeze()
        end_time = time.time()
        
        fps = 1.0 / (end_time - start_time)
        fps_list.append(fps)
        
        epe = calculate_epe(output, gt_keypoints, w, h)
        pck = calculate_pck(output, gt_keypoints, w, h)
        
        epe_list.append(epe)
        pck_list.append(pck)
    
    return {
        'epe_mean': float(np.mean(epe_list)),
        'epe_std': float(np.std(epe_list)),
        'pck_mean': float(np.mean(pck_list)),
        'pck_std': float(np.std(pck_list)),
        'fps_mean': float(np.mean(fps_list)),
        'fps_std': float(np.std(fps_list)),
        'num_samples': len(epe_list),
        'epe_list': epe_list,
        'pck_list': pck_list
    }

def main():
    # 测试数据路径
    data_path = 'D:/data1/handpose_datasets_v1-2021-01-31/handpose_datasets_v1/'
    test_images = glob.glob(os.path.join(data_path, '*.jpg'))[:50]
    test_jsons = [img.replace('.jpg', '.json') for img in test_images]
    
    print(f'测试图像数量: {len(test_images)}')
    
    # 模型配置
    model_configs = [
        {
            'name': 'ReXNetV1',
            'model_class': ReXNetV1,
            'model_args': {'num_classes': 42},
            'checkpoint': 'ReXNetV1-size-256-wingloss102-0.122.pth',
            'loss_type': 'Wing Loss'
        },
        {
            'name': 'ResNet-50',
            'model_class': resnet50,
            'model_args': {'num_classes': 42, 'img_size': 256},
            'checkpoint': 'resnet_50-size-256-wingloss102-0.119.pth',
            'loss_type': 'Wing Loss'
        },
        {
            'name': 'SqueezeNet',
            'model_class': squeezenet1_1,
            'model_args': {'num_classes': 42},
            'checkpoint': 'squeezenet1_1-size-256-loss-0.0732.pth',
            'loss_type': 'MSE'
        }
    ]
    
    # 评估所有模型
    results = {}
    for config in model_configs:
        try:
            model = config['model_class'](**config['model_args'])
            params = count_parameters(model)
            
            checkpoint = torch.load(config['checkpoint'], map_location='cpu', weights_only=True)
            model.load_state_dict(checkpoint)
            
            metrics = evaluate_model(model, test_images, test_jsons, config['name'])
            metrics['params'] = float(params)
            metrics['loss_type'] = config['loss_type']
            metrics['checkpoint'] = config['checkpoint']
            
            results[config['name']] = metrics
            print(f'{config["name"]} 评估完成')
        except Exception as e:
            print(f'{config["name"]} 评估失败: {e}')
            results[config['name']] = {'error': str(e)}
    
    # 保存结果
    output_file = 'model_evaluation_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f'\n评估结果已保存到 {output_file}')
    
    # 打印摘要
    print('\n=== 评估结果摘要 ===')
    for name, metrics in results.items():
        if 'error' in metrics:
            print(f'{name}: {metrics["error"]}')
        else:
            print(f'{name}:')
            print(f'  EPE: {metrics["epe_mean"]:.2f}±{metrics["epe_std"]:.2f} px')
            print(f'  PCK@0.2: {metrics["pck_mean"]*100:.2f}±{metrics["pck_std"]*100:.2f}%')
            print(f'  FPS: {metrics["fps_mean"]:.1f}±{metrics["fps_std"]:.1f}')
            print(f'  参数量: {metrics["params"]:.2f}M')

if __name__ == '__main__':
    main()
