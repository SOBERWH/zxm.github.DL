# -*-coding:utf-8-*-
# date:2024-01-15
# Author: Eric.Lee & 改进
# function: 手部关键点检测 - 支持IOU跟踪和双手食指指定边界框
#

import os
import cv2
import torch
import numpy as np
from collections import deque
from models.resnet import resnet18, resnet34, resnet50, resnet101
from models.squeezenet import squeezenet1_1, squeezenet1_0
from models.shufflenetv2 import ShuffleNetV2
from models.shufflenet import ShuffleNet
from models.mobilenetv2 import MobileNetV2
from torchvision.models import shufflenet_v2_x1_5, shufflenet_v2_x1_0, shufflenet_v2_x2_0
from models.rexnetv1 import ReXNetV1
from hand_data_iter.datasets import draw_bd_handpose

MEDIAPIPE_AVAILABLE = False
MEDIAPIPE_NEW_API = False
try:
    import mediapipe as mp
    if hasattr(mp, 'solutions'):
        MEDIAPIPE_AVAILABLE = True
    else:
        try:
            from mediapipe import solutions
            MEDIAPIPE_AVAILABLE = True
        except ImportError:
            try:
                from mediapipe.tasks.python.vision import HandLandmarker
                MEDIAPIPE_AVAILABLE = True
                MEDIAPIPE_NEW_API = True
            except ImportError:
                pass
except ImportError:
    pass


def compute_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    inter_area = inter_width * inter_height

    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area


class HandTracker:
    def __init__(self, max_hands=2, iou_threshold=0.3, history_size=15):
        self.max_hands = max_hands
        self.iou_threshold = iou_threshold
        self.history_size = history_size
        self.tracked_hands = {}
        self.next_id = 0

        self.position_history = {}
        self.stability_buffer = {}

    def update(self, hand_bboxes, hand_keypoints=None):
        current_ids = set()
        matched_ids = set()

        if len(hand_bboxes) == 0:
            for hand_id in list(self.tracked_hands.keys()):
                self.tracked_hands[hand_id]['missed_frames'] += 1
                if self.tracked_hands[hand_id]['missed_frames'] > 5:
                    del self.tracked_hands[hand_id]
                    if hand_id in self.position_history:
                        del self.position_history[hand_id]
                    if hand_id in self.stability_buffer:
                        del self.stability_buffer[hand_id]
            return {}

        for bbox in hand_bboxes:
            best_match_id = None
            best_iou = self.iou_threshold

            for hand_id, tracked_data in self.tracked_hands.items():
                if hand_id in matched_ids:
                    continue
                tracked_bbox = tracked_data['bbox']
                iou = compute_iou(bbox, tracked_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_match_id = hand_id

            if best_match_id is not None:
                matched_ids.add(best_match_id)
                self.tracked_hands[best_match_id]['bbox'] = bbox
                self.tracked_hands[best_match_id]['missed_frames'] = 0

                if hand_keypoints is not None:
                    self.tracked_hands[best_match_id]['keypoints'] = hand_keypoints
                current_ids.add(best_match_id)
            else:
                new_id = self.next_id
                self.next_id += 1
                self.tracked_hands[new_id] = {
                    'bbox': bbox,
                    'keypoints': hand_keypoints if hand_keypoints else {},
                    'missed_frames': 0
                }
                self.position_history[new_id] = deque(maxlen=self.history_size)
                self.stability_buffer[new_id] = deque(maxlen=self.history_size)
                current_ids.add(new_id)

        for hand_id in list(self.tracked_hands.keys()):
            if hand_id not in current_ids:
                self.tracked_hands[hand_id]['missed_frames'] += 1
                if self.tracked_hands[hand_id]['missed_frames'] > 5:
                    del self.tracked_hands[hand_id]
                    if hand_id in self.position_history:
                        del self.position_history[hand_id]
                    if hand_id in self.stability_buffer:
                        del self.stability_buffer[hand_id]

        return self.tracked_hands

    def get_stable_position(self, hand_id, threshold=15.0, min_stable_frames=10):
        if hand_id not in self.position_history:
            return None, False

        history = list(self.position_history[hand_id])
        if len(history) < min_stable_frames:
            return None, False

        recent = history[-min_stable_frames:]
        avg_x = np.mean([p[0] for p in recent])
        avg_y = np.mean([p[1] for p in recent])

        deviations = [np.sqrt((p[0] - avg_x)**2 + (p[1] - avg_y)**2) for p in recent]
        max_deviation = max(deviations)

        is_stable = max_deviation < threshold
        return (avg_x, avg_y), is_stable

    def update_position_history(self):
        for hand_id, tracked_data in self.tracked_hands.items():
            bbox = tracked_data['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2

            if hand_id not in self.position_history:
                self.position_history[hand_id] = deque(maxlen=self.history_size)
            self.position_history[hand_id].append((center_x, center_y))


class IndexFingerTracker:
    def __init__(self, stability_threshold=15.0, min_stable_frames=10):
        self.stability_threshold = stability_threshold
        self.min_stable_frames = min_stable_frames
        self.position_history = {}

    def update(self, tracked_hands):
        for hand_id, tracked_data in tracked_hands.items():
            keypoints = tracked_data.get('keypoints', {})
            if '8' in keypoints:
                finger_x = keypoints['8']['x']
                finger_y = keypoints['8']['y']

                if hand_id not in self.position_history:
                    self.position_history[hand_id] = deque(maxlen=30)

                self.position_history[hand_id].append((finger_x, finger_y))

    def get_finger_position(self, hand_id):
        if hand_id not in self.position_history:
            return None, False

        history = list(self.position_history[hand_id])
        if len(history) < self.min_stable_frames:
            return None, False

        recent = history[-self.min_stable_frames:]
        avg_x = np.mean([p[0] for p in recent])
        avg_y = np.mean([p[1] for p in recent])

        deviations = [np.sqrt((p[0] - avg_x)**2 + (p[1] - avg_y)**2) for p in recent]
        max_deviation = max(deviations)

        is_stable = max_deviation < self.stability_threshold
        return (int(avg_x), int(avg_y)), is_stable

    def get_all_finger_positions(self):
        positions = {}
        for hand_id in self.position_history.keys():
            pos, stable = self.get_finger_position(hand_id)
            if pos is not None:
                positions[hand_id] = {'position': pos, 'stable': stable}
        return positions


class GestureTrigger:
    def __init__(self, stable_distance_threshold=50, trigger_distance_threshold=100):
        self.stable_distance_threshold = stable_distance_threshold
        self.trigger_distance_threshold = trigger_distance_threshold
        self.left_hand_id = None
        self.right_hand_id = None
        self.corner_positions = {'top_left': None, 'bottom_right': None}
        self.corner_stable = {'top_left': False, 'bottom_right': False}
        self.triggered = False
        self.triggered_region = None
        self.trigger_count = 0
        self.last_trigger_time = 0

    def assign_hands_by_position(self, tracked_hands):
        if len(tracked_hands) < 2:
            return

        hand_centers = []
        for hand_id, data in tracked_hands.items():
            bbox = data['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            hand_centers.append((hand_id, center_x))

        hand_centers.sort(key=lambda x: x[1])

        self.left_hand_id = hand_centers[0][0] if len(hand_centers) > 0 else None
        self.right_hand_id = hand_centers[-1][0] if len(hand_centers) > 1 else None

    def update_corners(self, finger_tracker):
        if self.triggered and self.trigger_count > 0:
            return

        finger_positions = finger_tracker.get_all_finger_positions()

        if self.left_hand_id and self.left_hand_id in finger_positions:
            self.corner_positions['top_left'] = finger_positions[self.left_hand_id]['position']
            self.corner_stable['top_left'] = finger_positions[self.left_hand_id]['stable']

        if self.right_hand_id and self.right_hand_id in finger_positions:
            self.corner_positions['bottom_right'] = finger_positions[self.right_hand_id]['position']
            self.corner_stable['bottom_right'] = finger_positions[self.right_hand_id]['stable']

    def check_trigger(self):
        if self.corner_positions['top_left'] is None or self.corner_positions['bottom_right'] is None:
            return False

        if not (self.corner_stable['top_left'] and self.corner_stable['bottom_right']):
            return False

        tl = self.corner_positions['top_left']
        br = self.corner_positions['bottom_right']

        distance = np.sqrt((br[0] - tl[0])**2 + (br[1] - tl[1])**2)

        if distance >= self.trigger_distance_threshold:
            self.triggered = True
            self.triggered_region = (
                max(0, tl[0]),
                max(0, tl[1]),
                br[0],
                br[1]
            )
            self.trigger_count += 1
            import time
            self.last_trigger_time = time.time()
            return True

        return False

    def reset_trigger(self):
        self.triggered = False
        self.triggered_region = None

    def get_triggered_region(self):
        if self.triggered and self.triggered_region is not None:
            x1, y1, x2, y2 = self.triggered_region
            if x2 > x1 and y2 > y1 and (x2 - x1) > 20 and (y2 - y1) > 20:
                return self.triggered_region
        return None


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

    def predict(self, img, hand_bboxes=None):
        if hand_bboxes is None or len(hand_bboxes) == 0:
            return []

        results = []
        for hand_bbox in hand_bboxes:
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

            results.append({
                "bbox": hand_bbox,
                "keypoints": pts_hand
            })

        return results

    def predict_in_region(self, img, region_bbox):
        x1, y1, x2, y2 = region_bbox
        x1 = int(np.clip(x1, 0, img.shape[1] - 1))
        y1 = int(np.clip(y1, 0, img.shape[0] - 1))
        x2 = int(np.clip(x2, 0, img.shape[1] - 1))
        y2 = int(np.clip(y2, 0, img.shape[0] - 1))

        if x2 <= x1 or y2 <= y1:
            return []

        region_img = img[y1:y2, x1:x2, :]
        img_ = cv2.resize(region_img, self.img_size, interpolation=cv2.INTER_CUBIC)
        img_ = img_.astype(np.float32)
        img_ = (img_ - 128.) / 256.
        img_ = img_.transpose(2, 0, 1)
        img_ = torch.from_numpy(img_).unsqueeze_(0).to(self.device)

        with torch.no_grad():
            output = self.model(img_.float())
            output = output.cpu().detach().numpy()
            output = np.squeeze(output)

        pts_hand = {}
        w, h = x2 - x1, y2 - y1

        for i in range(int(output.shape[0] / 2)):
            x = output[i * 2 + 0] * float(w) + x1
            y = output[i * 2 + 1] * float(h) + y1
            pts_hand[str(i)] = {"x": x, "y": y}

        return [{
            "bbox": [x1, y1, x2, y2],
            "keypoints": pts_hand
        }]

    def visualize(self, img, results, fps=None, triggered_region=None):
        vis_img = img.copy()

        for result in results:
            pts_hand = result["keypoints"]
            hand_bbox = result["bbox"]

            if hand_bbox is not None:
                cv2.rectangle(vis_img, (int(hand_bbox[0]), int(hand_bbox[1])),
                             (int(hand_bbox[2]), int(hand_bbox[3])), (0, 255, 0), 2)

            draw_bd_handpose(vis_img, pts_hand, 0, 0)

            for i in range(21):
                x = int(pts_hand[str(i)]["x"])
                y = int(pts_hand[str(i)]["y"])
                cv2.circle(vis_img, (x, y), 3, (255, 50, 60), -1)
                cv2.circle(vis_img, (x, y), 1, (255, 150, 180), -1)

        if triggered_region is not None:
            x1, y1, x2, y2 = triggered_region
            cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
            cv2.putText(vis_img, 'TRIGGER REGION', (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if fps:
            cv2.putText(vis_img, f'FPS: {fps:.1f}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        hand_count = len(results)
        cv2.putText(vis_img, f'Hands: {hand_count}', (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return vis_img


class MediaPipeHandDetector:
    def __init__(self, max_num_hands=2, min_detection_confidence=0.5, model_path=None):
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe 不可用，请使用皮肤检测器 (--detector skin)")

        if MEDIAPIPE_NEW_API:
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

            if model_path is None:
                model_path = './hand_landmarker.task'

            if not os.path.exists(model_path):
                print(f"警告: MediaPipe 模型文件不存在: {model_path}")
                print("将使用皮肤检测器作为后备方案")
                raise FileNotFoundError(f"MediaPipe 模型文件不存在: {model_path}")

            base_options = BaseOptions(model_asset_path=model_path)
            options = HandLandmarkerOptions(
                base_options=base_options,
                running_mode=RunningMode.IMAGE,
                num_hands=max_num_hands,
                min_hand_detection_confidence=min_detection_confidence,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.detector = HandLandmarker.create_from_options(options)
            self.use_new_api = True
        else:
            from mediapipe import solutions
            self.mp_hands = solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=0.5
            )
            self.use_new_api = False

        print(f"MediaPipe HandDetector 初始化成功 (最大手数: {max_num_hands}, 新API: {MEDIAPIPE_NEW_API})")

    def detect(self, img):
        if self.use_new_api:
            from mediapipe.tasks.python.vision.core import image as mp_image
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image_obj = mp_image.Image(image_format=mp_image.ImageFormat.SRGB, data=img_rgb)
            results = self.detector.detect(mp_image_obj)

            bboxes = []
            all_keypoints = []

            if results.hand_landmarks:
                h, w = img.shape[:2]
                for hand_landmarks in results.hand_landmarks:
                    x_coords = [landmark.x for landmark in hand_landmarks]
                    y_coords = [landmark.y for landmark in hand_landmarks]

                    x_min = int(min(x_coords) * w)
                    x_max = int(max(x_coords) * w)
                    y_min = int(min(y_coords) * h)
                    y_max = int(max(y_coords) * h)

                    margin = 20
                    x_min = max(0, x_min - margin)
                    y_min = max(0, y_min - margin)
                    x_max = min(w, x_max + margin)
                    y_max = min(h, y_max + margin)

                    bboxes.append([x_min, y_min, x_max, y_max])

                    keypoints = {}
                    for i, landmark in enumerate(hand_landmarks):
                        keypoints[str(i)] = {
                            "x": int(landmark.x * w),
                            "y": int(landmark.y * h)
                        }
                    all_keypoints.append(keypoints)

            return bboxes, all_keypoints
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)

            bboxes = []
            all_keypoints = []

            if results.multi_hand_landmarks:
                h, w = img.shape[:2]
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    x_coords = [landmark.x for landmark in hand_landmarks.landmark]
                    y_coords = [landmark.y for landmark in hand_landmarks.landmark]

                    x_min = int(min(x_coords) * w)
                    x_max = int(max(x_coords) * w)
                    y_min = int(min(y_coords) * h)
                    y_max = int(max(y_coords) * h)

                    margin = 20
                    x_min = max(0, x_min - margin)
                    y_min = max(0, y_min - margin)
                    x_max = min(w, x_max + margin)
                    y_max = min(h, y_max + margin)

                    bboxes.append([x_min, y_min, x_max, y_max])

                    keypoints = {}
                    for i, landmark in enumerate(hand_landmarks.landmark):
                        keypoints[str(i)] = {
                            "x": int(landmark.x * w),
                            "y": int(landmark.y * h)
                        }
                    all_keypoints.append(keypoints)

            return bboxes, all_keypoints


class SimpleHandDetector:
    def __init__(self, max_hands=2):
        self.max_hands = max_hands

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

        bboxes = []
        if len(contours) > 0:
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 2000:
                    x, y, w, h = cv2.boundingRect(contour)
                    bboxes.append([x, y, x + w, y + h])

            bboxes = sorted(bboxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
            bboxes = bboxes[:self.max_hands]

        return bboxes, [{} for _ in bboxes]


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description='手势触发区域检测')
    parser.add_argument('--model_path', type=str, default='D:\\pretrain\\ReXNetV1-size-256-wingloss102-0.122.pth', help='模型路径')
    parser.add_argument('--model', type=str, default='ReXNetV1', help='模型类型')
    parser.add_argument('--mode', type=str, default='webcam', choices=['webcam', 'image', 'video'], help='模式')
    parser.add_argument('--input', type=str, default='./image/', help='输入路径')
    parser.add_argument('--detector', type=str, default='skin', choices=['mediapipe', 'skin'], help='检测器')
    parser.add_argument('--camera_id', type=int, default=0, help='摄像头ID')
    parser.add_argument('--max_hands', type=int, default=2, help='最大手数')

    args = parser.parse_args()

    print('='*60)
    print('手势触发区域检测系统')
    print('='*60)
    print(f'模式: {args.mode}')
    print(f'说明: 左手食指=左上角, 右手食指=右下角')
    print('='*60)

    detector = HandPoseDetector(model_path=args.model_path, model_type=args.model)

    if args.detector == 'mediapipe' and MEDIAPIPE_AVAILABLE:
        hand_detector = MediaPipeHandDetector(max_num_hands=args.max_hands)
    else:
        hand_detector = SimpleHandDetector(max_hands=args.max_hands)

    hand_tracker = HandTracker(max_hands=args.max_hands)
    finger_tracker = IndexFingerTracker(stability_threshold=15.0, min_stable_frames=10)
    gesture_trigger = GestureTrigger(stable_distance_threshold=50, trigger_distance_threshold=100)

    if args.mode == 'webcam':
        cap = cv2.VideoCapture(args.camera_id)
        if not cap.isOpened():
            print(f"无法打开摄像头: {args.camera_id}")
            exit()

        print("摄像头已打开")
        print("操作说明:")
        print("  1. 同时举起两只手")
        print("  2. 左手食指指定区域左上角")
        print("  3. 右手食指指定区域右下角")
        print("  4. 保持手指稳定直到触发")
        print("  5. 按 'r' 重置触发区域")
        print("  6. 按 'q' 退出")
        print()

        prev_time = 0
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            hand_bboxes, _ = hand_detector.detect(frame)
            
            all_results = detector.predict(frame, hand_bboxes)
            
            hand_keypoints_with_bbox = []
            for result in all_results:
                hand_keypoints_with_bbox.append(result['keypoints'])

            tracked_hands = hand_tracker.update(hand_bboxes, hand_keypoints_with_bbox)
            hand_tracker.update_position_history()

            finger_tracker.update(tracked_hands)

            if len(tracked_hands) >= 2:
                gesture_trigger.assign_hands_by_position(tracked_hands)

            gesture_trigger.update_corners(finger_tracker)

            triggered = gesture_trigger.check_trigger()
            triggered_region = gesture_trigger.get_triggered_region()

            target_results = []
            if triggered_region:
                region_results = detector.predict_in_region(frame, triggered_region)
                target_results.extend(region_results)

            all_results.extend(target_results)

            current_time = time.time()
            fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
            prev_time = current_time

            vis_frame = detector.visualize(frame, all_results, fps, triggered_region)

            if gesture_trigger.corner_positions['top_left']:
                cv2.circle(vis_frame, gesture_trigger.corner_positions['top_left'], 10, (255, 0, 0), -1)
                cv2.putText(vis_frame, 'TOP-LEFT', gesture_trigger.corner_positions['top_left'],
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            if gesture_trigger.corner_positions['bottom_right']:
                cv2.circle(vis_frame, gesture_trigger.corner_positions['bottom_right'], 10, (0, 0, 255), -1)
                cv2.putText(vis_frame, 'BOTTOM-RIGHT', gesture_trigger.corner_positions['bottom_right'],
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            status_text = "TRIGGERED" if triggered else "WAITING"
            color = (0, 255, 0) if triggered else (0, 255, 255)
            cv2.putText(vis_frame, f'Status: {status_text}', (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if triggered:
                cv2.putText(vis_frame, f'Triggers: {gesture_trigger.trigger_count}', (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('Gesture Trigger Detection', vis_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                gesture_trigger.reset_trigger()
                print("触发区域已重置")

        cap.release()
        cv2.destroyAllWindows()

    print('\n检测完成!')