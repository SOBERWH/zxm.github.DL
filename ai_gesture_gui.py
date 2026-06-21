# -*-coding:utf-8-*-
# date:2024-01-15
# Author: AI Assistant
# function: AI手势交互GUI应用 - 支持手势框选识图、涂鸦、拖拽等功能
#

import os
import sys
import cv2
import torch
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtMultimedia import *
import time
import threading
import json
import urllib.request
import urllib.parse
import base64

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.rexnetv1 import ReXNetV1
from hand_data_iter.datasets import draw_bd_handpose
from gesture_recognizer import HandGestureRecognizer, GestureType

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


class HandPoseDetector:
    def __init__(self, model_path, model_type='ReXNetV1', img_size=(256, 256), device='cuda'):
        self.img_size = img_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
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


class MediaPipeHandDetector:
    def __init__(self, max_num_hands=2, min_detection_confidence=0.5, model_path=None):
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe 不可用")

        if MEDIAPIPE_NEW_API:
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

            if model_path is None:
                model_path = './hand_landmarker.task'

            if not os.path.exists(model_path):
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

        print(f"MediaPipe HandDetector 初始化成功 (最大手数: {max_num_hands})")

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
                        keypoints[str(i)] = {"x": landmark.x * w, "y": landmark.y * h}
                    all_keypoints.append(keypoints)

            return bboxes, all_keypoints
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)

            bboxes = []
            all_keypoints = []

            if results.multi_hand_landmarks:
                h, w = img.shape[:2]
                for hand_landmarks in results.multi_hand_landmarks:
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
                        keypoints[str(i)] = {"x": landmark.x * w, "y": landmark.y * h}
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

        return bboxes, []


class ImageRecognitionAI:
    def __init__(self, api_key=None, api_type="baidu"):
        self.api_key = api_key
        self.api_type = api_type
        self.cache = {}

    def recognize(self, img, bbox=None):
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.shape[1], x2)
            y2 = min(img.shape[0], y2)
            crop_img = img[y1:y2, x1:x2]
        else:
            crop_img = img

        if self.api_type == "baidu":
            return self._baidu_recognize(crop_img)
        elif self.api_type == "local":
            return self._local_recognize(crop_img)
        else:
            return "未配置AI识别服务"

    def _baidu_recognize(self, img):
        try:
            _, buffer = cv2.imencode('.jpg', img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')

            return f"图像识别结果 (模拟)\n检测到物体，置信度: 0.95\n建议使用百度AI API获取真实结果"
        except Exception as e:
            return f"识别失败: {str(e)}"

    def _local_recognize(self, img):
        h, w = img.shape[:2]
        avg_color = np.mean(img, axis=(0, 1))
        return f"本地分析结果:\n尺寸: {w}x{h}\n平均颜色: BGR({avg_color[0]:.0f}, {avg_color[1]:.0f}, {avg_color[2]:.0f})"


class DrawingCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None  # 视频帧背景
        self.drawing_layer = None  # 涂鸦层
        self.drawing = False
        self.last_point = QPoint()
        self.pen_color = QColor(255, 0, 0)
        self.pen_width = 3
        self.setMinimumSize(640, 480)
        self.setStyleSheet("background-color: white;")

    def setImage(self, image):
        self.image = image.copy()
        # 如果涂鸦层不存在或尺寸不匹配，创建新的涂鸦层
        if self.drawing_layer is None or \
           self.drawing_layer.shape[:2] != image.shape[:2]:
            self.drawing_layer = np.zeros_like(image)
        self.update()

    def getImage(self):
        if self.image is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        return self.image.copy()

    def clear(self):
        if self.drawing_layer is not None:
            self.drawing_layer = np.zeros_like(self.drawing_layer)
        self.update()

    def drawPoint(self, x, y):
        if self.drawing_layer is not None:
            cv2.circle(self.drawing_layer, (int(x), int(y)), self.pen_width, 
                      (self.pen_color.blue(), self.pen_color.green(), self.pen_color.red()), -1)
            self.update()

    def drawLine(self, x1, y1, x2, y2):
        if self.drawing_layer is not None:
            cv2.line(self.drawing_layer, (int(x1), int(y1)), (int(x2), int(y2)),
                    (self.pen_color.blue(), self.pen_color.green(), self.pen_color.red()), self.pen_width)
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        if self.image is not None:
            h, w, ch = self.image.shape
            bytes_per_line = ch * w
            q_img = QImage(self.image.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            painter.drawImage(0, 0, q_img.scaled(self.size(), Qt.KeepAspectRatio))
            
            # 绘制涂鸦层
            if self.drawing_layer is not None:
                # 创建带透明度的涂鸦层
                alpha = 0.8  # 透明度
                combined = self.image.copy()
                mask = self.drawing_layer > 0
                combined[mask] = (alpha * self.drawing_layer[mask] + (1 - alpha) * combined[mask]).astype(np.uint8)
                q_img_combined = QImage(combined.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
                painter.drawImage(0, 0, q_img_combined.scaled(self.size(), Qt.KeepAspectRatio))


class VideoPlayer(QThread):
    """视频播放线程"""
    frame_ready = pyqtSignal(np.ndarray, list, list, list)

    def __init__(self, video_path, hand_detector, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.hand_detector = hand_detector
        self.running = False
        self.cap = None

    def run(self):
        self.running = True
        self.cap = cv2.VideoCapture(self.video_path)

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            hand_bboxes, all_keypoints = self.hand_detector.detect(frame)

            self.frame_ready.emit(frame, hand_bboxes, all_keypoints, [])

            QThread.msleep(33)

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()


class VideoThread(QThread):
    frame_ready = pyqtSignal(np.ndarray, list, list, list)
    gesture_detected = pyqtSignal(object, float)
    trigger_region = pyqtSignal(list, list)

    def __init__(self, hand_detector, keypoint_detector, gesture_recognizer, parent=None):
        super().__init__(parent)
        self.hand_detector = hand_detector
        self.keypoint_detector = keypoint_detector
        self.gesture_recognizer = gesture_recognizer
        self.running = False
        self.cap = None

        self.trigger_mode = False
        self.left_index_stable = None
        self.right_index_stable = None
        self.stability_threshold = 15
        self.stability_frames = 0
        self.required_stability_frames = 10

    def run(self):
        self.running = True
        self.cap = cv2.VideoCapture(0)

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            try:
                hand_bboxes, all_keypoints = self.hand_detector.detect(frame)

                if len(all_keypoints) == 0 and len(hand_bboxes) > 0:
                    results = self.keypoint_detector.predict(frame, hand_bboxes)
                    all_keypoints = [r['keypoints'] for r in results]

                gesture_labels = []
                for i, keypoints in enumerate(all_keypoints):
                    result = self.gesture_recognizer.update(keypoints)
                    if isinstance(result, tuple):
                        gesture, confidence = result
                    else:
                        gesture = result
                        confidence = 0.0

                    gesture_names_short = {
                        GestureType.FIST: "FIST", GestureType.CATCH: "CATCH",
                        GestureType.OK: "OK", GestureType.FIVE: "FIVE",
                        GestureType.GUN: "GUN", GestureType.LOVE: "LOVE",
                        GestureType.ONE: "ONE", GestureType.SIX: "SIX",
                        GestureType.THREE: "THREE", GestureType.THUMBUP: "THUMBUP",
                        GestureType.YEAH: "YEAH"
                    }
                    gesture_label = gesture_names_short.get(gesture, None)
                    gesture_labels.append((gesture_label, confidence) if gesture_label else (None, 0.0))

                    if gesture != GestureType.NONE:
                        self.gesture_detected.emit(gesture, confidence)

                if self.trigger_mode and len(all_keypoints) >= 2:
                    self._check_trigger_region(frame, all_keypoints)

                self.frame_ready.emit(frame, hand_bboxes, all_keypoints, gesture_labels)

            except Exception as e:
                print(f"处理错误: {e}")
                self.frame_ready.emit(frame, [], [], [])

        if self.cap:
            self.cap.release()

    def _check_trigger_region(self, frame, all_keypoints):
        if len(all_keypoints) < 2:
            return

        left_index = None
        right_index = None

        for keypoints in all_keypoints:
            if '8' in keypoints:
                index_tip = keypoints['8']
                x, y = index_tip['x'], index_tip['y']

                frame_center_x = frame.shape[1] / 2
                if x < frame_center_x:
                    left_index = (x, y)
                else:
                    right_index = (x, y)

        if left_index and right_index:
            if self.left_index_stable is None:
                self.left_index_stable = left_index
                self.right_index_stable = right_index
                self.stability_frames = 0
            else:
                left_dist = np.sqrt((left_index[0] - self.left_index_stable[0])**2 + 
                                   (left_index[1] - self.left_index_stable[1])**2)
                right_dist = np.sqrt((right_index[0] - self.right_index_stable[0])**2 + 
                                    (right_index[1] - self.right_index_stable[1])**2)

                if left_dist < self.stability_threshold and right_dist < self.stability_threshold:
                    self.stability_frames += 1
                    if self.stability_frames >= self.required_stability_frames:
                        self.trigger_region.emit(list(left_index), list(right_index))
                        self.stability_frames = 0
                else:
                    self.left_index_stable = left_index
                    self.right_index_stable = right_index
                    self.stability_frames = 0

    def stop(self):
        self.running = False
        self.wait()


class AIGestureGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("手势交互系统")
        self.setGeometry(100, 100, 1400, 800)

        self.hand_detector = None
        self.keypoint_detector = None
        self.gesture_recognizer = None
        self.ai_recognizer = ImageRecognitionAI(api_type="local")
        self.video_thread = None

        self.current_mode = "detect"
        self.drawing_points = []
        self.trigger_region = None
        self.last_recognition_result = ""
        self.current_gesture_confidence = 0.0
        
        # 涂鸦模式颜色按钮配置（五个常见色）
        self.colors = [
            {'name': '红色', 'bgr': (0, 0, 255), 'pos': (20, 50)},
            {'name': '绿色', 'bgr': (0, 255, 0), 'pos': (20, 100)},
            {'name': '蓝色', 'bgr': (255, 0, 0), 'pos': (20, 150)},
            {'name': '黄色', 'bgr': (0, 255, 255), 'pos': (20, 200)},
            {'name': '黑色', 'bgr': (0, 0, 0), 'pos': (20, 250)},
        ]
        self.color_button_size = 30
        self.current_draw_color = (0, 0, 255)  # 默认红色
        
        # 涂鸦相关状态
        self.drawing_strokes = []  # 存储所有涂鸦笔画
        self.current_stroke = []   # 当前正在绘制的笔画
        self.selected_stroke = None  # 当前选中的笔画
        self.is_drawing = False  # 是否正在涂鸦
        self.last_double_tap_time = 0  # 上次双击时间
        self.move_offset = (0, 0)  # 移动偏移量

        self._init_ui()
        self._init_models()

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        left_panel = QWidget()
        left_panel.setFixedWidth(300)
        left_layout = QVBoxLayout(left_panel)

        mode_group = QGroupBox("功能模式")
        mode_layout = QVBoxLayout(mode_group)

        self.detect_radio = QRadioButton("手势检测")
        self.detect_radio.setChecked(True)
        self.detect_radio.toggled.connect(self._on_mode_changed)
        mode_layout.addWidget(self.detect_radio)

        self.trigger_radio = QRadioButton("框选识图")
        self.trigger_radio.toggled.connect(self._on_mode_changed)
        mode_layout.addWidget(self.trigger_radio)

        self.draw_radio = QRadioButton("涂鸦模式")
        self.draw_radio.toggled.connect(self._on_mode_changed)
        mode_layout.addWidget(self.draw_radio)

        self.drag_radio = QRadioButton("拖拽模式")
        self.drag_radio.toggled.connect(self._on_mode_changed)
        mode_layout.addWidget(self.drag_radio)

        left_layout.addWidget(mode_group)

        control_group = QGroupBox("控制面板")
        control_layout = QVBoxLayout(control_group)

        self.start_btn = QPushButton("启动摄像头")
        self.start_btn.clicked.connect(self._toggle_camera)
        control_layout.addWidget(self.start_btn)

        # 添加图片和视频识别按钮
        file_group = QGroupBox("文件识别")
        file_layout = QVBoxLayout(file_group)
        
        self.image_btn = QPushButton("选择图片")
        self.image_btn.clicked.connect(self._open_image)
        file_layout.addWidget(self.image_btn)
        
        self.video_btn = QPushButton("选择视频")
        self.video_btn.clicked.connect(self._open_video)
        file_layout.addWidget(self.video_btn)
        
        control_layout.addWidget(file_group)

        self.clear_btn = QPushButton("清除画布")
        self.clear_btn.clicked.connect(self._clear_canvas)
        control_layout.addWidget(self.clear_btn)

        self.recognize_btn = QPushButton("识别选中区域")
        self.recognize_btn.clicked.connect(self._recognize_region)
        control_layout.addWidget(self.recognize_btn)

        left_layout.addWidget(control_group)

        color_group = QGroupBox("画笔设置")
        color_layout = QVBoxLayout(color_group)

        color_row = QHBoxLayout()
        self.color_btn = QPushButton("选择颜色")
        self.color_btn.clicked.connect(self._select_color)
        color_row.addWidget(self.color_btn)

        self.color_preview = QLabel()
        self.color_preview.setFixedSize(30, 30)
        self.color_preview.setStyleSheet("background-color: rgb(255, 0, 0);")
        color_row.addWidget(self.color_preview)
        color_row.addStretch()
        color_layout.addLayout(color_row)

        width_row = QHBoxLayout()
        width_row.addWidget(QLabel("画笔粗细:"))
        self.width_slider = QSlider(Qt.Horizontal)
        self.width_slider.setRange(1, 20)
        self.width_slider.setValue(3)
        self.width_slider.valueChanged.connect(self._on_width_changed)
        width_row.addWidget(self.width_slider)
        color_layout.addLayout(width_row)

        left_layout.addWidget(color_group)

        gesture_group = QGroupBox("手势识别状态")
        gesture_layout = QVBoxLayout(gesture_group)

        self.gesture_label = QLabel("当前手势: 无")
        self.gesture_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        gesture_layout.addWidget(self.gesture_label)

        self.gesture_history = QTextEdit()
        self.gesture_history.setReadOnly(True)
        self.gesture_history.setMaximumHeight(100)
        gesture_layout.addWidget(self.gesture_history)

        left_layout.addWidget(gesture_group)

        ai_group = QGroupBox("手势识别结果")
        ai_layout = QVBoxLayout(ai_group)

        self.ai_result = QTextEdit()
        self.ai_result.setReadOnly(True)
        self.ai_result.setMaximumHeight(150)
        ai_layout.addWidget(self.ai_result)

        left_layout.addWidget(ai_group)

        left_layout.addStretch()

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.video_label)

        self.canvas = DrawingCanvas()
        self.canvas.setMinimumSize(640, 480)
        self.canvas.hide()
        right_layout.addWidget(self.canvas)

        info_label = QLabel("操作提示: 双手食指框选区域进行识别 | 涂鸦模式下用食指绘画")
        info_label.setStyleSheet("color: gray; font-size: 12px;")
        right_layout.addWidget(info_label)

        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, 1)

    def _init_models(self):
        try:
            model_path = "D:/pretrain/ReXNetV1-size-256-wingloss102-0.122.pth"
            self.keypoint_detector = HandPoseDetector(model_path)

            if MEDIAPIPE_AVAILABLE:
                try:
                    self.hand_detector = MediaPipeHandDetector(max_num_hands=2)
                except Exception as e:
                    print(f"MediaPipe初始化失败: {e}，使用皮肤检测")
                    self.hand_detector = SimpleHandDetector(max_hands=2)
            else:
                self.hand_detector = SimpleHandDetector(max_hands=2)

            self.gesture_recognizer = HandGestureRecognizer()

            print("模型初始化完成")
        except Exception as e:
            print(f"模型初始化失败: {e}")
            QMessageBox.critical(self, "错误", f"模型初始化失败: {e}")

    def _on_mode_changed(self):
        if self.detect_radio.isChecked():
            self.current_mode = "detect"
            self.canvas.hide()
            self.video_label.show()
        elif self.trigger_radio.isChecked():
            self.current_mode = "trigger"
            self.canvas.hide()
            self.video_label.show()
        elif self.draw_radio.isChecked():
            self.current_mode = "draw"
            self.video_label.hide()
            self.canvas.show()
            self.canvas.clear()
        elif self.drag_radio.isChecked():
            self.current_mode = "drag"
            self.canvas.hide()
            self.video_label.show()

        if self.video_thread:
            self.video_thread.trigger_mode = (self.current_mode == "trigger")

    def _toggle_camera(self):
        if self.video_thread is None or not self.video_thread.isRunning():
            # 停止视频播放器（如果正在播放）
            if hasattr(self, 'video_player') and self.video_player.isRunning():
                self.video_player.stop()
                self.video_player = None
            
            self.video_thread = VideoThread(
                self.hand_detector,
                self.keypoint_detector,
                self.gesture_recognizer
            )
            self.video_thread.frame_ready.connect(self._on_frame_ready)
            self.video_thread.gesture_detected.connect(self._on_gesture_detected)
            self.video_thread.trigger_region.connect(self._on_trigger_region)
            self.video_thread.trigger_mode = (self.current_mode == "trigger")
            self.video_thread.start()
            self.start_btn.setText("停止摄像头")
        else:
            self.video_thread.stop()
            self.video_thread = None
            self.start_btn.setText("启动摄像头")

    def _on_frame_ready(self, frame, hand_bboxes, all_keypoints, gesture_labels=None):
        if gesture_labels is None:
            gesture_labels = []

        vis_frame = frame.copy()

        for i, bbox in enumerate(hand_bboxes):
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if i < len(gesture_labels) and gesture_labels[i][0]:
                label, conf = gesture_labels[i]
                cv2.putText(vis_frame, f"{label}", (x1, max(y1 - 10, 20)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        for keypoints in all_keypoints:
            draw_bd_handpose(vis_frame, keypoints, 0, 0)
            for i in range(21):
                if str(i) in keypoints:
                    x = int(keypoints[str(i)]["x"])
                    y = int(keypoints[str(i)]["y"])
                    cv2.circle(vis_frame, (x, y), 3, (255, 50, 60), -1)

        if self.current_mode == "draw":
            # 先设置视频帧作为画布背景
            self.canvas.setImage(vis_frame)

            # 检测手指关键点
            index_tips = []  # 食指指尖列表
            thumb_tips = []  # 拇指指尖列表

            for keypoints in all_keypoints:
                if '8' in keypoints:  # 食指指尖
                    x, y = int(keypoints['8']['x']), int(keypoints['8']['y'])
                    index_tips.append((x, y))
                if '4' in keypoints:  # 拇指指尖
                    x, y = int(keypoints['4']['x']), int(keypoints['4']['y'])
                    thumb_tips.append((x, y))

            # 检测双指捏合（食指和拇指靠近）- 用于涂鸦
            current_time = time.time()
            drawing_point = None
            for i, index_tip in enumerate(index_tips):
                for j, thumb_tip in enumerate(thumb_tips):
                    distance = ((index_tip[0] - thumb_tip[0])**2 +
                               (index_tip[1] - thumb_tip[1])**2)**0.5

                    if distance < 40:  # 捏合阈值
                        self.is_drawing = True
                        drawing_point = index_tip
                        break
            
            # 如果正在捏合，添加连续线条
            if self.is_drawing and drawing_point:
                x, y = drawing_point
                # 如果当前笔画已有点，从最后一个点画线到当前点
                if len(self.current_stroke) > 0:
                    last_x, last_y, _ = self.current_stroke[-1]
                    # 添加线段上的多个点，使线条更连续
                    dx = x - last_x
                    dy = y - last_y
                    steps = max(abs(dx), abs(dy), 1)
                    for step in range(1, steps + 1):
                        interp_x = int(last_x + dx * step / steps)
                        interp_y = int(last_y + dy * step / steps)
                        self.current_stroke.append((interp_x, interp_y, self.current_draw_color))
                else:
                    self.current_stroke.append((x, y, self.current_draw_color))

            # 如果没有捏合，结束当前笔画
            if not self.is_drawing and len(self.current_stroke) > 0:
                if len(self.current_stroke) > 1:  # 至少两个点才保存
                    self.drawing_strokes.append({
                        'points': self.current_stroke.copy(),
                        'color': self.current_draw_color
                    })
                self.current_stroke = []

            self.is_drawing = False  # 重置状态

            # 检测食指双击 - 用于选中涂鸦或切换颜色
            if len(index_tips) == 1:
                index_tip = index_tips[0]

                # 检查是否双击颜色按钮
                if self._check_color_button_double_tap(index_tip, current_time):
                    pass  # 已处理颜色切换

                # 检查是否双击涂鸦区域
                elif self._check_stroke_double_tap(index_tip, current_time):
                    pass  # 已处理选中

            # 处理选中状态下的移动
            if self.selected_stroke is not None and len(index_tips) == 1:
                index_tip = index_tips[0]
                if not self._is_in_color_button_area(index_tip):
                    self._move_selected_stroke(index_tip)

            # 处理双指缩放
            if self.selected_stroke is not None and len(index_tips) >= 2:
                self._scale_selected_stroke(index_tips[0], index_tips[1])

            # 在canvas上绘制颜色按钮和涂鸦
            self._draw_color_buttons_on_canvas()
        
        elif self.current_mode == "drag":
            # 拖拽模式：显示视频画面和已保存的涂鸦
            self.canvas.setImage(vis_frame)
            
            # 绘制所有已保存的涂鸦
            for stroke in self.drawing_strokes:
                color = stroke['color']
                points = stroke['points']
                for i in range(len(points) - 1):
                    x1, y1, _ = points[i]
                    x2, y2, _ = points[i + 1]
                    cv2.line(vis_frame, (x1, y1), (x2, y2), color, 3)
            
            # 检测食指关键点
            index_tips = []
            for keypoints in all_keypoints:
                if '8' in keypoints:
                    x, y = int(keypoints['8']['x']), int(keypoints['8']['y'])
                    index_tips.append((x, y))
            
            current_time = time.time()
            
            # 双击选中涂鸦
            if len(index_tips) == 1:
                index_tip = index_tips[0]
                self._check_stroke_double_tap(index_tip, current_time)
            
            # 移动选中的涂鸦
            if self.selected_stroke is not None and len(index_tips) == 1:
                index_tip = index_tips[0]
                self._move_selected_stroke(index_tip)
            
            # 双指缩放
            if self.selected_stroke is not None and len(index_tips) >= 2:
                self._scale_selected_stroke(index_tips[0], index_tips[1])
            
            # 更新canvas显示
            self._draw_color_buttons_on_canvas()

        if self.current_mode == "trigger" and len(all_keypoints) >= 2:
            left_index = None
            right_index = None

            for keypoints in all_keypoints:
                if '8' in keypoints:
                    index_tip = keypoints['8']
                    x, y = index_tip['x'], index_tip['y']
                    frame_center_x = vis_frame.shape[1] / 2
                    if x < frame_center_x:
                        left_index = (int(x), int(y))
                    else:
                        right_index = (int(x), int(y))

            if left_index and right_index:
                cv2.circle(vis_frame, left_index, 10, (0, 0, 255), -1)
                cv2.circle(vis_frame, right_index, 10, (255, 0, 0), -1)
                cv2.rectangle(vis_frame, left_index, right_index, (0, 255, 255), 2)

                if self.trigger_region:
                    x1, y1 = self.trigger_region[0]
                    x2, y2 = self.trigger_region[1]
                    cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                    cv2.putText(vis_frame, "Triggered!", (int(x1), int(y1) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        mode_text = {
            "detect": "Detection Mode",
            "trigger": "Trigger Mode (2 hands)",
            "draw": "Drawing Mode",
            "drag": "Drag Mode"
        }
        cv2.putText(vis_frame, mode_text.get(self.current_mode, ""), (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if self.current_gesture_confidence > 0:
            confidence_text = f"Confidence: {self.current_gesture_confidence:.2%}"
            cv2.putText(vis_frame, confidence_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # 在画面右上角显示框选对象
        if hasattr(self, 'show_selected_region') and self.show_selected_region and hasattr(self, 'selected_region_image'):
            region_img = self.selected_region_image
            # 调整大小为150x150的缩略图
            region_h, region_w = region_img.shape[:2]
            scale = min(150/region_w, 150/region_h)
            thumb_w, thumb_h = int(region_w*scale), int(region_h*scale)
            thumbnail = cv2.resize(region_img, (thumb_w, thumb_h))
            
            # 计算右上角位置
            frame_h, frame_w = vis_frame.shape[:2]
            x_offset = frame_w - thumb_w - 10
            y_offset = 10
            
            # 添加背景框
            cv2.rectangle(vis_frame, (x_offset-5, y_offset-5), 
                         (x_offset+thumb_w+5, y_offset+thumb_h+5), (0, 255, 0), 2)
            # 添加标题
            cv2.putText(vis_frame, "Selected", (x_offset, y_offset-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 放置缩略图
            vis_frame[y_offset:y_offset+thumb_h, x_offset:x_offset+thumb_w] = thumbnail

        h, w, ch = vis_frame.shape
        bytes_per_line = ch * w
        q_img = QImage(vis_frame.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.video_label.setPixmap(QPixmap.fromImage(q_img.scaled(
            self.video_label.size(), Qt.KeepAspectRatio)))

        # 涂鸦模式下不再重新设置画布图像，以免覆盖已绘制的内容

    def _on_gesture_detected(self, gesture, confidence=0.0):
        gesture_names = {
            GestureType.TAP: "Tap (点击)",
            GestureType.DOUBLE_TAP: "Double Tap (双击)",
            GestureType.PINCH_OPEN: "Pinch Open (张开)",
            GestureType.PINCH_CLOSE: "Pinch Close (捏合)",
            GestureType.ROTATION_CW: "Rotation CW (顺时针旋转)",
            GestureType.ROTATION_CCW: "Rotation CCW (逆时针旋转)",
            GestureType.SWIPE_LEFT: "Swipe Left (左滑)",
            GestureType.SWIPE_RIGHT: "Swipe Right (右滑)",
            GestureType.SWIPE_UP: "Swipe Up (上滑)",
            GestureType.SWIPE_DOWN: "Swipe Down (下滑)",
            GestureType.PAN: "Pan (拖动)",
            GestureType.LONG_PRESS: "Long Press (长按)",
            GestureType.FIST: "Fist (拳头)",
            GestureType.CATCH: "Catch (抓取)",
            GestureType.OK: "OK (OK手势)",
            GestureType.FIVE: "Five (五指张开)",
            GestureType.GUN: "Gun (手枪)",
            GestureType.LOVE: "Love (爱心)",
            GestureType.ONE: "One (食指)",
            GestureType.SIX: "Six (六)",
            GestureType.THREE: "Three (三)",
            GestureType.THUMBUP: "ThumbUp (拇指向上)",
            GestureType.YEAH: "Yeah (胜利)",
        }

        gesture_name = gesture_names.get(gesture, str(gesture))
        confidence_str = f"{confidence:.2%}"
        self.current_gesture_confidence = confidence
        self.gesture_label.setText(f"当前手势: {gesture_name} | 置信度: {confidence_str}")

        timestamp = time.strftime("%H:%M:%S")
        self.gesture_history.append(f"[{timestamp}] {gesture_name} ({confidence_str})")

        self._handle_gesture_action(gesture)

    def _handle_gesture_action(self, gesture):
        if gesture == GestureType.TAP:
            self._handle_tap()
        elif gesture == GestureType.DOUBLE_TAP:
            self._handle_double_tap()
        elif gesture == GestureType.LONG_PRESS:
            self._handle_long_press()
        elif gesture == GestureType.PINCH_CLOSE:
            self._handle_pinch_close()
        elif gesture == GestureType.PINCH_OPEN:
            self._handle_pinch_open()
        elif gesture == GestureType.SWIPE_UP:
            self._handle_swipe_up()
        elif gesture == GestureType.SWIPE_DOWN:
            self._handle_swipe_down()
        elif gesture == GestureType.SWIPE_LEFT:
            self._handle_swipe_left()
        elif gesture == GestureType.SWIPE_RIGHT:
            self._handle_swipe_right()
        elif gesture == GestureType.PAN:
            self._handle_pan()

    def _handle_tap(self):
        if self.current_mode == "draw":
            self.canvas.clear()
            self.ai_result.setText("画布已清除 (Tap手势)")
        elif self.current_mode == "trigger" and self.trigger_region:
            self._recognize_region()

    def _handle_double_tap(self):
        if self.current_mode == "draw":
            self._select_color()
            self.ai_result.setText("已触发颜色选择 (Double Tap)")
        else:
            self.ai_result.setText("双击手势检测到!")

    def _handle_long_press(self):
        if self.current_mode == "trigger" and self.trigger_region:
            self._recognize_region()
        elif self.current_mode == "draw":
            self.ai_result.setText("长按手势检测到!")

    def _handle_pinch_close(self):
        if self.current_mode == "draw":
            self.canvas.pen_width = max(1, self.canvas.pen_width - 1)
            self.ai_result.setText(f"画笔变细: {self.canvas.pen_width}px (Pinch Close)")

    def _handle_pinch_open(self):
        if self.current_mode == "draw":
            self.canvas.pen_width = min(20, self.canvas.pen_width + 1)
            self.ai_result.setText(f"画笔变粗: {self.canvas.pen_width}px (Pinch Open)")
        else:
            self._clear_canvas()

    def _handle_swipe_up(self):
        self.ai_result.setText("向上滑动 - 可以用于滚动/放大")

    def _handle_swipe_down(self):
        self.ai_result.setText("向下滑动 - 可以用于滚动/缩小")

    def _handle_swipe_left(self):
        self.ai_result.setText("向左滑动 - 返回/撤销")

    def _handle_swipe_right(self):
        self.ai_result.setText("向右滑动 - 前进/重做")

    def _handle_pan(self):
        self.ai_result.setText("拖拽手势 - 可以移动选中对象")

    def _on_trigger_region(self, left_pos, right_pos):
        self.trigger_region = [left_pos, right_pos]
        x1, y1 = left_pos
        x2, y2 = right_pos

        self.ai_result.setText(f"检测到触发区域:\n左上角: ({x1:.0f}, {y1:.0f})\n右下角: ({x2:.0f}, {y2:.0f})\n\n点击'识别选中区域'进行AI识别")

    def _recognize_region(self):
        if self.trigger_region is None:
            self.ai_result.setText("请先用双手食指框选区域")
            return

        if self.video_thread is None and not hasattr(self, 'video_player'):
            self.ai_result.setText("请先启动摄像头或打开视频")
            return

        # 获取当前帧
        frame = None
        if self.video_thread and self.video_thread.cap and self.video_thread.cap.isOpened():
            ret, frame = self.video_thread.cap.read()
        elif hasattr(self, 'video_player') and self.video_player.cap and self.video_player.cap.isOpened():
            ret, frame = self.video_player.cap.read()
        
        if frame is None:
            self.ai_result.setText("无法获取图像")
            return

        x1, y1 = self.trigger_region[0]
        x2, y2 = self.trigger_region[1]

        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        bbox = [x1, y1, x2, y2]
        result = self.ai_recognizer.recognize(frame, bbox)

        # 保存框选区域的图像用于显示
        self.selected_region_image = frame[int(y1):int(y2), int(x1):int(x2)].copy()
        
        self.ai_result.setText(f"AI识别结果:\n{result}")
        self.last_recognition_result = result
        self.show_selected_region = True  # 标记显示框选对象

    def _clear_canvas(self):
        self.canvas.clear()
        self.trigger_region = None
        self.show_selected_region = False  # 隐藏框选对象显示
        if hasattr(self, 'selected_region_image'):
            delattr(self, 'selected_region_image')
        self.ai_result.setText("画布已清除")

    def _select_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.pen_color = color
            self.canvas.pen_color = color
            self.color_preview.setStyleSheet(f"background-color: rgb({color.red()}, {color.green()}, {color.blue()});")

    def _draw_color_buttons(self, frame):
        """在画面左侧绘制颜色按钮"""
        for color in self.colors:
            x, y = color['pos']
            size = self.color_button_size
            
            # 绘制按钮背景
            cv2.rectangle(frame, (x, y), (x + size, y + size), (200, 200, 200), -1)
            
            # 绘制颜色方块
            cv2.rectangle(frame, (x + 2, y + 2), (x + size - 2, y + size - 2), color['bgr'], -1)
            
            # 如果是当前选中的颜色，绘制边框
            if color['bgr'] == self.current_draw_color:
                cv2.rectangle(frame, (x - 2, y - 2), (x + size + 2, y + size + 2), (0, 255, 0), 2)
            
            # 绘制颜色名称
            cv2.putText(frame, color['name'], (x + size + 5, y + size // 2 + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _check_color_button_click(self, point):
        """检测食指是否点击了颜色按钮"""
        x, y = point
        for color in self.colors:
            bx, by = color['pos']
            size = self.color_button_size
            if bx <= x <= bx + size and by <= y <= by + size:
                return color
        return None
    
    def _is_in_color_button_area(self, point):
        """检测点是否在颜色按钮区域"""
        x, y = point
        # 颜色按钮区域范围
        for color in self.colors:
            bx, by = color['pos']
            size = self.color_button_size
            if bx - 10 <= x <= bx + size + 10 and by - 10 <= y <= by + size + 10:
                return True
        return False
    
    def _check_color_button_double_tap(self, point, current_time):
        """检测是否双击颜色按钮"""
        for color in self.colors:
            bx, by = color['pos']
            size = self.color_button_size
            if bx <= point[0] <= bx + size and by <= point[1] <= by + size:
                # 检查是否是双击（500ms内两次点击）
                if current_time - self.last_double_tap_time < 0.5:
                    self.current_draw_color = color['bgr']
                    self.canvas.pen_color = QColor(color['bgr'][2], color['bgr'][1], color['bgr'][0])
                    self.ai_result.setText(f"已选择颜色: {color['name']}")
                    self.last_double_tap_time = 0  # 重置双击计时
                    return True
                else:
                    self.last_double_tap_time = current_time
                    return False
        return False
    
    def _check_stroke_double_tap(self, point, current_time):
        """检测是否双击选中涂鸦"""
        # 检查是否是双击（500ms内两次点击）
        if current_time - self.last_double_tap_time >= 0.5:
            self.last_double_tap_time = current_time
            return False
        
        # 检查点击位置是否在某个笔画上
        for i, stroke in enumerate(self.drawing_strokes):
            for px, py, _ in stroke['points']:
                distance = ((point[0] - px)**2 + (point[1] - py)**2)**0.5
                if distance < 20:
                    self.selected_stroke = i
                    self.move_offset = (px - point[0], py - point[1])
                    self.ai_result.setText(f"已选中第 {i+1} 个涂鸦")
                    self.last_double_tap_time = 0  # 重置双击计时
                    return True
        
        # 如果没有选中任何涂鸦，取消选中状态
        self.selected_stroke = None
        self.last_double_tap_time = 0
        return False

    def _draw_color_buttons_on_canvas(self):
        """在canvas上绘制颜色按钮和所有涂鸦"""
        if self.canvas.image is None:
            return

        # 复制背景图像
        canvas_img = self.canvas.image.copy()

        # 绘制颜色按钮（在左侧）
        for color in self.colors:
            x, y = color['pos']
            size = self.color_button_size

            # 绘制按钮背景（灰色边框）
            cv2.rectangle(canvas_img, (x-2, y-2), (x + size + 2, y + size + 2), (100, 100, 100), -1)

            # 绘制颜色方块
            cv2.rectangle(canvas_img, (x, y), (x + size, y + size), color['bgr'], -1)

            # 如果是当前选中的颜色，绘制白色边框
            if color['bgr'] == self.current_draw_color:
                cv2.rectangle(canvas_img, (x-3, y-3), (x + size + 3, y + size + 3), (255, 255, 255), 2)

        # 绘制所有涂鸦笔画
        for i, stroke in enumerate(self.drawing_strokes):
            color = stroke['color']
            for px, py, _ in stroke['points']:
                cv2.circle(canvas_img, (px, py), 5, color, -1)

            # 如果是选中的笔画，绘制边框
            if self.selected_stroke == i and len(stroke['points']) > 0:
                xs = [p[0] for p in stroke['points']]
                ys = [p[1] for p in stroke['points']]
                cv2.rectangle(canvas_img, (min(xs)-5, min(ys)-5), (max(xs)+5, max(ys)+5), (0, 255, 255), 2)

        # 更新canvas图像
        self.canvas.image = canvas_img
        self.canvas.update()

    def _move_selected_stroke(self, point):
        """移动选中的涂鸦"""
        if self.selected_stroke is None:
            return
        
        stroke = self.drawing_strokes[self.selected_stroke]
        new_points = []
        
        for px, py, color in stroke['points']:
            new_x = px + (point[0] + self.move_offset[0] - px) * 0.3  # 平滑移动
            new_y = py + (point[1] + self.move_offset[1] - py) * 0.3
            new_points.append((int(new_x), int(new_y), color))
        
        stroke['points'] = new_points
        
        # 清除画布并重绘
        self.canvas.clear()
        for s in self.drawing_strokes:
            for px, py, color in s['points']:
                self.canvas.pen_color = QColor(color[2], color[1], color[0])
                self.canvas.drawPoint(px, py)
    
    def _scale_selected_stroke(self, point1, point2):
        """使用双指缩放选中的涂鸦"""
        if self.selected_stroke is None:
            return
        
        stroke = self.drawing_strokes[self.selected_stroke]
        if len(stroke['points']) == 0:
            return
        
        # 计算双指距离
        current_distance = ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5
        
        # 计算笔画中心
        center_x = sum(p[0] for p in stroke['points']) / len(stroke['points'])
        center_y = sum(p[1] for p in stroke['points']) / len(stroke['points'])
        
        # 检查是否有上次距离记录
        if not hasattr(self, 'last_scale_distance'):
            self.last_scale_distance = current_distance
            return
        
        # 计算缩放比例
        scale = current_distance / self.last_scale_distance
        
        # 缩放笔画
        new_points = []
        for px, py, color in stroke['points']:
            dx = px - center_x
            dy = py - center_y
            new_x = center_x + dx * scale
            new_y = center_y + dy * scale
            new_points.append((int(new_x), int(new_y), color))
        
        stroke['points'] = new_points
        self.last_scale_distance = current_distance
        
        # 清除画布并重绘
        self.canvas.clear()
        for s in self.drawing_strokes:
            for px, py, color in s['points']:
                self.canvas.pen_color = QColor(color[2], color[1], color[0])
                self.canvas.drawPoint(px, py)
    
    def _draw_all_strokes(self, frame):
        """在视频帧上绘制所有涂鸦笔画"""
        for i, stroke in enumerate(self.drawing_strokes):
            for px, py, color in stroke['points']:
                cv2.circle(frame, (px, py), 5, color, -1)
            # 如果是选中的笔画，绘制边框
            if self.selected_stroke == i:
                # 找到笔画边界
                xs = [p[0] for p in stroke['points']]
                ys = [p[1] for p in stroke['points']]
                cv2.rectangle(frame, (min(xs)-5, min(ys)-5), (max(xs)+5, max(ys)+5), (0, 255, 255), 2)
    
    def _open_image(self):
        """打开图片文件进行识别"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片文件 (*.jpg *.jpeg *.png *.bmp)")

        if not file_path:
            return

        # 停止摄像头和视频
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None
            self.start_btn.setText("启动摄像头")

        if hasattr(self, 'video_player') and self.video_player.isRunning():
            self.video_player.stop()
            self.video_player = None

        # 使用npfromfile读取图片以支持中文路径
        try:
            frame = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            self.ai_result.setText(f"读取图片失败: {str(e)}")
            return

        if frame is None:
            self.ai_result.setText("无法读取图片文件，请检查文件格式")
            return
        
        # 检测手部关键点
        hand_bboxes, all_keypoints = self.hand_detector.detect(frame)

        # 绘制关键点和骨架（与摄像头识别一致）
        vis_frame = frame.copy()
        for bbox in hand_bboxes:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for keypoints in all_keypoints:
            draw_bd_handpose(vis_frame, keypoints, 0, 0)
            for i in range(21):
                if str(i) in keypoints:
                    x = int(keypoints[str(i)]["x"])
                    y = int(keypoints[str(i)]["y"])
                    cv2.circle(vis_frame, (x, y), 3, (255, 50, 60), -1)
        
        # 显示结果
        h, w, ch = vis_frame.shape
        bytes_per_line = ch * w
        q_img = QImage(vis_frame.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.video_label.setPixmap(QPixmap.fromImage(q_img.scaled(
            self.video_label.size(), Qt.KeepAspectRatio)))
        
        # 更新状态
        self.video_label.show()
        self.canvas.hide()
        self.ai_result.setText(f"图片识别完成\n检测到 {len(all_keypoints)} 只手")
    
    def _open_video(self):
        """打开视频文件进行播放和识别"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)")
        
        if not file_path:
            return
        
        # 停止摄像头
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None
            self.start_btn.setText("启动摄像头")
        
        # 创建视频播放线程
        self.video_player = VideoPlayer(file_path, self.hand_detector, self)
        self.video_player.frame_ready.connect(self._on_video_frame)
        self.video_player.start()
        
        self.ai_result.setText("正在播放视频...")
    
    def _on_video_frame(self, frame, hand_bboxes, all_keypoints):
        """处理视频帧"""
        vis_frame = frame.copy()
        
        # 绘制手部边界框和关键点（与摄像头识别一致）
        for bbox in hand_bboxes:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for keypoints in all_keypoints:
            draw_bd_handpose(vis_frame, keypoints, 0, 0)
            for i in range(21):
                if str(i) in keypoints:
                    x = int(keypoints[str(i)]["x"])
                    y = int(keypoints[str(i)]["y"])
                    cv2.circle(vis_frame, (x, y), 3, (255, 50, 60), -1)
        
        # 显示结果
        h, w, ch = vis_frame.shape
        bytes_per_line = ch * w
        q_img = QImage(vis_frame.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.video_label.setPixmap(QPixmap.fromImage(q_img.scaled(
            self.video_label.size(), Qt.KeepAspectRatio)))
        
        self.video_label.show()
        self.canvas.hide()
    
    def _on_width_changed(self, value):
        self.canvas.pen_width = value

    def closeEvent(self, event):
        if self.video_thread:
            self.video_thread.stop()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AIGestureGUI()
    window.show()
    sys.exit(app.exec_())
