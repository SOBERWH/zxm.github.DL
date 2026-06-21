# -*-coding:utf-8-*-
# date:2024-01-15
# Author: Eric.Lee & 改进
# function: 手势识别 - 支持 Tap, Double Tap, Pinch, Rotation, Swipe, Pan, LongPress
#

import os
import cv2
import numpy as np
import time
from collections import deque
from enum import Enum


class GestureType(Enum):
    NONE = "none"
    TAP = "tap"
    DOUBLE_TAP = "double_tap"
    PINCH = "pinch"
    PINCH_OPEN = "pinch_open"
    PINCH_CLOSE = "pinch_close"
    ROTATION = "rotation"
    ROTATION_CW = "rotation_cw"
    ROTATION_CCW = "rotation_ccw"
    SWIPE = "swipe"
    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"
    SWIPE_UP = "swipe_up"
    SWIPE_DOWN = "swipe_down"
    PAN = "pan"
    LONG_PRESS = "long_press"
    FIST = "fist"
    CATCH = "catch"
    OK = "ok"
    FIVE = "five"
    GUN = "gun"
    LOVE = "love"
    ONE = "one"
    SIX = "six"
    THREE = "three"
    THUMBUP = "thumbup"
    YEAH = "yeah"


class KeypointHistory:
    def __init__(self, maxlen=30):
        self.history = deque(maxlen=maxlen)

    def update(self, keypoints):
        self.history.append(keypoints.copy())

    def get_position(self, point_id):
        if len(self.history) == 0:
            return None
        return self.history[-1].get(str(point_id))

    def get_velocity(self, point_id, frames=5):
        if len(self.history) < frames + 1:
            return None
        current = self.history[-1].get(str(point_id))
        previous = self.history[-(frames + 1)].get(str(point_id))
        if current is None or previous is None:
            return None
        dt = frames * 0.033
        return {
            'x': (current['x'] - previous['x']) / dt,
            'y': (current['y'] - previous['y']) / dt
        }


class HandGestureRecognizer:
    THUMB = 4
    INDEX = 8
    MIDDLE = 12
    RING = 16
    PINKY = 20

    def __init__(self,
                 tap_threshold=30.0,
                 tap_duration=0.5,
                 double_tap_interval=0.3,
                 double_tap_max_gap=0.5,
                 pinch_threshold_open=80.0,
                 pinch_threshold_close=40.0,
                 rotation_threshold=15.0,
                 swipe_threshold=200.0,
                 pan_threshold=50.0,
                 long_press_duration=1.0,
                 velocity_smooth=5):

        self.tap_threshold = tap_threshold
        self.tap_duration = tap_duration
        self.double_tap_interval = double_tap_interval
        self.double_tap_max_gap = double_tap_max_gap
        self.pinch_threshold_open = pinch_threshold_open
        self.pinch_threshold_close = pinch_threshold_close
        self.rotation_threshold = rotation_threshold
        self.swipe_threshold = swipe_threshold
        self.pan_threshold = pan_threshold
        self.long_press_duration = long_press_duration
        self.velocity_smooth = velocity_smooth

        self.index_history = KeypointHistory(maxlen=30)
        self.thumb_history = KeypointHistory(maxlen=30)
        self.index_pinch_history = deque(maxlen=30)
        self.rotation_history = deque(maxlen=30)

        self.tap_start_time = None
        self.tap_position = None
        self.last_tap_time = None
        self.last_tap_position = None
        self.pending_double_tap = False

        self.long_press_start = None
        self.long_press_position = None
        self.is_long_pressing = False

        self.pinch_state = 'open'
        self.pinch_start_ratio = None

        self.rotation_start_angle = None
        self.is_rotating = False

        self.pan_start_position = None
        self.pan_start_time = None
        self.is_panning = False
        self.swipe_start_position = None
        self.swipe_start_time = None
        self.is_swiping = False

        self.last_gesture = GestureType.NONE
        self.gesture_start_time = None

    def _get_distance(self, p1, p2):
        if p1 is None or p2 is None:
            return None
        return np.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)

    def _get_angle(self, center, point):
        if center is None or point is None:
            return None
        return np.arctan2(point['y'] - center['y'], point['x'] - center['x'])

    def _get_pinch_ratio(self, keypoints):
        thumb_tip = keypoints.get(str(self.THUMB))
        index_tip = keypoints.get(str(self.INDEX))

        if thumb_tip is None or index_tip is None:
            return None

        distance = self._get_distance(thumb_tip, index_tip)

        wrist = keypoints.get('0')
        if wrist is None:
            return distance

        ref_distance = self._get_distance(thumb_tip, wrist)
        if ref_distance is None or ref_distance == 0:
            return distance

        return distance / ref_distance

    def _is_finger_extended(self, keypoints, finger_tip, finger_base):
        if finger_tip is None or finger_base is None:
            return False

        tip_y = finger_tip['y']
        base_y = finger_base['y']

        return tip_y < base_y

    def _get_extended_fingers(self, keypoints):
        thumb_tip = keypoints.get(str(self.THUMB))
        thumb_base = keypoints.get('1')
        index_tip = keypoints.get(str(self.INDEX))
        index_base = keypoints.get('5')
        middle_tip = keypoints.get(str(self.MIDDLE))
        middle_base = keypoints.get('9')
        ring_tip = keypoints.get(str(self.RING))
        ring_base = keypoints.get('13')
        pinky_tip = keypoints.get(str(self.PINKY))
        pinky_base = keypoints.get('17')

        fingers = {
            'thumb': self._is_finger_extended(keypoints, thumb_tip, thumb_base) if thumb_tip and thumb_base else False,
            'index': self._is_finger_extended(keypoints, index_tip, index_base) if index_tip and index_base else False,
            'middle': self._is_finger_extended(keypoints, middle_tip, middle_base) if middle_tip and middle_base else False,
            'ring': self._is_finger_extended(keypoints, ring_tip, ring_base) if ring_tip and ring_base else False,
            'pinky': self._is_finger_extended(keypoints, pinky_tip, pinky_base) if pinky_tip and pinky_base else False
        }
        return fingers

    def _detect_static_gesture(self, keypoints):
        fingers = self._get_extended_fingers(keypoints)

        thumb_tip = keypoints.get(str(self.THUMB))
        index_tip = keypoints.get(str(self.INDEX))
        middle_tip = keypoints.get(str(self.MIDDLE))
        ring_tip = keypoints.get(str(self.RING))
        pinky_tip = keypoints.get(str(self.PINKY))

        thumb_index_dist = self._get_distance(thumb_tip, index_tip) if thumb_tip and index_tip else None
        index_middle_dist = self._get_distance(index_tip, middle_tip) if index_tip and middle_tip else None
        middle_ring_dist = self._get_distance(middle_tip, ring_tip) if middle_tip and ring_tip else None
        ring_pinky_dist = self._get_distance(ring_tip, pinky_tip) if ring_tip and pinky_tip else None

        palm_center = self._get_palm_center(keypoints)
        index_palm_dist = self._get_distance(index_tip, palm_center) if index_tip and palm_center else None
        middle_palm_dist = self._get_distance(middle_tip, palm_center) if middle_tip and palm_center else None
        pinky_palm_dist = self._get_distance(pinky_tip, palm_center) if pinky_tip and palm_center else None

        extended_count = sum(fingers.values())

        if extended_count == 0:
            return GestureType.FIST, 0.92

        if fingers['thumb'] and fingers['index'] and not fingers['middle'] and not fingers['ring'] and not fingers['pinky']:
            if thumb_index_dist is not None and thumb_index_dist < 50:
                return GestureType.OK, 0.90

        if extended_count == 5:
            return GestureType.FIVE, 0.95

        if fingers['index'] and not fingers['middle'] and not fingers['ring'] and not fingers['pinky'] and not fingers['thumb']:
            return GestureType.ONE, 0.92

        if fingers['index'] and fingers['middle'] and not fingers['ring'] and not fingers['pinky'] and not fingers['thumb']:
            return GestureType.YEAH, 0.90

        if fingers['index'] and fingers['middle'] and fingers['ring'] and not fingers['pinky'] and not fingers['thumb']:
            return GestureType.THREE, 0.88

        if not fingers['index'] and not fingers['middle'] and not fingers['ring'] and not fingers['pinky'] and fingers['thumb']:
            return GestureType.THUMBUP, 0.90

        if fingers['index'] and not fingers['middle'] and not fingers['ring'] and fingers['pinky'] and not fingers['thumb']:
            return GestureType.SIX, 0.85

        if fingers['index'] and not fingers['middle'] and not fingers['ring'] and fingers['pinky'] and fingers['thumb']:
            if thumb_index_dist is not None and index_palm_dist is not None and thumb_index_dist < index_palm_dist * 0.5:
                return GestureType.GUN, 0.85

        if fingers['thumb'] and fingers['index'] and not fingers['middle'] and not fingers['ring'] and not fingers['pinky']:
            if thumb_index_dist is not None and thumb_index_dist < 40:
                if index_palm_dist is not None and middle_palm_dist is not None:
                    if index_palm_dist < 80 and middle_palm_dist < 80:
                        return GestureType.LOVE, 0.85

        if extended_count <= 2 and not fingers['middle'] and not fingers['ring']:
            if not fingers['thumb'] and not fingers['pinky']:
                return GestureType.CATCH, 0.80

        return GestureType.NONE, 0.0

    def _get_palm_center(self, keypoints):
        indices = [0, 1, 5, 9, 13, 17]
        xs = []
        ys = []
        for idx in indices:
            pt = keypoints.get(str(idx))
            if pt:
                xs.append(pt['x'])
                ys.append(pt['y'])

        if len(xs) == 0:
            return None
        return {'x': np.mean(xs), 'y': np.mean(ys)}

    def update(self, keypoints):
        if keypoints is None or len(keypoints) == 0:
            self.last_confidence = 0.0
            return GestureType.NONE

        self.index_history.update(keypoints)
        self.thumb_history.update(keypoints)

        gesture, confidence = self._detect_gesture(keypoints)
        self.last_confidence = confidence
        return gesture, confidence

    def get_last_confidence(self):
        return getattr(self, 'last_confidence', 0.0)

    def _detect_gesture(self, keypoints):
        current_time = time.time()

        index_tip = keypoints.get(str(self.INDEX))
        thumb_tip = keypoints.get(str(self.THUMB))

        pinch_ratio = self._get_pinch_ratio(keypoints)
        if pinch_ratio is not None:
            self.index_pinch_history.append(pinch_ratio)

        if self._detect_long_press(current_time, index_tip):
            self.last_gesture = GestureType.LONG_PRESS
            self.gesture_start_time = current_time
            return GestureType.LONG_PRESS, 0.95

        rotation_result = self._detect_rotation(keypoints)
        if isinstance(rotation_result, tuple):
            rotation_gesture, rotation_conf = rotation_result
        else:
            rotation_gesture, rotation_conf = rotation_result, 0.0
        if rotation_gesture != GestureType.NONE:
            return rotation_gesture, rotation_conf

        pinch_result = self._detect_pinch(pinch_ratio)
        if isinstance(pinch_result, tuple):
            pinch_gesture, pinch_conf = pinch_result
        else:
            pinch_gesture, pinch_conf = pinch_result, 0.0
        if pinch_gesture != GestureType.NONE:
            return pinch_gesture, pinch_conf

        swipe_pan_result = self._detect_swipe_pan(current_time, index_tip)
        if isinstance(swipe_pan_result, tuple):
            swipe_pan_gesture, swipe_pan_conf = swipe_pan_result
        else:
            swipe_pan_gesture, swipe_pan_conf = swipe_pan_result, 0.0
        if swipe_pan_gesture != GestureType.NONE:
            return swipe_pan_gesture, swipe_pan_conf

        tap_result = self._detect_tap(current_time, index_tip)
        if isinstance(tap_result, tuple):
            tap_gesture, tap_conf = tap_result
        else:
            tap_gesture, tap_conf = tap_result, 0.0
        if tap_gesture != GestureType.NONE:
            return tap_gesture, tap_conf

        static_result = self._detect_static_gesture(keypoints)
        if isinstance(static_result, tuple):
            static_gesture, static_conf = static_result
        else:
            static_gesture, static_conf = static_result, 0.0
        if static_gesture != GestureType.NONE:
            return static_gesture, static_conf

        return GestureType.NONE, 0.0

    def _detect_tap(self, current_time, index_tip):
        if index_tip is None:
            if self.tap_start_time is not None:
                self.tap_start_time = None
                self.tap_position = None
            return GestureType.NONE, 0.0

        velocity = self.index_history.get_velocity(str(self.INDEX), frames=3)

        if velocity is None:
            return GestureType.NONE, 0.0

        speed = np.sqrt(velocity['x']**2 + velocity['y']**2)

        if self.tap_start_time is None:
            if speed < 50:
                self.tap_start_time = current_time
                self.tap_position = index_tip.copy()
                self.pending_double_tap = False
        else:
            elapsed = current_time - self.tap_start_time

            if elapsed > self.tap_duration:
                self.tap_start_time = None
                self.tap_position = None
                return GestureType.NONE, 0.0

            if speed > 100:
                tap_duration = elapsed

                self.tap_start_time = None
                position = self.tap_position
                self.tap_position = None

                if tap_duration < self.tap_duration:
                    if self.last_tap_time is not None:
                        time_gap = current_time - self.last_tap_time
                        if time_gap < self.double_tap_max_gap:
                            self.last_tap_time = None
                            self.last_tap_position = None
                            self.pending_double_tap = False
                            return GestureType.DOUBLE_TAP, 0.92

                    self.last_tap_time = current_time
                    self.last_tap_position = position
                    return GestureType.TAP, 0.92

        return GestureType.NONE, 0.0

    def _detect_long_press(self, current_time, index_tip):
        if index_tip is None:
            self.long_press_start = None
            self.long_press_position = None
            self.is_long_pressing = False
            return False

        velocity = self.index_history.get_velocity(str(self.INDEX), frames=5)

        if velocity is None:
            return False

        speed = np.sqrt(velocity['x']**2 + velocity['y']**2)

        if speed < 10:
            if self.long_press_start is None:
                self.long_press_start = current_time
                self.long_press_position = index_tip.copy()
            else:
                elapsed = current_time - self.long_press_start
                if elapsed >= self.long_press_duration:
                    self.is_long_pressing = True
                    return True
        else:
            self.long_press_start = None
            self.long_press_position = None
            self.is_long_pressing = False

        return False

    def _detect_pinch(self, pinch_ratio):
        if pinch_ratio is None or len(self.index_pinch_history) < 5:
            return GestureType.NONE, 0.0

        recent_ratios = list(self.index_pinch_history)[-10:]
        avg_ratio = np.mean(recent_ratios)

        if self.pinch_state == 'open':
            if avg_ratio < self.pinch_threshold_close:
                self.pinch_state = 'closing'
                self.pinch_start_ratio = avg_ratio
        elif self.pinch_state == 'closing':
            if avg_ratio > self.pinch_threshold_open:
                delta = avg_ratio - self.pinch_start_ratio
                if delta > 0.15:
                    self.pinch_state = 'open'
                    return GestureType.PINCH_CLOSE, 0.90
        elif self.pinch_state == 'close':
            if avg_ratio > self.pinch_threshold_open:
                self.pinch_state = 'opening'
                self.pinch_start_ratio = avg_ratio
        elif self.pinch_state == 'opening':
            if avg_ratio < self.pinch_threshold_close:
                delta = self.pinch_start_ratio - avg_ratio
                if delta > 0.15:
                    self.pinch_state = 'close'
                    return GestureType.PINCH_OPEN, 0.90

        if avg_ratio < self.pinch_threshold_close:
            self.pinch_state = 'close'
        else:
            self.pinch_state = 'open'

        return GestureType.NONE, 0.0

    def _detect_rotation(self, keypoints):
        thumb_tip = keypoints.get(str(self.THUMB))
        index_tip = keypoints.get(str(self.INDEX))
        middle_tip = keypoints.get(str(self.MIDDLE))

        if thumb_tip is None or index_tip is None:
            self.rotation_start_angle = None
            self.is_rotating = False
            return GestureType.NONE, 0.0

        palm_center = self._get_palm_center(keypoints)
        if palm_center is None:
            return GestureType.NONE, 0.0

        angle = self._get_angle(palm_center, index_tip)
        if angle is None:
            return GestureType.NONE, 0.0

        self.rotation_history.append(angle)

        if len(self.rotation_history) < 3:
            return GestureType.NONE, 0.0

        angles = list(self.rotation_history)
        total_rotation = 0
        for i in range(1, len(angles)):
            diff = angles[i] - angles[i-1]
            if diff > np.pi:
                diff -= 2 * np.pi
            elif diff < -np.pi:
                diff += 2 * np.pi
            total_rotation += diff

        total_degrees = abs(np.degrees(total_rotation))

        velocity = self.index_history.get_velocity(str(self.INDEX), frames=5)
        if velocity is None:
            return GestureType.NONE, 0.0

        speed = np.sqrt(velocity['x']**2 + velocity['y']**2)

        if speed > 80 and total_degrees > self.rotation_threshold:
            self.is_rotating = True
            if total_rotation > 0:
                return GestureType.ROTATION_CW, 0.85
            else:
                return GestureType.ROTATION_CCW, 0.85

        if total_degrees > 60:
            self.is_rotating = True

        if self.is_rotating and speed < 30:
            self.is_rotating = False
            self.rotation_start_angle = None
            if total_rotation > 0:
                return GestureType.ROTATION_CW, 0.85
            else:
                return GestureType.ROTATION_CCW, 0.85

        return GestureType.NONE, 0.0

    def _detect_swipe_pan(self, current_time, index_tip):
        if index_tip is None:
            self.swipe_start_position = None
            self.swipe_start_time = None
            self.is_swiping = False
            self.pan_start_position = None
            self.pan_start_time = None
            self.is_panning = False
            return GestureType.NONE, 0.0

        velocity = self.index_history.get_velocity(str(self.INDEX), frames=3)

        if velocity is None:
            return GestureType.NONE, 0.0

        speed = np.sqrt(velocity['x']**2 + velocity['y']**2)

        if speed > 150:
            if not self.is_swiping:
                self.is_swiping = True
                self.swipe_start_position = index_tip.copy()
                self.swipe_start_time = current_time
            else:
                elapsed = current_time - self.swipe_start_time
                if elapsed > 0.8:
                    self.is_swiping = False
                    self.swipe_start_position = None
                    self.swipe_start_time = None
                    return GestureType.SWIPE, 0.88
        else:
            if self.is_swiping:
                self.is_swiping = False
                start_pos = self.swipe_start_position
                self.swipe_start_position = None
                self.swipe_start_time = None
                if start_pos is not None:
                    distance = self._get_distance(start_pos, index_tip)
                    if distance is not None and distance > 50:
                        dx = index_tip['x'] - start_pos['x']
                        dy = index_tip['y'] - start_pos['y']
                        if abs(dx) > abs(dy):
                            if dx > 0:
                                return GestureType.SWIPE_RIGHT, 0.88
                            else:
                                return GestureType.SWIPE_LEFT, 0.88
                        else:
                            if dy > 0:
                                return GestureType.SWIPE_DOWN, 0.88
                            else:
                                return GestureType.SWIPE_UP, 0.88

        if 30 < speed < 100:
            if not self.is_panning:
                self.is_panning = True
                self.pan_start_position = index_tip.copy()
                self.pan_start_time = current_time
        else:
            if self.is_panning and speed < 30:
                self.is_panning = False
                start_pos = self.pan_start_position
                start_time = self.pan_start_time
                self.pan_start_position = None
                self.pan_start_time = None

                if start_pos is not None and start_time is not None:
                    elapsed = current_time - start_time
                    distance = self._get_distance(start_pos, index_tip)
                    if distance is not None and elapsed > 0.3:
                        return GestureType.PAN, 0.85

        return GestureType.NONE, 0.0

    def get_pinch_scale(self):
        if len(self.index_pinch_history) < 2:
            return 1.0
        current = self.index_pinch_history[-1]
        initial = self.index_pinch_history[0]
        if initial == 0:
            return 1.0
        return current / initial

    def get_rotation_angle(self):
        if len(self.rotation_history) < 2:
            return 0.0

        angles = list(self.rotation_history)
        total_rotation = 0
        for i in range(1, len(angles)):
            diff = angles[i] - angles[i-1]
            if diff > np.pi:
                diff -= 2 * np.pi
            elif diff < -np.pi:
                diff += 2 * np.pi
            total_rotation += diff

        return np.degrees(total_rotation)

    def get_pan_delta(self, current_pos):
        if self.pan_start_position is None or current_pos is None:
            return None, None
        return (current_pos['x'] - self.pan_start_position['x'],
                current_pos['y'] - self.pan_start_position['y'])

    def reset(self):
        self.tap_start_time = None
        self.tap_position = None
        self.last_tap_time = None
        self.last_tap_position = None
        self.pending_double_tap = False
        self.long_press_start = None
        self.long_press_position = None
        self.is_long_pressing = False
        self.pinch_state = 'open'
        self.pinch_start_ratio = None
        self.rotation_start_angle = None
        self.is_rotating = False
        self.pan_start_position = None
        self.pan_start_time = None
        self.is_panning = False
        self.swipe_start_position = None
        self.swipe_start_time = None
        self.is_swiping = False
        self.last_gesture = GestureType.NONE
        self.gesture_start_time = None
        self.index_history.history.clear()
        self.thumb_history.history.clear()
        self.index_pinch_history.clear()
        self.rotation_history.clear()


class MultiHandGestureRecognizer:
    def __init__(self, max_hands=2, **kwargs):
        self.max_hands = max_hands
        self.hand_gestures = {}
        self.hand_trackers = {}
        self.kwargs = kwargs

    def update(self, hand_results):
        gestures = {}

        for idx, hand_data in enumerate(hand_results):
            hand_id = hand_data.get('hand_id', idx)
            keypoints = hand_data.get('keypoints', {})

            if hand_id not in self.hand_trackers:
                self.hand_trackers[hand_id] = HandGestureRecognizer(**self.kwargs)

            gesture = self.hand_trackers[hand_id].update(keypoints)
            gestures[hand_id] = gesture

        self.hand_gestures = gestures
        return gestures

    def get_gesture(self, hand_id):
        return self.hand_gestures.get(hand_id, GestureType.NONE)

    def get_all_gestures(self):
        return self.hand_gestures.copy()

    def reset(self, hand_id=None):
        if hand_id is None:
            for tracker in self.hand_trackers.values():
                tracker.reset()
            self.hand_gestures = {}
        elif hand_id in self.hand_trackers:
            self.hand_trackers[hand_id].reset()
            del self.hand_gestures[hand_id]


def draw_gesture_text(img, gesture, position=(10, 150), scale=0.8, color=(255, 255, 0)):
    gesture_names = {
        GestureType.TAP: "TAP",
        GestureType.DOUBLE_TAP: "DOUBLE TAP",
        GestureType.PINCH: "PINCH",
        GestureType.ROTATION: "ROTATION",
        GestureType.SWIPE: "SWIPE",
        GestureType.PAN: "PAN",
        GestureType.LONG_PRESS: "LONG PRESS"
    }

    text = gesture_names.get(gesture, "NONE")

    cv2.putText(img, f"Gesture: {text}", position,
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)
    return img


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    from hand_data_iter.datasets import draw_bd_handpose

    try:
        from gesture_trigger import HandPoseDetector, SimpleHandDetector, MEDIAPIPE_AVAILABLE, MediaPipeHandDetector
    except ImportError:
        print("请先确保 gesture_trigger.py 存在")
        exit()

    print('='*60)
    print('手势识别测试')
    print('='*60)
    print('支持手势: Tap, Double Tap, Pinch, Rotation, Swipe, Pan, LongPress')
    print('='*60)

    model_path = 'D:\\pretrain\\ReXNetV1-size-256-wingloss102-0.122.pth'
    detector = HandPoseDetector(model_path=model_path, model_type='ReXNetV1')

    hand_detector = SimpleHandDetector(max_hands=2)

    gesture_recognizer = HandGestureRecognizer(
        tap_threshold=30.0,
        tap_duration=0.5,
        double_tap_interval=0.3,
        double_tap_max_gap=0.5,
        pinch_threshold_open=80.0,
        pinch_threshold_close=40.0,
        rotation_threshold=15.0,
        swipe_threshold=200.0,
        pan_threshold=50.0,
        long_press_duration=1.0
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        exit()

    print("\n操作说明:")
    print("  - 单手举起: Tap, Double Tap, Long Press")
    print("  - 双指缩放: Pinch (拇指+食指)")
    print("  - 旋转: Rotation (移动时旋转)")
    print("  - 滑动: Swipe (快速移动)")
    print("  - 拖移: Pan (慢速移动)")
    print("  - 按 'r' 重置")
    print("  - 按 'q' 退出\n")

    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hand_bboxes, _ = hand_detector.detect(frame)
        hand_results = detector.predict(frame, hand_bboxes)

        for hand_data in hand_results:
            keypoints = hand_data['keypoints']
            gesture = gesture_recognizer.update(keypoints)

            gesture_color = (0, 255, 255)
            if gesture == GestureType.TAP:
                gesture_color = (0, 255, 0)
            elif gesture == GestureType.DOUBLE_TAP:
                gesture_color = (0, 128, 255)
            elif gesture == GestureType.PINCH:
                gesture_color = (255, 0, 255)
            elif gesture == GestureType.ROTATION:
                gesture_color = (255, 255, 0)
            elif gesture == GestureType.SWIPE:
                gesture_color = (0, 165, 255)
            elif gesture == GestureType.PAN:
                gesture_color = (128, 128, 255)
            elif gesture == GestureType.LONG_PRESS:
                gesture_color = (0, 0, 255)

            draw_gesture_text(frame, gesture, color=gesture_color)

        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
        prev_time = current_time

        vis_frame = detector.visualize(frame, hand_results, fps)
        cv2.imshow('Gesture Recognition', vis_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            gesture_recognizer.reset()
            print("已重置")

    cap.release()
    cv2.destroyAllWindows()
    print("\n测试完成!")