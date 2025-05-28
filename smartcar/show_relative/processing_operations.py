import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
import time
import math
from .constants import VEHICLE_TYPE_DISTANCE, DEFAULT_DISTANCE
from ultralytics.utils.plotting import Annotator, colors
from collections import deque
import pandas as pd

class ProcessingOperations:
    def __init__(self, app_instance):
        self.app = app_instance

    def toggle_vehicle_detection(self):
        # ... (Copy from original file's toggle_vehicle_detection method)
        if self.app.is_vehicle_processing:
            self.app.is_vehicle_processing = False
            self.app.button_vehicle_detection.config(text="开始车辆检测")
            print("停止车辆检测。")
            self.toggle_speed_detection(auto=True)
            self.toggle_exception_detection(auto=True)
            self.app.update_status_label()
        else:
            if not self.app.video_path and not self.app.virtual_cam_playing:
                messagebox.showerror("错误", "请先上传视频或连接摄像头。")
                return
            self.app.is_vehicle_processing = True
            self.app.button_vehicle_detection.config(text="停止车辆检测")
            print("开始车辆检测。")
            self.app.update_status_label()

    def toggle_speed_detection(self, auto=False):
        # ... (Copy from original file's toggle_speed_detection method)
        if auto:
            self.app.is_speed_processing = False
            self.app.button_speed_detection.config(text="开始速度检测")
            print("自动停止速度检测。")
            return

        if not self.app.is_vehicle_processing:
            messagebox.showerror("提示", "请先开始车辆检测。")
            return

        if self.app.is_speed_processing:
            self.app.is_speed_processing = False
            self.app.button_speed_detection.config(text="开始速度检测")
            print("停止速度检测。")
        else:
            self.app.is_speed_processing = True
            self.app.button_speed_detection.config(text="停止速度检测")
            print("开始速度检测。")

    def toggle_exception_detection(self, auto=False):
        # ... (Copy from original file's toggle_exception_detection method)
        if auto:
            self.app.is_exception_processing = False
            self.app.button_exception_detection.config(text="开始异常检测")
            print("自动停止异常检测。")
            return

        if not self.app.is_vehicle_processing:
            messagebox.showerror("提示", "请先开始车辆检测。")
            return

        if self.app.is_exception_processing:
            self.app.is_exception_processing = False
            self.app.button_exception_detection.config(text="开始异常检测")
            print("停止异常检测。")
        else:
            self.app.is_exception_processing = True
            self.app.button_exception_detection.config(text="停止异常检测")
            print("开始异常检测。")

    def calculate_speed(self, obj_id, current_x_center, current_y_center, previous_x_center, previous_y_center,
                        y_pixels, class_name):
        # ... (Copy from original file's calculate_speed method)
        if obj_id not in self.app.prev_info:
            self.app.prev_info[obj_id] = {
                'x_center': current_x_center,
                'y_center': current_y_center,
                'frame_num': self.app.current_frame,
                'real_distance_per_pixel': VEHICLE_TYPE_DISTANCE.get(class_name,
                                                                     DEFAULT_DISTANCE) / y_pixels if y_pixels > 0 else DEFAULT_DISTANCE
            }
            return 0.0

        prev_frame = self.app.prev_info[obj_id]['frame_num']
        prev_x_center = self.app.prev_info[obj_id]['x_center']
        prev_y_center = self.app.prev_info[obj_id]['y_center']
        prev_real_distance_per_pixel = self.app.prev_info[obj_id]['real_distance_per_pixel']

        frame_gap = self.app.current_frame - prev_frame

        if frame_gap <= 0:
            return 0.0

        pixel_displacement = math.sqrt(
            (current_x_center - prev_x_center) ** 2 + (current_y_center - prev_y_center) ** 2)

        current_real_distance_per_pixel = VEHICLE_TYPE_DISTANCE.get(class_name,
                                                                    DEFAULT_DISTANCE) / y_pixels if y_pixels > 0 else prev_real_distance_per_pixel

        average_real_distance_per_pixel = (prev_real_distance_per_pixel + current_real_distance_per_pixel) / 2
        real_distance = average_real_distance_per_pixel * pixel_displacement
        time_interval = frame_gap / self.app.fps
        speed_m_s = real_distance / time_interval if time_interval > 0 else 0
        speed_km_h = speed_m_s * 3.6
        speed_km_h = max(0, min(speed_km_h, 200))

        self.app.prev_info[obj_id]['x_center'] = current_x_center
        self.app.prev_info[obj_id]['y_center'] = current_y_center
        self.app.prev_info[obj_id]['frame_num'] = self.app.current_frame
        self.app.prev_info[obj_id]['real_distance_per_pixel'] = current_real_distance_per_pixel

        return speed_km_h

    def process_frames(self):
        # ... (Copy from original file's process_frames method, ensure all self.app references are correct)
        exception_priority = {
            'Speeding Exception': 1,
            'Trajectory Exception': 2,
            'Collision Exception': 3
        }

        while True:
            try:
                frame, frame_num = self.app.raw_frame_queue.get(timeout=1)
                if self.app.y_max is None or self.app.line_y is None:
                    self.app.y_max = frame.shape[0]
                    self.app.line_y = int(self.app.y_max * 0.5)

            except self.app.queue.Empty:
                continue

            processed_frame = frame.copy()

            with self.app.stats_lock:
                self.app.processing = True

            exceptions = {} 
            detections = [] 

            try:
                if self.app.is_vehicle_processing:
                    self.app.vehicle_model.conf = self.app.conf_threshold
                    self.app.vehicle_model.iou = self.app.iou_threshold
                    self.app.vehicle_model.imgsz = 640

                    results = self.app.vehicle_model.track(source=frame, conf=self.app.vehicle_model.conf,
                                                       iou=self.app.vehicle_model.iou,
                                                       show=False, verbose=False, stream=True,
                                                       tracker="bytetrack.yaml",
                                                       persist=True, imgsz=self.app.vehicle_model.imgsz, augment=False)
                    for result in results:
                        if result.boxes.id is None:
                            continue
                        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                        ids = result.boxes.id.cpu().numpy()
                        classes = result.boxes.cls.cpu().numpy().astype(int)

                        for box, obj_id, cls_idx in zip(boxes_xyxy, ids, classes):
                            if obj_id == -1:
                                self.app.current_vehicle_id += 1
                                obj_id = self.app.current_vehicle_id
                                
                            class_name = self.app.vehicle_model.names.get(cls_idx, "unknown")
                            if class_name not in VEHICLE_TYPE_DISTANCE:
                                continue

                            x1, y1, x2, y2 = map(int, box)
                            y_pixels = y2 - y1 if (y2 - y1) > 0 else 1
                            detections.append({
                                'id': int(obj_id),
                                'class_name': class_name,
                                'x1': x1,
                                'y1': y1,
                                'x2': x2,
                                'y2': y2,
                                'y_pixels': y_pixels
                            })

                self.app.current_frame_num = frame_num
                detections = self.filter_overlapping_boxes(detections)

                if self.app.is_vehicle_processing:
                    for det in detections:
                        obj_id = det['id']
                        y_center = (det['y1'] + det['y2']) / 2
                
                        if obj_id in self.app.vehicle_positions:
                            prev_position = self.app.vehicle_positions[obj_id]['y_center']
                            counted = self.app.vehicle_positions[obj_id]['counted']
                            if prev_position < self.app.line_y <= y_center and not counted:
                                self.app.flow_counter += 1
                                self.app.vehicle_positions[obj_id]['counted'] = True
                        else:
                            self.app.vehicle_positions[obj_id] = {'y_center': y_center, 'counted': False}
                
                        if y_center < self.app.line_y:
                            self.app.vehicle_positions[obj_id]['counted'] = False
                
                    if detections:
                        self.app.last_detection_time = time.time()
                
                    current_time = time.time()
                    if current_time - self.app.window_start_time >= self.app.flow_window_size_seconds:
                        if current_time - self.app.last_detection_time >= 2:
                            self.app.current_flow_rate = 0
                            self.app.flow_history.append(0)
                            if len(self.app.flow_history) > 100:
                                self.app.flow_history.pop(0)
                            self.app.flow_counter = 0
                            self.app.window_start_time = current_time
                        else:
                            flow_per_window = self.app.flow_counter * (3600 / self.app.flow_window_size_seconds) / self.app.number_of_lanes
                            flow_per_window = max(flow_per_window, 0)
                            self.app.flow_history.append(flow_per_window)
                            if len(self.app.flow_history) > 100:
                                self.app.flow_history.pop(0)
                            self.app.current_flow_rate = sum(self.app.flow_history) / len(self.app.flow_history) if self.app.flow_history else 0
                            self.app.flow_counter = 0
                            self.app.window_start_time = current_time

                for det in detections:
                    obj_id = det['id']
                    class_name = det['class_name']
                    x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
                    y_center = (y1 + y2) / 2
                    y_pixels = det['y_pixels']
                    x_center = (x1 + x2) / 2

                    if obj_id not in self.app.trajectory_history:
                        self.app.trajectory_history[obj_id] = []
                    self.app.trajectory_history[obj_id].append(np.array([x_center, y_center]))
                    if len(self.app.trajectory_history[obj_id]) > 3:
                        self.app.trajectory_history[obj_id].pop(0)

                    if self.app.is_speed_processing:
                        speed = self.calculate_speed(obj_id, x_center, y_center,
                                                     self.app.prev_info[obj_id][
                                                         'x_center'] if obj_id in self.app.prev_info else x_center,
                                                     self.app.prev_info[obj_id][
                                                         'y_center'] if obj_id in self.app.prev_info else y_center,
                                                     y_pixels, class_name)

                        if obj_id not in self.app.speed_history:
                            self.app.speed_history[obj_id] = speed
                        else:
                            self.app.speed_history[obj_id] = self.app.alpha * speed + (1 - self.app.alpha) * self.app.speed_history[
                                obj_id]
                        avg_speed = self.app.speed_history[obj_id]
                        det['speed'] = avg_speed
                        self.app.speed_stats.append(avg_speed)

                        if avg_speed > self.app.speed_threshold:
                            current_exception = 'Speeding Exception'
                            if (obj_id not in exceptions) or (
                                    exception_priority[current_exception] > exception_priority.get(exceptions[obj_id],
                                                                                                   0)):
                                exceptions[obj_id] = current_exception
                            if self.app.is_exception_processing:
                                print(f"车辆ID {obj_id} 超速异常，速度: {avg_speed:.2f} km/h")

                    if self.app.is_exception_processing:
                        if obj_id in self.app.trajectory_history:
                            past_trace = self.app.trajectory_history[obj_id]
                            if len(past_trace) == 3:
                                p1 = past_trace[-3]
                                p2 = past_trace[-2]
                                p3 = past_trace[-1]
                                vec1 = p2 - p1
                                vec2 = p3 - p2
                                angle = self.calculate_angle_between_vectors(vec1, vec2)
                                displacement1 = np.linalg.norm(vec1)
                                displacement2 = np.linalg.norm(vec2)
                                if displacement1 >= self.app.min_displacement and displacement2 >= self.app.min_displacement:
                                    if angle > self.app.angle_threshold and angle <= 90:
                                        if (obj_id, self.app.current_frame_num) not in self.app.trajectory_exception_records:
                                            current_exception = 'Trajectory Exception'
                                            if (obj_id not in exceptions) or (
                                                    exception_priority[current_exception] > exception_priority.get(
                                                exceptions[obj_id], 0)):
                                                exceptions[obj_id] = current_exception
                                            print(f"车辆ID {obj_id} 检测到轨迹异常：角度变化 {angle:.2f} 度")
                                            self.app.trajectory_exception_records.add((obj_id, self.app.current_frame_num))

                if self.app.is_vehicle_processing:
                    for det in detections:
                        obj_id = det['id']
                        class_name = det['class_name']
                        if obj_id not in self.app.counted_vehicle_ids:
                            if class_name in self.app.vehicle_type_counts:
                                self.app.vehicle_type_counts[class_name] += 1
                                self.app.counted_vehicle_ids.add(obj_id)

                if self.app.is_vehicle_processing:
                    for i, det1 in enumerate(detections):
                        for det2 in detections[i + 1:]:
                            obj_id1 = det1['id']
                            obj_id2 = det2['id']
                            iou = self.compute_iou(det1, det2)
                            if self.app.iou_threshold < iou < 0.6:
                                collision_key = (obj_id1, obj_id2)
                                if (collision_key not in self.app.collision_records) or \
                                   (self.app.collision_records[collision_key] != frame_num - 1):
                                    self.app.collision_records[collision_key] = frame_num
                                    current_exception = 'Collision Exception'
                                    for obj_id in [obj_id1, obj_id2]:
                                        if (obj_id not in exceptions) or (
                                                exception_priority[current_exception] > exception_priority.get(exceptions[obj_id], 0)):
                                            exceptions[obj_id] = current_exception
                                        det = next((d for d in detections if d['id'] == obj_id), None)
                                        if det:
                                            det['speed'] = self.app.speed_history.get(obj_id, 0.0)
                                    if self.app.is_exception_processing:
                                        print(f"车辆ID {obj_id1} 与车辆ID {obj_id2} 发生碰撞异常。")

                for det in detections:
                    obj_id = det['id']
                    class_name = det['class_name']
                    x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
                    speed = det.get('speed', 0.0)
                    box_color = (255, 0, 0)
                    status_label = ""

                    if self.app.is_exception_processing:
                        exception = exceptions.get(obj_id, 'Normal')
                        if exception == 'Speeding Exception':
                            box_color = (0, 0, 255)
                            status_label = "Speeding Exception"
                        elif exception == 'Collision Exception':
                            box_color = (0, 255, 255)
                            status_label = "Collision Exception"
                        elif exception == 'Trajectory Exception':
                            box_color = (255, 0, 255)
                            status_label = "Trajectory Exception"

                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), box_color, 2)

                    if self.app.is_speed_processing:
                        label = f"ID:{obj_id} {class_name} Speed:{speed:.2f} km/h"
                    else:
                        label = f"ID:{obj_id} {class_name}"

                    (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    label_y = max(y1 - label_height - baseline, 0)
                    cv2.rectangle(processed_frame, (x1, label_y), (x1 + label_width, y1), box_color, -1)
                    cv2.putText(processed_frame, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 2)

                    if status_label:
                        (status_width, status_height), status_baseline = cv2.getTextSize(status_label,
                                                                                         cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                                                                         2)
                        status_x = x1
                        status_y = y2 + status_height + 10 if y2 + status_height + 10 < self.app.y_max else y2 - 10
                        cv2.putText(processed_frame, status_label, (status_x, status_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if self.app.is_vehicle_processing:
                    traffic_flow_display = self.app.current_flow_rate
                    cv2.putText(processed_frame, f"Traffic Flow: {traffic_flow_display:.2f} veh/h",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

                try:
                    self.app.processed_frame_queue.put_nowait(processed_frame.copy())
                except self.app.queue.Full:
                    try:
                        self.app.processed_frame_queue.get_nowait()
                        self.app.processed_frame_queue.put_nowait(processed_frame.copy())
                    except self.app.queue.Empty:
                        pass
            except Exception as e:
                print(f"处理帧时发生错误: {e}")
            finally:
                with self.app.stats_lock:
                    self.app.processing = False

    def filter_overlapping_boxes(self, detections):
        # ... (Copy from original file's filter_overlapping_boxes method)
        if not detections:
            return []

        detections = sorted(detections, key=lambda x: x.get('confidence', 0), reverse=True)
        filtered = []
        occupied_areas = []

        for det in detections:
            x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
            overlap = False
            for area in occupied_areas:
                iou = self.compute_iou({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}, area)
                if iou > 0.5:
                    overlap = True
                    break
            if not overlap:
                filtered.append(det)
                occupied_areas.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
        return filtered

    def is_enclosed(self, det1, det2):
        # ... (Copy from original file's is_enclosed method)
        return det1['x1'] >= det2['x1'] and det1['y1'] >= det2['y1'] and \
            det1['x2'] <= det2['x2'] and det1['y2'] <= det2['y2']

    def calculate_angle_between_vectors(self, vec1, vec2):
        # ... (Copy from original file's calculate_angle_between_vectors method)
        dot_prod = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        cos_theta = np.clip(dot_prod / (norm1 * norm2), -1.0, 1.0)
        angle = math.degrees(math.acos(cos_theta))
        return angle

    def compute_iou(self, det1, det2):
        # ... (Copy from original file's compute_iou method)
        xA = max(det1['x1'], det2['x1'])
        yA = max(det1['y1'], det2['y1'])
        xB = min(det1['x2'], det2['x2'])
        yB = min(det1['y2'], det2['y2'])

        interWidth = max(0, xB - xA + 1)
        interHeight = max(0, yB - yA + 1)
        interArea = interWidth * interHeight

        boxAArea = (det1['x2'] - det1['x1'] + 1) * (det1['y2'] - det1['y1'] + 1)
        boxBArea = (det2['x2'] - det2['x1'] + 1) * (det2['y2'] - det2['y1'] + 1)

        iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
        return iou 