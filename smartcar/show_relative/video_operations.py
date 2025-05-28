import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import queue
import threading
import time

class VideoOperations:
    def __init__(self, app_instance):
        self.app = app_instance

    def check_conflict(self, action):
        # ... (Copy from original file's check_conflict method)
        if action == 'upload_video':
            if self.app.playing:
                messagebox.showwarning("操作冲突", "视频正在播放中，请暂停或停止后再上传视频。")
                return True
            if self.app.virtual_cam_playing:
                messagebox.showwarning("操作冲突", "摄像头正在连接中，请停止摄像头后再上传视频。")
                return True
        elif action == 'connect_camera':
            if self.app.playing:
                messagebox.showwarning("操作冲突", "视频正在播放中，请暂停或停止后再连接摄像头。")
                return True
            if self.app.virtual_cam_playing:
                messagebox.showwarning("操作冲突", "摄像头已连接。")
                return True
        elif action in ['play_pause', 'stop']:
            if self.app.virtual_cam_playing:
                messagebox.showwarning("操作冲突", "摄像头正在连接中，请停止摄像头后再操作视频播放。")
                return True
        elif action == 'disconnect_camera':
            if not self.app.virtual_cam_playing:
                messagebox.showwarning("操作冲突", "摄像头未连接。")
                return True
        return False

    def upload_video(self):
        # ... (Copy from original file's upload_video method)
        if self.check_conflict('upload_video'):
            return
    
        if self.app.check_processing():
            return
    
        self.app.current_vehicle_id = 0 
        self.app.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv *.flv ")])
        if not self.app.video_path:
            return
    
        if self.app.cap_1 is not None:
            if self.app.cap_1.isOpened():
                self.app.cap_1.release()
            self.app.cap_1 = None
    
        cap = cv2.VideoCapture(self.app.video_path)
        if not cap.isOpened():
            messagebox.showerror("错误", "无法打开视频。")
            return
    
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
        max_resolution = 1280
        if video_width > max_resolution:
            scale_factor = max_resolution / video_width
            new_width = max_resolution
            new_height = int(video_height * scale_factor)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, new_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, new_height)
    
        self.app.fps = cap.get(cv2.CAP_PROP_FPS)
        self.app.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.app.y_max = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.app.line_y = int(self.app.y_max * 0.5)
        cap.release()
    
        with self.app.stats_lock:
            self.app.tracked_data = self.app.pd.DataFrame(columns=['frame_num', 'id', 'class_name', 'x1', 'y1', 'x2', 'y2', 'speed'])
            self.app.speed_stats = []
            self.app.flow_history = []
            self.app.flow_counter = 0
            self.app.current_flow_rate = 0
            self.app.vehicle_positions = {}
            self.app.collision_records = {}
            self.app.trajectory_records = set()
            self.app.speed_history = {}
            self.app.trajectory_history = {}
            self.app.processed_objects = {}
            self.app.vehicle_type_counts = {key: 0 for key in self.app.VEHICLE_TYPE_DISTANCE.keys()}
            self.app.trajectory_exception_records = set()
            self.app.counted_vehicle_ids = set()
            self.app.window_start_time = time.time()
    
        cap = cv2.VideoCapture(self.app.video_path)
        if not cap.isOpened():
            messagebox.showerror("错误", "无法打开视频。")
            return
    
        ret, frame = cap.read()
        if ret:
            self.app.display_frame(frame, self.app.canvas_original)
            self.app.display_frame(frame, self.app.canvas_processed)
            with self.app.frame_lock:
                self.app.latest_frame = frame.copy()
    
            try:
                self.app.raw_frame_queue.put_nowait((frame.copy(), self.app.current_frame))
            except queue.Full:
                try:
                    self.app.raw_frame_queue.get_nowait()
                    self.app.raw_frame_queue.put_nowait((frame.copy(), self.app.current_frame))
                except queue.Empty:
                    pass
        cap.release()
        self.app.cap_1 = None
        self.app.playing = False
        self.app.current_frame = 0
        self.app.update_status_label()

    def toggle_play_pause(self):
        # ... (Copy from original file's toggle_play_pause method)
        if self.check_conflict('play_pause'):
            return
    
        if self.app.playing:
            self.app.playing = False
            print("暂停播放。")
            self.app.update_status_label()
        else:
            if not self.app.video_path and not self.app.virtual_cam_playing:
                messagebox.showerror("错误", "请先上传至少一个视频或连接摄像头。")
                return
    
            if self.app.current_frame >= self.app.total_frames:
                self.app.current_frame = 0
                print("视频已播放完毕，重新开始播放")
    
            self.app.playing = True
            self.play_video()

    def stop_playback(self):
        # ... (Copy from original file's stop_playback method)
        if self.check_conflict('stop'):
            return

        if self.app.playing:
            self.app.playing = False
            print("停止播放。")

        if self.app.cap_1 is not None:
            if self.app.cap_1.isOpened():
                self.app.cap_1.release()
            self.app.cap_1 = None

        if self.app.video_path:
            cap = cv2.VideoCapture(self.app.video_path)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    self.app.display_frame(frame, self.app.canvas_original)
                    self.app.display_frame(frame, self.app.canvas_processed)
                    with self.app.frame_lock:
                        self.app.latest_frame = frame.copy()
            cap.release()

            with self.app.raw_frame_queue.mutex:
                self.app.raw_frame_queue.queue.clear()
            with self.app.processed_frame_queue.mutex:
                self.app.processed_frame_queue.queue.clear()

            self.app.update_status_label()
            self.app.current_frame = 0

    def play_video(self):
        # ... (Copy from original file's play_video method)
        if not self.app.playing:
            return

        if self.app.cap_1 is None:
            self.app.cap_1 = cv2.VideoCapture(self.app.video_path)
            if not self.app.cap_1.isOpened():
                messagebox.showerror("错误", "无法打开视频。")
                self.app.cap_1 = None
                self.app.playing = False
                return
            self.app.current_vehicle_id = 0 
            self.app.current_frame = 0 
            
        ret, frame = self.app.cap_1.read()
        if ret:
            self.app.display_frame(frame, self.app.canvas_original)
            try:
                self.app.raw_frame_queue.put_nowait((frame.copy(), self.app.current_frame))
            except queue.Full:
                try:
                    self.app.raw_frame_queue.get_nowait()
                    self.app.raw_frame_queue.put_nowait((frame.copy(), self.app.current_frame))
                except queue.Empty:
                    pass

            self.app.current_frame += 1
            self.app.update_status_label()

            if self.app.current_frame < self.app.total_frames:
                self.app.root.after(int(1000 / self.app.fps), self.play_video)
            else:
                self.app.playing = False
                self.app.cap_1.release()
                print("视频播放完毕。")
        else:
            self.app.playing = False
            self.app.cap_1.release()
            print("视频播放完毕。")

    def toggle_virtual_camera(self):
        # ... (Copy from original file's toggle_virtual_camera method)
        if not self.app.virtual_cam_playing:
            if self.check_conflict('connect_camera'):
                return
            try:
                self.app.virtual_cam_cap = cv2.VideoCapture(0)
    
                self.app.virtual_cam_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.app.virtual_cam_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.app.virtual_cam_cap.set(cv2.CAP_PROP_FPS, 30)
    
                if not self.app.virtual_cam_cap.isOpened():
                    messagebox.showerror("错误", "无法打开摄像头。")
                    return
    
                self.app.current_vehicle_id = 0
                self.app.counted_vehicle_ids = set()
                self.app.virtual_cam_playing = True
                self.app.video_menu.entryconfig(1, label="停止摄像头")
                self.app.button_connect_camera.config(text="停止摄像头")
                self.app.status_label.config(text="正在使用摄像头")
                print("摄像头已连接。")
    
                with self.app.stats_lock:
                    self.app.prev_info = {}
                    self.app.speed_history = {}
                    self.app.trajectory_history = {}
    
                self.app.raw_frame_queue = queue.Queue(maxsize=50)
                self.app.processed_frame_queue = queue.Queue(maxsize=50)
    
                self.app.virtual_cam_thread = threading.Thread(target=self.play_virtual_camera, daemon=True)
                self.app.virtual_cam_thread.start()
    
            except Exception as e:
                print(f"连接摄像头时发生错误: {e}")
                messagebox.showerror("错误", f"连接摄像头失败：{e}")
        else:
            self.app.virtual_cam_playing = False
            if self.app.virtual_cam_cap:
                self.app.virtual_cam_cap.release()
            self.app.video_menu.entryconfig(1, label="连接摄像头")
            self.app.button_connect_camera.config(text="连接摄像头")
            self.app.update_status_label()
            print("摄像头已停止。")
    
            self.app.current_vehicle_id = 0

    def play_virtual_camera(self):
        # ... (Copy from original file's play_virtual_camera method)
        while self.app.virtual_cam_playing and self.app.virtual_cam_cap.isOpened():
            ret, frame = self.app.virtual_cam_cap.read()
            if not ret:
                continue

            frame = cv2.resize(frame, (1280, 720))

            self.app.display_frame(frame, self.app.canvas_original)
            try:
                with self.app.frame_lock:
                    self.app.current_frame += 1
                    frame_num = self.app.current_frame

                if self.app.raw_frame_queue.full():
                    try:
                        self.app.raw_frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.app.raw_frame_queue.put_nowait((frame.copy(), frame_num))
            except Exception as e:
                print(f"帧队列操作错误: {e}")

            time.sleep(1 / 60)

        if self.app.virtual_cam_cap:
            self.app.virtual_cam_cap.release()
        self.app.virtual_cam_playing = False
        self.app.button_connect_camera.config(text="连接摄像头")
        self.app.update_status_label() 