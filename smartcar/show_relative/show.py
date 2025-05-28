import threading 
import tkinter as tk
from tkinter import scrolledtext, Menu, Frame, Button, Label, Canvas, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
import sys
import warnings
import queue
from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
import pandas as pd
import time
from urllib3.exceptions import SystemTimeWarning

from constants import VEHICLE_TYPE_DISTANCE, DEFAULT_DISTANCE, y_to_depth
from video_operations import VideoOperations
from processing_operations import ProcessingOperations
from statistics_operations import StatisticsOperations
from parameter_settings import ParameterSettings

# 忽略 libpng 的 iCCP 警告
warnings.filterwarnings("ignore", message=".*iCCP: known incorrect sRGB profile.*", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=SystemTimeWarning)

plt.rcParams['font.sans-serif']=['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

class TextRedirector:
    def __init__(self, text_widget, original_stream):
        self.text_widget = text_widget
        self.original_stream = original_stream

    def write(self, str_val):
        self.text_widget.configure(state='normal')
        self.text_widget.insert(tk.END, str_val)
        self.text_widget.configure(state='disabled')
        self.text_widget.see(tk.END)
        self.original_stream.write(str_val)

    def flush(self):
        pass

class VideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("智慧道路车辆跟踪监测系统")
        self.last_detection_time = 0
        self.pd = pd # Make pandas accessible
        self.queue = queue # Make queue accessible
        self.VEHICLE_TYPE_DISTANCE = VEHICLE_TYPE_DISTANCE # Make constant accessible

        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.9)
        x_pos = (screen_width - window_width) // 2
        y_pos = (screen_height - window_height - 80) // 2
        root.geometry(f"{window_width}x{window_height}+{x_pos}+{y_pos}")

        self.main_frame = tk.Frame(root, bg='white')
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.video_ops = VideoOperations(self)
        self.processing_ops = ProcessingOperations(self)
        self.stats_ops = StatisticsOperations(self)
        self.params_ops = ParameterSettings(self)

        menu_bar = tk.Menu(root)
        root.config(menu=menu_bar)

        self.video_menu = tk.Menu(menu_bar, tearoff=0)
        self.video_menu.add_command(label="上传视频", command=self.video_ops.upload_video)
        self.video_menu.add_command(label="连接摄像头", command=self.video_ops.toggle_virtual_camera)
        self.video_menu.add_command(label="播放/暂停", command=self.video_ops.toggle_play_pause)
        self.video_menu.add_command(label="停止", command=self.video_ops.stop_playback)

        self.process_menu = tk.Menu(menu_bar, tearoff=0)
        self.process_menu.add_command(label="车辆检测", command=self.processing_ops.toggle_vehicle_detection)
        self.process_menu.add_command(label="速度检测", command=self.processing_ops.toggle_speed_detection)
        self.process_menu.add_command(label="异常检测", command=self.processing_ops.toggle_exception_detection)
        self.stats_menu = tk.Menu(menu_bar, tearoff=0)
        self.stats_menu.add_command(label="显示车流量统计", command=self.stats_ops.show_traffic_statistics)
        self.stats_menu.add_command(label="显示速度统计", command=self.stats_ops.show_speed_statistics)
        self.stats_menu.add_command(label="显示车型统计", command=self.stats_ops.show_vehicle_type_statistics)

        menu_bar.add_cascade(label="视频源", menu=self.video_menu)
        menu_bar.add_cascade(label="视频处理", menu=self.process_menu)
        menu_bar.add_cascade(label="统计数据", menu=self.stats_menu)

        self.title_frame = tk.Frame(self.main_frame, bg='white')
        self.title_frame.pack(side=tk.TOP, fill=tk.X, pady=10)
        self.title_label = tk.Label(self.title_frame, text="智慧道路车辆跟踪监测系统", font=("SimHei", 20, "bold"), bg='white')
        self.title_label.pack()

        self.control_frame = tk.Frame(self.main_frame, bg='white')
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        self.status_label = tk.Label(self.control_frame, text="未加载视频", width=20, height=2)
        self.status_label.grid(row=0, column=1, padx=5, pady=5)
        self.back_button = tk.Button(self.control_frame, text="返回", command=self.go_back, width=15, height=2)
        self.back_button.grid(row=0, column=0, padx=5, pady=5)

        self.console_text = scrolledtext.ScrolledText(self.control_frame, width=60, height=30, state='disabled', bg='black', fg='white', wrap='word')
        self.console_text.grid(row=3, column=0, columnspan=2, padx=5, pady=10, sticky='nsew')
        sys.stdout = TextRedirector(self.console_text, sys.stdout)
        sys.stderr = TextRedirector(self.console_text, sys.stderr)

        self.frame0 = tk.Frame(self.control_frame)
        self.frame1 = tk.Frame(self.control_frame)
        self.frame2 = tk.Frame(self.control_frame)
        self.frame3 = tk.Frame(self.control_frame)

        self.is_vehicle_processing = False
        self.is_speed_processing = False
        self.is_exception_processing = False
        self.fps = 30

        self.init_models()
        self.create_frame0()

        self.video_frame = tk.Frame(self.main_frame, bg='white')
        self.video_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.canvas_width = int(window_width * 0.5)
        self.canvas_height = int(window_height * 0.42)

        self.canvas_original = tk.Canvas(self.video_frame, bg='white', width=self.canvas_width, height=self.canvas_height)
        self.canvas_original.pack(side=tk.TOP, padx=10, pady=10, expand=True)
        self.canvas_processed = tk.Canvas(self.video_frame, bg='white', width=self.canvas_width, height=self.canvas_height)
        self.canvas_processed.pack(side=tk.BOTTOM, padx=10, pady=10, expand=True)

        self.image_id_original = None
        self.image_tk_original = None
        self.image_id_processed = None
        self.image_tk_processed = None
        self.video_path = None
        self.traffic_flow = 0
        self.speed_stats = []
        self.flow_history = []
        self.flow_counter = 0
        self.current_flow_rate = 0
        self.vehicle_positions = {}
        self.line_y = None
        self.y_min = 0
        self.y_max = None
        self.vehicle_type_counts = {}
        self.playing = False
        self.cap_1 = None
        self.delay = 1
        self.virtual_cam_cap = None
        self.virtual_cam_playing = False
        self.virtual_cam_thread = None
        self.virtual_cam_lock = threading.Lock()
        self.processing = False
        self.stats_lock = threading.Lock()
        self.current_frame = 0
        self.total_frames = 0
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.raw_frame_queue = queue.Queue(maxsize=20)
        self.processed_frame_queue = queue.Queue(maxsize=20)
        self.tracked_data = pd.DataFrame(columns=['frame_num', 'id', 'class_name', 'x1', 'y1', 'x2', 'y2', 'speed'])
        self.prev_info = {}
        self.window_start_time = time.time()
        self.speed_history = {}
        self.alpha = 0.3
        self.trajectory_history = {}
        self.trajectory_exception_records = set()
        self.collision_records = {}
        self.current_vehicle_id = 0 # Added for process_frames logic

        self.processing_thread = threading.Thread(target=self.processing_ops.process_frames, daemon=True)
        self.processing_thread.start()
        self.root.after(30, self.update_processed_canvas)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_frame0(self):
        for widget in self.frame0.winfo_children():
            widget.destroy()
        button_video_source = tk.Button(self.frame0, text="视频源", width=15, height=2, command=self.show_video_frame)
        button_video_source.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        button_process = tk.Button(self.frame0, text="视频处理", width=15, height=2, command=self.show_process_frame)
        button_process.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        button_stats = tk.Button(self.frame0, text="统计数据", width=15, height=2, command=self.show_stats_frame)
        button_stats.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        button_params = tk.Button(self.frame0, text="参数设置", width=15, height=2, command=self.params_ops.open_parameter_settings)
        button_params.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.frame0.grid(row=2, column=0, columnspan=2)

    def show_video_frame(self):
        self.hide_all_frames()
        self.create_video_frame()
        self.frame1.grid(row=2, column=0, columnspan=2)

    def create_video_frame(self):
        for widget in self.frame1.winfo_children():
            widget.destroy()
        self.button_upload_video = tk.Button(self.frame1, text="上传视频", command=self.video_ops.upload_video, width=15, height=2)
        self.button_upload_video.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        camera_button_text = "停止摄像头" if self.virtual_cam_playing else "连接摄像头"
        self.button_connect_camera = tk.Button(self.frame1, text=camera_button_text, command=self.video_ops.toggle_virtual_camera, width=15, height=2)
        self.button_connect_camera.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.button_play = tk.Button(self.frame1, text="播放/暂停", command=self.video_ops.toggle_play_pause, width=15, height=2)
        self.button_play.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        self.button_stop = tk.Button(self.frame1, text="停止", command=self.video_ops.stop_playback, width=15, height=2)
        self.button_stop.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

    def show_process_frame(self):
        self.hide_all_frames()
        self.create_process_frame()
        self.frame2.grid(row=2, column=0, columnspan=2)

    def create_process_frame(self):
        for widget in self.frame2.winfo_children():
            widget.destroy()
        vehicle_text = "停止车辆检测" if self.is_vehicle_processing else "开始车辆检测"
        self.button_vehicle_detection = tk.Button(self.frame2, text=vehicle_text, command=self.processing_ops.toggle_vehicle_detection, width=15, height=2)
        self.button_vehicle_detection.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        speed_text = "停止速度检测" if self.is_speed_processing else "开始速度检测"
        self.button_speed_detection = tk.Button(self.frame2, text=speed_text, command=self.processing_ops.toggle_speed_detection, width=15, height=2)
        self.button_speed_detection.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        exception_text = "停止异常检测" if self.is_exception_processing else "开始异常检测"
        self.button_exception_detection = tk.Button(self.frame2, text=exception_text, command=self.processing_ops.toggle_exception_detection, width=15, height=2)
        self.button_exception_detection.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

    def show_stats_frame(self):
        self.hide_all_frames()
        self.create_stats_frame()
        self.frame3.grid(row=2, column=0, columnspan=2)

    def create_stats_frame(self):
        for widget in self.frame3.winfo_children():
            widget.destroy()
        self.button_traffic_stats = tk.Button(self.frame3, text="显示车流量统计", command=self.stats_ops.show_traffic_statistics, width=15, height=2)
        self.button_traffic_stats.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.button_speed_stats = tk.Button(self.frame3, text="显示速度统计", command=self.stats_ops.show_speed_statistics, width=15, height=2)
        self.button_speed_stats.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.button_vehicle_type_stats = tk.Button(self.frame3, text="显示车型统计", command=self.stats_ops.show_vehicle_type_statistics, width=15, height=2)
        self.button_vehicle_type_stats.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

    def hide_all_frames(self):
        self.frame0.grid_forget()
        self.frame1.grid_forget()
        self.frame2.grid_forget()
        self.frame3.grid_forget()

    def go_back(self):
        self.hide_all_frames()
        self.create_frame0()
        self.frame0.grid(row=2, column=0, columnspan=2)

    def on_closing(self):
        self.playing = False
        self.virtual_cam_playing = False
        if self.cap_1 is not None and self.cap_1.isOpened():
            self.cap_1.release()
        if self.virtual_cam_cap is not None and self.virtual_cam_cap.isOpened():
            self.virtual_cam_cap.release()
        self.root.destroy()

    def init_models(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            print("加载 YOLO 模型用于车辆检测...")
            self.vehicle_model = YOLO("model_data/best.pt")
            self.vehicle_model.to(self.device)
            print("YOLO 模型加载成功。")
        except Exception as e:
            print(f"加载 YOLO 模型时发生错误: {e}")
            messagebox.showerror("错误", f"加载 YOLO 模型失败：{e}")

        self.traffic_flow = 0
        self.flow_counter = 0
        self.current_flow_rate = 0
        self.flow_window_size_seconds = 5
        self.flow_history = []
        self.vehicle_positions = {}
        self.line_y = None
        self.y_min = 0
        self.y_max = None
        self.depth_min = 5.0
        self.depth_max = 80.0
        self.speed_stats = []
        self.vehicle_type_counts = {key: 0 for key in VEHICLE_TYPE_DISTANCE.keys()}
        self.counted_vehicle_ids = set()
        self.collision_records = {}
        self.trajectory_records = set()
        self.speed_history = {}
        self.alpha = 0.3
        self.trajectory_history = {}
        self.trajectory_exception_records = set()
        self.min_displacement = 10
        self.conf_threshold = 0.3
        self.iou_threshold = 0.1
        self.speed_threshold = 60.0
        self.number_of_lanes = 4
        self.angle_threshold = 30.0

    def update_processed_canvas(self):
        frame = None
        try:
            while not self.processed_frame_queue.empty():
                frame = self.processed_frame_queue.get_nowait()
        except queue.Empty:
            pass
        if frame is not None:
            self.display_frame(frame, self.canvas_processed)
        self.root.after(30, self.update_processed_canvas)

    def display_frame(self, frame, canvas):
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            canvas_width = self.canvas_width
            canvas_height = self.canvas_height
            img_width, img_height = img.size
            ratio = min(canvas_width / img_width, canvas_height / img_height)
            new_size = (int(img_width * ratio), int(img_height * ratio))
            img = img.resize(new_size, Image.LANCZOS)
            x = (canvas_width - new_size[0]) // 2
            y = (canvas_height - new_size[1]) // 2
            img_tk = ImageTk.PhotoImage(image=img)

            if canvas == self.canvas_original:
                if self.image_id_original is None:
                    self.image_id_original = canvas.create_image(x, y, anchor=tk.NW, image=img_tk)
                else:
                    canvas.coords(self.image_id_original, x, y)
                    canvas.itemconfig(self.image_id_original, image=img_tk)
                self.image_tk_original = img_tk
            elif canvas == self.canvas_processed:
                if self.image_id_processed is None:
                    self.image_id_processed = canvas.create_image(x, y, anchor=tk.NW, image=img_tk)
                else:
                    canvas.coords(self.image_id_processed, x, y)
                    canvas.itemconfig(self.image_id_processed, image=img_tk)
                self.image_tk_processed = img_tk
        except Exception as e:
            print(f"显示帧时发生错误: {e}")

    def update_status_label(self):
        if self.virtual_cam_playing:
            self.status_label.config(text="正在使用摄像头")
        elif self.video_path:
            if self.total_frames > 0:
                self.status_label.config(text=f"当前帧: {self.current_frame} / {self.total_frames}")
            else:
                self.status_label.config(text=f"当前帧: {self.current_frame}")
        else:
            self.status_label.config(text="未加载视频")

    def check_processing(self):
        with self.stats_lock:
            if self.processing:
                messagebox.showinfo("处理中", "视频正在处理，请稍后再操作。")
                return True
        return False

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoApp(root)
    root.mainloop() 