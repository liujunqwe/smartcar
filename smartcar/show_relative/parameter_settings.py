import tkinter as tk
from tkinter import Toplevel, Frame, Button, Label, Entry, messagebox

class ParameterSettings:
    def __init__(self, app_instance):
        self.app = app_instance

    def open_parameter_settings(self):
        # ... (Copy from original file's open_parameter_settings method)
        params_window = Toplevel(self.app.root)
        params_window.title("参数设置")
        params_window.geometry("600x750")
        params_window.resizable(False, False)
        
        title_font = ("", 12, "bold")
        desc_font = ("", 12)
        entry_width = 25
        section_pady = 15
        
        def create_param_section(parent, title, description, default_value):
            frame = Frame(parent)
            frame.pack(pady=section_pady, fill="x")
            Label(frame, text=title, font=title_font).pack(anchor="center", pady=(0, 3))
            Label(frame, text=description, font=desc_font, wraplength=550, justify="center").pack(anchor="center")
            entry = Entry(frame, width=entry_width)
            entry.pack(pady=(8, 0))
            entry.insert(0, str(default_value))
            return entry
        
        self.app.entry_lanes = create_param_section(
            params_window,
            "车道数 (1-10)",
            "（设置视频中的车道数量，用于计算车流量）",
            self.app.number_of_lanes
        )
        
        self.app.entry_speed = create_param_section(
            params_window,
            "速度阈值 (1-200 km/h)",
            "（设置检测车辆的速度阈值，超过这个值会被判定为超速异常）",
            self.app.speed_threshold
        )
        
        self.app.entry_angle = create_param_section(
            params_window,
            "轨迹角度阈值 (1-180 度)",
            "（设置车辆轨迹角度，变化超过此值时触发异常检测）",
            self.app.angle_threshold
        )
        
        self.app.entry_conf = create_param_section(
            params_window,
            "置信度阈值 (0.0-1.0)",
            "（设置检测框的最低置信度，预测概率高于此值的框才会显示）",
            self.app.conf_threshold
        )
        
        self.app.entry_iou = create_param_section(
            params_window,
            "IoU阈值 (0.0-1.0)",
            "（设置检测框的最小交并比（IoU），用于碰撞检测，框的重合度大于这个值才会被视为有碰撞异常）",
            self.app.iou_threshold
        )
        
        Button(params_window,
              text="确 定",
              command=lambda: self.set_parameters(params_window),
              font=("", 12),
              width=10).pack(pady=(15, 20))

    def set_parameters(self, window):
        # ... (Copy from original file's set_parameters method)
        lanes = self.app.entry_lanes.get()
        speed = self.app.entry_speed.get()
        angle = self.app.entry_angle.get()
        conf = self.app.entry_conf.get()
        iou = self.app.entry_iou.get()

        try:
            lanes = int(lanes)
            if not (1 <= lanes <= 10):
                raise ValueError
        except ValueError:
            messagebox.showerror("输入错误", "车道数必须是1到10之间的整数。")
            return

        try:
            speed = float(speed)
            if not (1 <= speed <= 200):
                raise ValueError
        except ValueError:
            messagebox.showerror("输入错误", "速度阈值必须是1到200之间的数字。")
            return

        try:
            angle = float(angle)
            if not (1 <= angle <= 180):
                raise ValueError
        except ValueError:
            messagebox.showerror("输入错误", "轨迹角度阈值必须是1到180之间的数字。")
            return

        try:
            conf = float(conf)
            if not (0.0 <= conf <= 1.0):
                raise ValueError
        except ValueError:
            messagebox.showerror("输入错误", "置信度阈值必须是0.0到1.0之间的数字。")
            return

        try:
            iou = float(iou)
            if not (0.0 <= iou <= 1.0):
                raise ValueError
        except ValueError:
            messagebox.showerror("输入错误", "IoU阈值必须是0.0到1.0之间的数字。")
            return

        with self.app.stats_lock:
            self.app.number_of_lanes = lanes
            self.app.speed_threshold = speed
            self.app.angle_threshold = angle
            self.app.conf_threshold = conf
            self.app.iou_threshold = iou

        print(f"参数已设置：车道数={self.app.number_of_lanes}, 速度阈值={self.app.speed_threshold} km/h, "
              f"轨迹角度阈值={self.app.angle_threshold} 度, 置信度阈值={self.app.conf_threshold}, "
              f"IoU阈值={self.app.iou_threshold}")
        messagebox.showinfo("成功",
                            f"参数已设置：\n车道数={self.app.number_of_lanes}\n速度阈值={self.app.speed_threshold} km/h\n"
                            f"轨迹角度阈值={self.app.angle_threshold} 度\n置信度阈值={self.app.conf_threshold}\n"
                            f"IoU阈值={self.app.iou_threshold}")
        window.destroy() 