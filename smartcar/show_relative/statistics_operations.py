import tkinter as tk
from tkinter import messagebox, Toplevel, Frame, Button
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class StatisticsOperations:
    def __init__(self, app_instance):
        self.app = app_instance

    def show_traffic_statistics(self):
        # ... (Copy from original file's show_traffic_statistics method)
        with self.app.stats_lock:
            if self.app.processing:
                messagebox.showinfo("处理中", "视频正在处理，请稍后再操作。")
                return

            if not self.app.flow_history:
                messagebox.showerror("错误", "暂无车流量统计数据。请先处理视频。")
                return
            flow_per_window = self.app.flow_history.copy()

        stats_window = Toplevel(self.app.root)
        stats_window.title("车流量统计")
        stats_window.geometry("800x600")
        button_frame = Frame(stats_window)
        button_frame.pack(side=tk.TOP, fill=tk.X)
        chart_frame = Frame(stats_window)
        chart_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.app.traffic_current_chart_type = tk.StringVar(value="柱状图")

        def plot_chart(chart_type):
            self.app.traffic_current_chart_type.set(chart_type)
            self.update_traffic_flow_chart(fig, ax, chart_type)
            canvas.draw()

        button_bar = Button(button_frame, text="柱状图", command=lambda: plot_chart("柱状图"))
        button_bar.pack(side=tk.LEFT, padx=5, pady=5)
        button_horizontal_bar = Button(button_frame, text="条形图", command=lambda: plot_chart("条形图"))
        button_horizontal_bar.pack(side=tk.LEFT, padx=5, pady=5)
        button_line = Button(button_frame, text="折线图", command=lambda: plot_chart("折线图"))
        button_line.pack(side=tk.LEFT, padx=5, pady=5)

        fig, ax = plt.subplots(figsize=(8, 6))
        self.update_traffic_flow_chart(fig, ax, self.app.traffic_current_chart_type.get())
        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_traffic_flow_chart(self, fig, ax, chart_type):
        # ... (Copy from original file's update_traffic_flow_chart method)
        ax.clear()
        flow_per_window = self.app.flow_history.copy()
        num_windows = len(flow_per_window)
        windows = range(1, num_windows + 1)
    
        if chart_type == "柱状图":
            ax.bar(windows, flow_per_window, color='skyblue', align='center')
            ax.set_title('车流量统计 (veh/h) - 柱状图')
            ax.set_xlabel(f'时间窗口 (每{self.app.flow_window_size_seconds}秒)')
            ax.set_ylabel('车流量 (veh/h)')
            for i, v in enumerate(flow_per_window):
                ax.text(windows[i], v + 0.1, f'{v:.2f}', ha='center', va='bottom')
        elif chart_type == "条形图":
            ax.barh(windows, flow_per_window, color='skyblue', align='center')
            ax.set_title('车流量统计 (veh/h) - 条形图')
            ax.set_xlabel('车流量 (veh/h)')
            ax.set_ylabel(f'时间窗口 (每{self.app.flow_window_size_seconds}秒)')
            for i, v in enumerate(flow_per_window):
                ax.text(v + 0.1, windows[i], f'{v:.2f}', ha='left', va='center')
        elif chart_type == "折线图":
            ax.plot(windows, flow_per_window, marker='o', linestyle='-', color='green')
            ax.set_title('车流量统计 (veh/h) - 折线图')
            ax.set_xlabel(f'时间窗口 (每{self.app.flow_window_size_seconds}秒)')
            ax.set_ylabel('车流量 (veh/h)')
            for i, v in enumerate(flow_per_window):
                ax.text(windows[i], v + 0.1, f'{v:.2f}', ha='center', va='bottom')
    
        if chart_type == "条形图":
            ax.set_xlim(0, max(flow_per_window) * 1.1 if flow_per_window else 1)
            ax.set_ylim(0.5, num_windows + 0.5)
        else:
            ax.set_xlim(0.5, num_windows + 0.5)
            ax.set_ylim(0, max(flow_per_window) * 1.1 if flow_per_window else 1)
    
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        fig.tight_layout()

    def show_speed_statistics(self):
        # ... (Copy from original file's show_speed_statistics method)
        with self.app.stats_lock:
            if self.app.processing:
                messagebox.showinfo("处理中", "视频正在处理，请稍后再操作。")
                return
    
            if not self.app.speed_stats:
                messagebox.showerror("错误", "暂无速度统计数据。请先处理视频。")
                return
    
            filtered_speeds = [speed for speed in self.app.speed_stats if speed > 0]
            total = len(filtered_speeds)
    
            if total == 0:
                messagebox.showerror("错误", "所有速度记录均为0。")
                return
    
        stats_window = Toplevel(self.app.root)
        stats_window.title("速度统计")
        stats_window.geometry("800x600")
    
        fig, ax = plt.subplots(figsize=(8, 6))
    
        counts, bins = np.histogram(filtered_speeds, bins=20)
        percentages = (counts / total) * 100
    
        ax.bar(bins[:-1], percentages, width=np.diff(bins), align='edge', color='green', edgecolor='black')
        ax.set_title('车辆速度分布（百分比）')
        ax.set_xlabel('速度 (km/h)')
        ax.set_ylabel('百分比 (%)')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}%'))
    
        for i, v in enumerate(percentages):
            ax.text(bins[i] + (bins[i + 1] - bins[i]) / 2, v + 0.5, f'{v:.2f}%', ha='center', va='bottom')
    
        plt.tight_layout()
    
        canvas = FigureCanvasTkAgg(fig, master=stats_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def show_vehicle_type_statistics(self):
        # ... (Copy from original file's show_vehicle_type_statistics method)
        with self.app.stats_lock:
            if self.app.processing:
                messagebox.showinfo("处理中", "视频正在处理，请稍后再操作。")
                return
    
            if not self.app.vehicle_type_counts:
                messagebox.showerror("错误", "暂无车型统计数据。请先处理视频。")
                return
    
        stats_window = Toplevel(self.app.root)
        stats_window.title("车型统计")
        stats_window.geometry("800x600")
    
        chart_frame = Frame(stats_window)
        chart_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
        fig, ax = plt.subplots(figsize=(8, 6))
        types = list(self.app.vehicle_type_counts.keys())
        counts = list(self.app.vehicle_type_counts.values())
        ax.bar(types, counts, color='orange', align='center')
        ax.set_title('车型统计 - 柱状图')
        ax.set_xlabel('车型')
        ax.set_ylabel('数量')
        ax.set_ylim(0, max(counts) * 1.1 if counts else 1)
    
        for i, v in enumerate(counts):
            ax.text(i, v + 0.1, f'{v}', ha='center', va='bottom')
    
        fig.tight_layout()
        ax.set_xlim(-0.5, len(types) - 0.5)
    
        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True) 