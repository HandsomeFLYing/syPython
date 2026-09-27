# -*- coding: utf-8 -*-
"""
分贝记录仪 (Decibel Meter)
================================
- 选择麦克风输入设备，实时采集并显示声音高低
- 大数字分贝显示 + 彩色量程条 + 峰值保持刻度
- 60 秒历史曲线（含参考阈值线）
- 统计信息：当前 / 3秒峰值 / 60秒平均 / 60秒最低 / 会话最高
- 日志记录（内存缓冲） + 导出 CSV + 导出图表 PNG

说明：读数基于麦克风灵敏度，为相对满幅度的分贝值（dBFS），
未经过声压级(SPL)校准，仅用于观察声音相对大小变化。
参考范围（dBFS）：安静 < -50 | 轻声 -50~-35 | 正常交谈 -35~-25
                | 大声 -25~-15 | 很吵 -15~-5 | 极响 > -5

运行方式：  python decibel_meter.py     （或双击“启动分贝记录仪.bat”）
依赖：     numpy, sounddevice, matplotlib（已装入项目虚拟环境 .venv）
"""

import csv
import queue
import time
from collections import deque
from datetime import datetime

import numpy as np
import sounddevice as sd

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.font_manager as fm
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# ---------- 中文字体设置（Windows） ----------
_available_fonts = {f.name for f in fm.fontManager.ttflist}
for _f in ("Microsoft YaHei", "SimHei", "Microsoft JhengHei"):
    if _f in _available_fonts:
        plt.rcParams["font.sans-serif"] = [_f, "Arial"]
        break
plt.rcParams["axes.unicode_minus"] = False

# ---------- 全局参数 ----------
GAUGE_MIN, GAUGE_MAX = -60.0, 0.0          # 量程条范围（dBFS）
POLL_MS = 100                              # 界面刷新周期
BLOCK_MS = 50                              # 音频块时长（回调粒度）
LOG_INTERVAL = 1.0                         # 日志落盘/缓冲间隔（秒）

# 可选的曲线记录时长（历史窗口）
HISTORY_OPTIONS = [("30秒", 30), ("1分钟", 60), ("3分钟", 180),
                   ("5分钟", 300), ("10分钟", 600)]
# 可选的最大记录总时长（到点自动停止），0 表示不限
DURATION_OPTIONS = [("不限", 0), ("1分钟", 60), ("5分钟", 300),
                    ("10分钟", 600), ("30分钟", 1800), ("1小时", 3600)]

# 曲线抽样（时间片聚合）方式
SAMPLE_OPTIONS = [("最高值", "max"), ("平均值", "avg"),
                  ("平均+最高/2", "avgmax")]

# 分贝区间 -> (颜色, 描述)
ZONES = [
    (-60.0, -50.0, "#1abc9c", "安静"),
    (-50.0, -35.0, "#2ecc71", "轻声"),
    (-35.0, -25.0, "#f1c40f", "正常交谈"),
    (-25.0, -15.0, "#f39c12", "大声"),
    (-15.0,  -5.0, "#e67e22", "很吵"),
    ( -5.0,   1.0, "#e74c3c", "极响"),
]

# 历史曲线上的参考阈值线（dBFS, 标签, 颜色）
THRESHOLDS = [(-50, "安静", "#1abc9c"), (-35, "轻声", "#2ecc71"),
              (-25, "交谈", "#f1c40f"), (-15, "大声", "#e67e22"),
              ( -5, "极响", "#e74c3c")]


def db_of_rms(rms):
    """RMS 幅度 -> 分贝(dBFS)，下限 -120 dB 避免除零"""
    if rms > 1e-12:
        return float(20.0 * np.log10(rms))
    return -120.0


def zone_of_db(db):
    """返回 (颜色, 描述)"""
    for lo, hi, color, name in ZONES:
        if lo <= db < hi:
            return color, name
    return "#95a5a6", "无信号"


class AudioEngine:
    """音频采集引擎：在后台线程回调中计算分贝，放入线程安全队列"""

    def __init__(self, device_id, samplerate):
        self.device_id = device_id
        self.samplerate = samplerate
        self.stream = None
        self.queue = queue.Queue()

    def _callback(self, indata, frames, time_info, status):
        if status:
            pass  # 忽略 underflow/overflow 状态，不影响显示
        samples = indata[:, 0]                     # 取单声道
        rms = float(np.sqrt(np.mean(samples ** 2)))
        db = db_of_rms(rms)
        self.queue.put((time.time(), db))

    def start(self):
        self.stream = sd.InputStream(
            device=self.device_id,
            samplerate=self.samplerate,
            channels=1,
            blocksize=int(self.samplerate * BLOCK_MS / 1000.0),
            dtype="float32",
            latency="high",
            callback=self._callback,
        )
        self.stream.start()

    def stop(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def is_running(self):
        return self.stream is not None


def list_input_devices():
    """列出可用的输入设备（按名称去重，保留 MME 项），返回 [(id, 名称), ...]"""
    devices = sd.query_devices()
    seen = {}
    order = []
    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] <= 0:
            continue
        name = dev["name"].strip()
        key = (name, dev["max_input_channels"])
        if key not in seen:
            seen[key] = idx
            order.append(key)
    result = []
    for key in order:
        idx = seen[key]
        dev = devices[idx]
        name = dev["name"].strip()
        if name.startswith("Microsoft ") and "Input" in name:
            name = "系统默认输入"
        result.append((idx, f"[{idx}] {name}"))
    return result


class DecibelMeterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("分贝记录仪 - 麦克风声音监测")
        self.root.geometry("920x700")
        self.root.minsize(840, 640)

        # 状态
        self.engine = None
        self.running = False
        self.window_secs = 60                       # 当前曲线/统计窗口（秒）
        self.max_duration = 0                       # 最大记录总时长（秒），0=不限
        self._make_history()
        self.start_time = time.time()
        self.session_max_db = -120.0
        self.session_min_db = 0.0
        self._last_chart_ts = 0.0                   # 图表重绘节流时间戳
        self.sample_mode = "max"                    # 曲线抽样方式：max/avg/avgmax
        self.log_rows = []            # 日志缓冲
        self.logging_enabled = False
        self._last_log_time = 0.0

        self._build_ui()
        self._refresh_device_list()
        self._poll()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _make_history(self):
        """按窗口时长重建历史缓冲。

        关键：麦克风回调每 BLOCK_MS(50ms) 产生一个样本，若按
        刷新周期(100ms)估算容量，数据会提前一半被挤出。
        容量必须按真实回调频率计算：窗口秒数 * 1000 / BLOCK_MS。
        """
        capacity = self.window_secs * 1000 // BLOCK_MS
        self.history = deque(maxlen=capacity)

    # ---------------------------------------------------------- UI 构建
    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}

        # 顶栏：设备选择 + 开始/停止 + 曲线时长 + 记录上限
        top = ttk.Frame(self.root)
        top.pack(fill="x", **pad)
        ttk.Label(top, text="麦克风:").pack(side="left")
        self.device_var = tk.StringVar()
        self.device_box = ttk.Combobox(top, textvariable=self.device_var,
                                       state="readonly", width=28)
        self.device_box.pack(side="left", padx=4)
        self.start_btn = ttk.Button(top, text="开始监测", command=self._toggle_run)
        self.start_btn.pack(side="left", padx=8)

        ttk.Label(top, text="曲线时长:").pack(side="left", padx=(8, 0))
        self.window_var = tk.StringVar()
        self.window_box = ttk.Combobox(top, textvariable=self.window_var,
                                       state="readonly", width=7,
                                       values=[o[0] for o in HISTORY_OPTIONS])
        self.window_box.current(1)   # 默认 1 分钟
        self.window_box.pack(side="left", padx=4)
        self.window_box.bind("<<ComboboxSelected>>", self._on_window_change)

        ttk.Label(top, text="记录上限:").pack(side="left", padx=(8, 0))
        self.duration_var = tk.StringVar()
        self.duration_box = ttk.Combobox(top, textvariable=self.duration_var,
                                         state="readonly", width=8,
                                         values=[o[0] for o in DURATION_OPTIONS])
        self.duration_box.current(0)   # 默认不限
        self.duration_box.pack(side="left", padx=4)

        # 抽样方式：仅影响曲线显示，监测中也可随时切换
        ttk.Label(top, text="抽样方式:").pack(side="left", padx=(8, 0))
        self.sample_var = tk.StringVar()
        self.sample_box = ttk.Combobox(top, textvariable=self.sample_var,
                                       state="readonly", width=9,
                                       values=[o[0] for o in SAMPLE_OPTIONS])
        self.sample_box.current(0)     # 默认最高值
        self.sample_box.pack(side="left", padx=4)
        self.sample_box.bind("<<ComboboxSelected>>", self._on_sample_change)

        # 第二行：日志选项
        bar2 = ttk.Frame(self.root)
        bar2.pack(fill="x", **pad)
        self.log_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar2, text="启用日志记录", variable=self.log_var,
                        command=self._toggle_logging).pack(side="left")
        ttk.Button(bar2, text="清空", command=self._clear_all).pack(side="left", padx=6)
        ttk.Button(bar2, text="保存日志(CSV)", command=self._save_log).pack(side="left", padx=6)
        ttk.Button(bar2, text="保存图表(PNG)", command=self._save_chart).pack(side="left", padx=6)
        self.status_var = tk.StringVar(value="就绪：请选择麦克风后点击开始")
        self.status_lab = tk.Label(bar2, textvariable=self.status_var,
                                   fg="#555555", bg="#f0f0f0")
        self.status_lab.pack(side="left", padx=12)

        # 中间：大数字 + 量程条
        meter = ttk.Frame(self.root)
        meter.pack(fill="x", **pad)
        self.db_label = tk.Label(meter, text="--.-", font=("Segoe UI", 46, "bold"),
                                 fg="#95a5a6")
        self.db_label.pack()
        self.desc_label = tk.Label(meter, text="等待信号", font=("Microsoft YaHei", 14),
                                   fg="#95a5a6")
        self.desc_label.pack()
        # 量程画布：上条=静态参考刻度（标尺），下条=当前音量电平条
        self.gauge = tk.Canvas(meter, height=126, highlightthickness=0)
        self.gauge.pack(fill="x", padx=12, pady=4)
        # 不绑定 <Configure>：窗口拖拽时的重绘风暴会加剧卡顿，
        # 由 10fps 的 _poll 每 100ms 自然重绘，拖拽结束 100ms 内即跟上。
        self._gauge_w = 0  # 记录上一次画布宽度

        # 统计面板
        stats = ttk.Frame(self.root)
        stats.pack(fill="x", **pad)
        self.stat_items = {}
        self.stat_captions = {}
        for key, text in [("cur", "当前"), ("peak", "峰值(3秒)"),
                          ("avg", "平均(1分钟)"), ("low", "最低(1分钟)"),
                          ("smax", "会话最大"), ("smin", "会话最小")]:
            cell = ttk.Frame(stats)
            cell.pack(side="left", expand=True, fill="x")
            cap = ttk.Label(cell, text=text, foreground="#777777")
            cap.pack()
            lab = ttk.Label(cell, text="--.- dB", font=("Segoe UI", 16, "bold"))
            lab.pack()
            self.stat_items[key] = lab
            self.stat_captions[key] = cap

        # 历史曲线
        fig_frame = ttk.Frame(self.root)
        fig_frame.pack(fill="both", expand=True, padx=8, pady=(4, 8))
        self.fig = Figure(figsize=(9, 3.2), dpi=100, facecolor="white")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_ylim(-70, 0)
        self.ax.set_xlim(self.window_secs, 0)
        self.ax.set_xlabel("时间（右侧为现在）")
        self.ax.set_ylabel("dBFS")
        self.ax.grid(True, linestyle="--", alpha=0.4)
        for lvl, lbl, col in THRESHOLDS:
            self.ax.axhline(lvl, color=col, linestyle=":", alpha=0.9)
            self.ax.text(1.5, lvl + 1.5, lbl, fontsize=8, color=col,
                         zorder=5,
                         bbox=dict(facecolor="white", alpha=0.75,
                                   edgecolor="none", pad=1))
        self.line, = self.ax.plot([], [], color="#2980b9", linewidth=1.4,
                                  zorder=2)
        self._set_chart_ticks()
        self.canvas = FigureCanvasTkAgg(self.fig, master=fig_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # 底部说明（随窗口宽度自动换行）
        self.tip_lab = tk.Label(self.root,
                                text="相对分贝值(dBFS)，未校准SPL；参考：安静<-50 | 轻声-50~-35 | 正常交谈-35~-25 | 大声-25~-15 | 很吵-15~-5 | 极响>-5",
                                fg="#888888", font=("Microsoft YaHei", 9),
                                justify="center", anchor="center")
        self.tip_lab.pack(side="bottom", fill="x", pady=4)
        self.tip_lab.bind(
            "<Configure>",
            lambda e: self.tip_lab.config(wraplength=e.width - 24))

    def _refresh_device_list(self):
        try:
            items = list_input_devices()
        except Exception as e:
            messagebox.showerror("错误", f"获取音频设备失败：\n{e}")
            self.root.destroy()
            return
        self.devices = items
        self.device_box["values"] = [n for _, n in items]
        if items:
            # 优先选中默认输入设备
            default_id = None
            try:
                default_id = sd.default.device[0]
            except Exception:
                default_id = None
            if default_id is not None:
                for did, name in items:
                    if did == default_id:
                        self.device_var.set(name)
                        break
            if not self.device_var.get():
                self.device_var.set(items[0][1])

    # ---------------------------------------------------------- 参数设置
    def _fmt_window(self, secs):
        if secs % 60 == 0:
            return f"{secs // 60}分钟" if secs >= 60 else f"{secs}秒"
        return f"{secs}秒"

    def _on_window_change(self, _event=None):
        """切换曲线记录时长：重建缓冲、更新坐标轴与统计标题"""
        if self.running:
            return
        label = self.window_var.get()
        secs = dict(HISTORY_OPTIONS)[label]
        self.window_secs = secs
        self._make_history()
        self.stat_captions["avg"].config(text=f"平均({self._fmt_window(secs)})")
        self.stat_captions["low"].config(text=f"最低({self._fmt_window(secs)})")
        self.ax.set_xlim(secs, 0)
        self._set_chart_ticks()
        self.canvas.draw_idle()
        self.status_var.set(f"曲线记录时长已设为 {label}")

    def _on_sample_change(self, _event=None):
        """切换抽样方式：仅重算显示曲线，全量原始数据保留"""
        label = self.sample_var.get()
        self.sample_mode = dict(SAMPLE_OPTIONS)[label]
        self._update_chart()
        self.status_var.set(f"曲线抽样方式：{label}")

    def _set_chart_ticks(self):
        w = self.window_secs
        step = 10 if w <= 60 else 60
        ticks = list(range(0, w + 1, step))
        if ticks[-1] != w:
            ticks.append(w)
        self.ax.set_xticks(ticks)
        self.ax.set_xticklabels(
            ["现在" if x == 0 else f"{x // 60}分{x % 60:02d}s前" if x >= 60
             else f"{x}s前" for x in ticks])

    def _fmt_elapsed(self, secs):
        m, s = divmod(int(secs), 60)
        return f"{m:02d}:{s:02d}"

    # ---------------------------------------------------------- 音频控制
    def _toggle_run(self):
        if self.running:
            self._stop()
        else:
            self._start()

    def _start(self):
        if not self.device_var.get():
            messagebox.showwarning("提示", "请先选择麦克风设备")
            return
        # 读取本次监测的参数设置
        new_window = dict(HISTORY_OPTIONS)[self.window_var.get()]
        self.max_duration = dict(DURATION_OPTIONS)[self.duration_var.get()]
        if new_window != self.window_secs:
            # 仅窗口变化时重建缓冲（重建会清空历史，属预期）；
            # 同窗口再次开始则保留上次数据，直到手动“清空”
            self.window_secs = new_window
            self._make_history()
            self.stat_captions["avg"].config(
                text=f"平均({self._fmt_window(self.window_secs)})")
            self.stat_captions["low"].config(
                text=f"最低({self._fmt_window(self.window_secs)})")
            self.ax.set_xlim(self.window_secs, 0)
            self._set_chart_ticks()

        did = int(self.device_var.get().split("]")[0].strip("["))
        try:
            dev = sd.query_devices(did)
            sr = int(dev["default_samplerate"]) or 44100
            self.engine = AudioEngine(did, sr)
            self.engine.start()
        except Exception as e:
            messagebox.showerror("启动失败",
                                 f"无法打开该麦克风，可能已被占用：\n{e}\n\n请换一个设备试试。")
            self.engine = None
            return
        self.running = True
        self.start_time = time.time()
        self.session_max_db = -120.0
        self.session_min_db = 0.0
        self.start_btn.config(text="停止监测")
        self.device_box.config(state="disabled")
        self.window_box.config(state="disabled")
        self.duration_box.config(state="disabled")
        self._update_status()
        self._set_status_color("#27ae60")

    def _stop(self, auto=False):
        """停止监测：不清空任何数据，界面保留最后一次读数供查看/保存"""
        if self.engine:
            self.engine.stop()
            self.engine = None
        self.running = False
        self.start_btn.config(text="开始监测")
        self.device_box.config(state="readonly")
        self.window_box.config(state="readonly")
        self.duration_box.config(state="readonly")
        if auto:
            self.status_var.set("已达到设定的记录上限，自动停止监测（数据保留）")
            self._set_status_color("#e67e22")
        else:
            self.status_var.set("已停止（数据保留，可继续查看或点“清空”重置）")
            self._set_status_color("#555555")

    def _update_status(self):
        """刷新状态栏：设备 + 采样率 + 已记录时长/剩余时长"""
        if not self.running:
            return
        elapsed = time.time() - self.start_time
        base = f"监测中：{self.device_var.get()} | {self.engine.samplerate} Hz"
        if self.max_duration > 0:
            remain = max(0.0, self.max_duration - elapsed)
            base += f" | 已记录 {self._fmt_elapsed(elapsed)} / {self._fmt_elapsed(self.max_duration)}（剩 {self._fmt_elapsed(remain)}）"
        else:
            base += f" | 已记录 {self._fmt_elapsed(elapsed)}"
        self.status_var.set(base)

    # ---------------------------------------------------------- 数据刷新
    def _poll(self):
        """定时从队列取数据并刷新界面"""
        if self.running and self.engine:
            # 每轮最多取 2 条（回调 50ms/条 = 20条/秒，轮询 100ms ≈ 2条/轮）。
            # 若界面某帧卡顿导致积压，多余旧样本直接丢弃，避免“越积越卡、
            # 积压突发灌入”的雪崩循环（画面乱跳的根源）。
            try:
                n = 0
                while n < 2:
                    t, db = self.engine.queue.get_nowait()
                    n += 1
                    self.history.append((t, db))
                    # 会话最大/最小：从监测开始持续累计（不受曲线窗口影响）
                    if db > self.session_max_db:
                        self.session_max_db = db
                    if db < self.session_min_db:
                        self.session_min_db = db
            except queue.Empty:
                pass
            if n == 2:
                # 丢弃积压的旧样本（新版 queue.Queue 无 clear()，用排空代替）
                while True:
                    try:
                        self.engine.queue.get_nowait()
                    except queue.Empty:
                        break

            # 达到记录上限 → 自动停止
            if self.max_duration > 0 and \
                    (time.time() - self.start_time) >= self.max_duration:
                self._stop(auto=True)
                messagebox.showinfo("记录完成",
                                    f"已达到设定的记录上限"
                                    f"（{self._fmt_elapsed(self.max_duration)}），自动停止。\n"
                                    f"会话最大 {self.session_max_db:.1f} dB / "
                                    f"会话最小 {self.session_min_db:.1f} dB\n"
                                    f"如需保存日志请点击“保存日志(CSV)”。")
                self.root.after(POLL_MS, self._poll)
                return

            if self.history:
                latest_db = self.history[-1][1]
                color, name = zone_of_db(latest_db)
                self.db_label.config(text=f"{latest_db:5.1f}", fg=color)
                self.desc_label.config(text=name, fg=color)
                self._update_stats()
                self._draw_gauge()
                # 图表重绘较重，节流到 300ms 一次，数字与量程条仍 100ms 更新
                now = time.time()
                if now - self._last_chart_ts >= 0.3:
                    self._last_chart_ts = now
                    self._update_chart()
                self._maybe_log()
            else:
                self._draw_gauge()
            self._update_status()
        self.root.after(POLL_MS, self._poll)

    def _update_stats(self):
        # 长窗口下（如 10 分钟=12000 条）全量遍历会拖慢刷新，
        # 峰值只回溯最近 3 秒，平均/最低交给 numpy 一次完成。
        now = time.time()
        dbs_list = []
        peak3 = -120.0
        for t, db in reversed(self.history):
            if now - t > 3.0:
                break                      # 超过 3 秒的样本不再参与峰值
            if db > peak3:
                peak3 = db
            dbs_list.append(db)
        dbs_all = [db for _, db in self.history]
        avg = float(np.mean(dbs_all))
        low = float(np.min(dbs_all))
        self.stat_items["cur"].config(text=f"{self.history[-1][1]:.1f} dB")
        self.stat_items["peak"].config(text=f"{peak3:.1f} dB")
        self.stat_items["avg"].config(text=f"{avg:.1f} dB")
        self.stat_items["low"].config(text=f"{low:.1f} dB")
        self.stat_items["smax"].config(text=f"{self.session_max_db:.1f} dB")
        self.stat_items["smin"].config(text=f"{self.session_min_db:.1f} dB")

    def _draw_gauge(self):
        """双条设计：
        上条 = 静态参考刻度（分区色永不变化、永不被覆盖）
        下条 = 当前音量电平条（动态填充 + 实时数值 + 峰值标记）
        """
        w = self.gauge.winfo_width()
        if w < 100:
            return
        self.gauge.delete("all")
        x0, x1 = 10, w - 10

        def clamp_x(px, margin):
            return max(x0 + margin, min(px, x1 - margin))

        def x_of_db(db):
            frac = min(max((db - GAUGE_MIN) / (GAUGE_MAX - GAUGE_MIN), 0.0), 1.0)
            return x0 + frac * (x1 - x0)

        # ============ 上条：静态参考刻度（标尺条） ============
        ry, rbar = 24, 20
        # 标尺刻度：每个分区边界的 dB 值（条上方，带刻度线）
        boundaries = [lo for lo, _, _, _ in ZONES] + [GAUGE_MAX]
        for b in boundaries:
            bx = clamp_x(x_of_db(b), 14)
            self.gauge.create_line(bx, ry - 5, bx, ry - 1,
                                   fill="#999999", width=1)
            self.gauge.create_text(bx, ry - 11, text=f"{b:.0f}",
                                   font=("Segoe UI", 8), fill="#555555")
        for lo, hi, color, _ in ZONES:
            a = x_of_db(max(lo, GAUGE_MIN))
            b = x_of_db(min(hi, GAUGE_MAX))
            self.gauge.create_rectangle(a, ry, b, ry + rbar, fill=color, outline="")
        self.gauge.create_rectangle(x0, ry, x1, ry + rbar, outline="#bbbbbb")
        # 分区名称标注（条下方）
        for lo, hi, _, name in ZONES:
            cx = x_of_db((lo + hi) / 2)
            self.gauge.create_text(clamp_x(cx, 16), ry + rbar + 12, text=name,
                                   font=("Microsoft YaHei", 9), fill="#666666")

        # ============ 下条：当前音量电平条（动态） ============
        ly, lbar = 68, 18
        self.gauge.create_rectangle(x0, ly, x1, ly + lbar, fill="#ececec",
                                    outline="#bbbbbb")
        # 分区边界刻度（浅色竖线，帮助对照参考刻度）
        for lo, _, _, _ in ZONES[1:]:
            lx = x_of_db(lo)
            self.gauge.create_line(lx, ly + 2, lx, ly + lbar - 2,
                                   fill="#cccccc", width=1)

        if self.history:
            db = self.history[-1][1]
            color, _ = zone_of_db(db)
            fx = x_of_db(db)
            if fx > x0 + 1:
                self.gauge.create_rectangle(x0, ly, fx, ly + lbar, fill=color,
                                            outline="")
            # 当前数值（靠边向内钳制，防溢出）
            self.gauge.create_text(clamp_x(fx, 34), ly + lbar + 12,
                                   text=f"{db:.1f}",
                                   font=("Segoe UI", 10, "bold"), fill=color)

            # 峰值保持刻度（红线 + 文字，均在电平条下方）
            now = time.time()
            peak3 = -120.0
            for t, db2 in reversed(self.history):
                if now - t > 3.0:
                    break
                if db2 > peak3:
                    peak3 = db2
            if peak3 > -100.0:
                px = clamp_x(x_of_db(peak3), 4)
                self.gauge.create_line(px, ly - 4, px, ly + lbar + 4,
                                       fill="#c0392b", width=2)
                self.gauge.create_text(clamp_x(px, 30), ly + lbar + 26,
                                       text=f"峰值 {peak3:.0f}",
                                       font=("Microsoft YaHei", 8),
                                       fill="#c0392b")
        else:
            self.gauge.create_text(x0 + 10, ly + lbar // 2, text="等待信号",
                                   font=("Microsoft YaHei", 9), fill="#aaaaaa",
                                   anchor="w")

    def _aggregate_series(self, now):
        """将全量历史按固定 600 个时间片聚合，返回 (xs, ys)。

        旧做法“隔 N 取 1”在数据量变化（n 越过 600 的整数倍）时
        选点整体换一批，曲线会整条跳变。这里分片数量固定为 600，
        与数据量无关：每片只统计片内样本，新样本入片/旧样本出片
        都是平滑过渡，曲线不再整体跳动。

        三种聚合方式（只影响显示，全量数据始终保留）：
          max    = 片内最高值（抓峰值）
          avg    = 片内平均值（看整体）
          avgmax = (平均 + 最高) / 2（两者折中）
        """
        n_bins = 600
        bs = self.window_secs / n_bins
        cnt = [0] * n_bins
        sm = [0.0] * n_bins
        mx = [-120.0] * n_bins
        for t, db in self.history:
            age = now - t
            k = int(age / bs)
            if 0 <= k < n_bins:
                cnt[k] += 1
                sm[k] += db
                if db > mx[k]:
                    mx[k] = db
        xs, ys = [], []
        mode = self.sample_mode
        for k in range(n_bins):
            if cnt[k] == 0:
                continue
            avg = sm[k] / cnt[k]
            if mode == "max":
                v = mx[k]
            elif mode == "avg":
                v = avg
            else:                       # avgmax
                v = (avg + mx[k]) / 2.0
            xs.append((k + 0.5) * bs)   # 片中心的“秒前”位置
            ys.append(v)
        return xs, ys

    def _update_chart(self):
        if not self.history:
            self.line.set_data([], [])
            self.canvas.draw_idle()
            return
        xs, ys = self._aggregate_series(time.time())
        self.line.set_data(xs, ys)
        self.ax.set_xlim(self.window_secs, 0)
        self.canvas.draw_idle()

    # ---------------------------------------------------------- 日志
    def _toggle_logging(self):
        self.logging_enabled = self.log_var.get()
        if self.logging_enabled:
            self.log_rows = []
            self._last_log_time = 0.0
            self.status_var.set("日志记录已开启（数据暂存在内存，点“保存日志”导出）")
            self._set_status_color("#2980b9")
        else:
            self.status_var.set("日志记录已关闭")

    def _maybe_log(self):
        if not (self.logging_enabled and self.history):
            return
        now = time.time()
        if now - self._last_log_time < LOG_INTERVAL:
            return
        self._last_log_time = now
        dbs = [db for _, db in self.history]
        peak3 = max((db for t, db in self.history if now - t <= 3.0), default=-120.0)
        row = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "current": round(self.history[-1][1], 1),
            "peak3": round(peak3, 1),
            "avg_window": round(float(np.mean(dbs)), 1),
            "low_window": round(float(np.min(dbs)), 1),
            "session_max": round(self.session_max_db, 1),
            "session_min": round(self.session_min_db, 1),
        }
        self.log_rows.append(row)

    def _save_log(self):
        if not self.log_rows:
            messagebox.showinfo("提示", "当前没有日志数据。请先勾选“启用日志记录”并监测一段时间。")
            return
        fname = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile=f"分贝日志_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            filetypes=[("CSV 文件", "*.csv")])
        if not fname:
            return
        with open(fname, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["time", "current", "peak3",
                                              "avg_window", "low_window",
                                              "session_max", "session_min"])
            w.writeheader()
            w.writerows(self.log_rows)
        self.status_var.set(f"日志已保存：{fname}")
        self._set_status_color("#27ae60")

    def _save_chart(self):
        fname = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=f"分贝曲线_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            filetypes=[("PNG 图片", "*.png")])
        if not fname:
            return
        self.fig.savefig(fname, dpi=120, bbox_inches="tight")
        self.status_var.set(f"图表已保存：{fname}")
        self._set_status_color("#27ae60")

    def _clear_all(self):
        """清空按钮：重置历史、统计、图表、会话最大/最小和日志缓冲。
        与停止不同——停止保留数据，清空是唯一的数据重置入口。"""
        if self.log_rows:
            if not messagebox.askyesno(
                    "清空确认",
                    f"将清空 {len(self.log_rows)} 条未保存日志和全部显示数据，确定？"):
                return
        self.history.clear()
        self.session_max_db = -120.0
        self.session_min_db = 0.0
        self.log_rows = []
        self._last_log_time = 0.0
        self.start_time = time.time()      # 监测中则从此刻重新计时
        # 大数字与描述
        self.db_label.config(text="--.-", fg="#95a5a6")
        self.desc_label.config(text="等待信号", fg="#95a5a6")
        # 统计面板
        for key in self.stat_items:
            self.stat_items[key].config(text="--.- dB")
        # 曲线
        self.line.set_data([], [])
        self.canvas.draw_idle()
        # 量程条
        self._draw_gauge()
        if self.running:
            self.status_var.set("已清空，继续监测中")
            self._set_status_color("#27ae60")
        else:
            self.status_var.set("已清空")
            self._set_status_color("#555555")

    # ---------------------------------------------------------- 其他
    def _set_status_color(self, color):
        self.status_lab.config(fg=color)

    def _on_close(self):
        if self.running:
            self._stop()
        if self.log_rows:
            if messagebox.askyesno("保存日志", "检测到未保存的日志数据，是否保存？"):
                self._save_log()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = DecibelMeterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
