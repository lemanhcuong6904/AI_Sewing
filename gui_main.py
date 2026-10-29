import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import threading
import time
import json
import os
import serial
from PIL import Image, ImageTk

# Import existing logic
from main_adjustor import get_default_port, send_and_receive
from utils.camera import get_camera
from utils.viz_adjustor import draw_adjustor_overlay
from utils.constants import (
    DEFAULT_X_BASIS, DEFAULT_Y_BASIS, PX_PER_CM_X, PX_PER_CM_Y,
    AXIS_LENGTH_PX, TRANSLATION_DEADBAND_CM, ROTATION_DEADBAND_DEG
)
from utils.line_detectors.detector import DetectorMode
from modules.fabric_adjustor_prior import PriorMaskAdjustor
from modules.fabric_adjustor import LineDetectionAdjustor, NonOrthoLineAdjustor

CONFIG_FILE = "gui_config.json"

class AI_Sewing_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Sewing - Native Adjustor")
        self.root.geometry("1200x800")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Threading and State
        self.running = False
        self.engine_thread = None
        self.cam = None
        self.ser = None
        self.trigger_compute = False
        self.keyboard_buffer = ""
        
        # Load Config
        self.config = self.load_config()

        self.setup_ui()
        self.root.bind('<Key>', self.on_key_press)

    def load_config(self):
        default_config = {
            "detector": "dexined",
            "x_basis": DEFAULT_X_BASIS,
            "y_basis": DEFAULT_Y_BASIS,
            "prior_mask": "",
            "match_method": "contour",
            "morphology": "fill",
            "enhance_contrast": False,
            "smooth_texture": True,
            "batch_size": 10,
            "max_moderations": 2,
            "max_angle_moderations": 2,
            "continuous": False,
            "dry_run": True,
            "port": "",
            "baud": 115200,
            "x_offset_cm": 0.175,
            "y_offset_cm": 0.3,
            "rot_alignment": "vertical"
        }
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    loaded = json.load(f)
                    default_config.update(loaded)
            except Exception as e:
                print("Error loading config:", e)
        return default_config

    def save_config(self):
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            print("Error saving config:", e)

    def setup_ui(self):
        # Left Panel (Controls)
        left_frame = ttk.Frame(self.root, width=350)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Right Panel (Video)
        right_frame = ttk.Frame(self.root)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.video_label = ttk.Label(right_frame, text="Video Feed", anchor=tk.CENTER)
        self.video_label.pack(fill=tk.BOTH, expand=True)

        self.status_var = tk.StringVar(value="Status: Stopped")
        status_label = ttk.Label(right_frame, textvariable=self.status_var, font=("Helvetica", 14, "bold"))
        status_label.pack(side=tk.BOTTOM, pady=10)

        # --- Settings (Left Panel) ---
        canvas = tk.Canvas(left_frame)
        scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.vars = {}
        row = 0

        def add_entry(label, key, type_func=str):
            ttk.Label(self.scrollable_frame, text=label).grid(row=row, column=0, sticky="w", pady=2)
            var = tk.StringVar(value=str(self.config.get(key, "")))
            ttk.Entry(self.scrollable_frame, textvariable=var).grid(row=row, column=1, sticky="ew", pady=2)
            self.vars[key] = (var, type_func)
            return row + 1

        def add_check(label, key):
            var = tk.BooleanVar(value=bool(self.config.get(key, False)))
            ttk.Checkbutton(self.scrollable_frame, text=label, variable=var).grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
            self.vars[key] = (var, bool)
            return row + 1

        def add_combo(label, key, values):
            ttk.Label(self.scrollable_frame, text=label).grid(row=row, column=0, sticky="w", pady=2)
            var = tk.StringVar(value=str(self.config.get(key, values[0])))
            cb = ttk.Combobox(self.scrollable_frame, textvariable=var, values=values, state="readonly")
            cb.grid(row=row, column=1, sticky="ew", pady=2)
            self.vars[key] = (var, str)
            return row + 1

        row = add_combo("Detector Mode", "detector", ['lsd', 'fld', 'edlines', 'ensemble', 'mlsd', 'dexined'])
        row = add_combo("Match Method", "match_method", ['moments', 'ecc', 'bbox', 'feature', 'contour'])
        row = add_combo("Morphology", "morphology", ['none', 'erode', 'open', 'fill'])
        row = add_combo("Rot Alignment", "rot_alignment", ["horizontal", "vertical"])

        row = add_entry("X Basis", "x_basis", float)
        row = add_entry("Y Basis", "y_basis", float)
        row = add_entry("X Offset CM", "x_offset_cm", float)
        row = add_entry("Y Offset CM", "y_offset_cm", float)
        
        # Prior Mask File Picker
        ttk.Label(self.scrollable_frame, text="Prior Mask").grid(row=row, column=0, sticky="w", pady=2)
        pm_frame = ttk.Frame(self.scrollable_frame)
        pm_frame.grid(row=row, column=1, sticky="ew")
        self.pm_var = tk.StringVar(value=self.config.get("prior_mask", ""))
        ttk.Entry(pm_frame, textvariable=self.pm_var, width=15).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(pm_frame, text="...", width=3, command=self.browse_mask).pack(side=tk.RIGHT)
        self.vars["prior_mask"] = (self.pm_var, str)
        row += 1

        row = add_entry("Batch Size", "batch_size", int)
        row = add_entry("Max Moderations", "max_moderations", int)
        row = add_entry("Max Angle Mod", "max_angle_moderations", int)
        row = add_entry("Baud Rate", "baud", int)
        row = add_entry("Port", "port", str)

        row = add_check("Enhance Contrast", "enhance_contrast")
        row = add_check("Smooth Texture", "smooth_texture")
        row = add_check("Continuous Mode", "continuous")
        row = add_check("Dry Run", "dry_run")

        # Action Buttons
        btn_frame = ttk.Frame(self.scrollable_frame)
        btn_frame.grid(row=row, column=0, columnspan=2, pady=10)
        
        self.start_btn = ttk.Button(btn_frame, text="START ENGINE", command=self.toggle_engine)
        self.start_btn.grid(row=0, column=0, padx=5, pady=5)
        
        ttk.Button(btn_frame, text="Trigger L1", command=lambda: self.manual_trigger('l1')).grid(row=1, column=0, padx=5, pady=5)
        ttk.Button(btn_frame, text="Trigger L2", command=lambda: self.manual_trigger('l2')).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(btn_frame, text="Dry-Run OK", command=lambda: self.manual_trigger('o')).grid(row=2, column=0, columnspan=2, padx=5, pady=5)

    def browse_mask(self):
        filepath = filedialog.askopenfilename(title="Select Prior Mask Image", filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if filepath:
            self.pm_var.set(filepath)

    def on_key_press(self, event):
        if not self.running:
            return
        key = event.keysym.lower()
        if len(key) == 1:
            self.keyboard_buffer += key
            self.keyboard_buffer = self.keyboard_buffer[-2:]
        if self.keyboard_buffer == 'l1':
            self.manual_trigger('l1')
            self.keyboard_buffer = ""
        elif self.keyboard_buffer == 'l2':
            self.manual_trigger('l2')
            self.keyboard_buffer = ""
        elif key == 'o' and self.vars['dry_run'][0].get():
            self.manual_trigger('o')
            self.keyboard_buffer = ""

    def manual_trigger(self, trigger_type):
        if not self.running: return
        self.keyboard_buffer = ""
        if trigger_type == 'l1':
            self.active_x = self.config['x_basis']
            self.active_y = self.config['y_basis']
            self.adjustor.x_basis = self.active_x
            self.adjustor.y_basis = self.active_y
            self.trigger_compute = True
            self.status_var.set("Status: Triggered L1")
        elif trigger_type == 'l2':
            self.active_x = self.config['x_basis'] + self.config['x_offset_cm'] * PX_PER_CM_X
            self.active_y = self.config['y_basis'] + self.config['y_offset_cm'] * PX_PER_CM_Y
            self.adjustor.x_basis = self.active_x
            self.adjustor.y_basis = self.active_y
            self.trigger_compute = True
            self.status_var.set("Status: Triggered L2")
        elif trigger_type == 'o':
            self.trigger_compute = True
            self.status_var.set("Status: Triggered OK")

    def update_config_from_ui(self):
        for key, (var, type_func) in self.vars.items():
            val = var.get()
            try:
                self.config[key] = type_func(val) if val != "" else type_func()
            except ValueError:
                pass
        self.save_config()

    def toggle_engine(self):
        if self.running:
            self.running = False
            self.start_btn.config(text="START ENGINE")
            self.status_var.set("Status: Stopped")
        else:
            self.update_config_from_ui()
            self.running = True
            self.start_btn.config(text="STOP ENGINE")
            self.status_var.set("Status: Running")
            self.engine_thread = threading.Thread(target=self.engine_loop, daemon=True)
            self.engine_thread.start()

    def engine_loop(self):
        cfg = self.config
        
        # 1. Setup UART
        self.ser = None
        if not cfg['dry_run']:
            port = cfg['port'] or get_default_port()
            print(f"Connecting to {port} at {cfg['baud']} baud...")
            try:
                self.ser = serial.Serial(port=port, baudrate=cfg['baud'], timeout=0.1)
                time.sleep(2)
            except Exception as e:
                print(f"UART Error: {e}")
                self.status_var.set(f"UART Error: {e}")
                self.running = False
                self.root.after(0, lambda: self.start_btn.config(text="START ENGINE"))
                return

        # 2. Setup Adjustor
        self.active_x = cfg['x_basis']
        self.active_y = cfg['y_basis']
        crop_roi = (
            max(0, int(self.active_x - 2.0 * PX_PER_CM_X)),
            int(self.active_x + 8.0 * PX_PER_CM_X),
            max(0, int(self.active_y - 4.1 * PX_PER_CM_Y)),
            int(self.active_y + 2 * PX_PER_CM_Y)
        )

        detector_mode = DetectorMode(cfg['detector'])
        if detector_mode == DetectorMode.DEXINED:
            if not cfg['prior_mask']:
                self.status_var.set("Error: Dexined requires Prior Mask")
                self.running = False
                self.root.after(0, lambda: self.start_btn.config(text="START ENGINE"))
                return
            self.adjustor = PriorMaskAdjustor(
                prior_mask_path=cfg['prior_mask'],
                x_basis=self.active_x, y_basis=self.active_y,
                enhance_contrast=cfg['enhance_contrast'], smooth_texture=cfg['smooth_texture'],
                match_method=cfg['match_method'], morphology=cfg['morphology'],
                batch_size=cfg['batch_size'], crop_roi=crop_roi
            )
        else:
            if cfg['prior_mask']:
                self.adjustor = NonOrthoLineAdjustor(
                    prior_img_path=cfg['prior_mask'], x_basis=self.active_x, y_basis=self.active_y,
                    enhance_contrast=cfg['enhance_contrast'], smooth_texture=cfg['smooth_texture'],
                    detector_mode=detector_mode, batch_size=cfg['batch_size'], crop_roi=crop_roi
                )
            else:
                self.adjustor = LineDetectionAdjustor(
                    x_basis=self.active_x, y_basis=self.active_y,
                    enhance_contrast=cfg['enhance_contrast'], smooth_texture=cfg['smooth_texture'],
                    detector_mode=detector_mode, batch_size=cfg['batch_size'], crop_roi=crop_roi
                )

        self.cam = get_camera()
        
        moderation_count = 0
        is_m1_m2_modified = False
        is_m3_modified = False
        rot_index = 0 if cfg['rot_alignment'] == 'horizontal' else 1
        
        self.trigger_compute = False
        
        while self.running:
            # Check UART
            if self.ser and self.ser.in_waiting > 0:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if "OK" in line: self.trigger_compute = True
                if "L1" in line: self.manual_trigger('l1')
                if "L2" in line: self.manual_trigger('l2')

            if not self.trigger_compute:
                # Idle: Update UI with camera feed
                if self.cam:
                    ret, frame, has_new = self.cam.read()
                    if ret and frame is not None:
                        display_img = frame.copy()
                        draw_adjustor_overlay(display_img, (self.adjustor.x_basis, self.adjustor.y_basis), AXIS_LENGTH_PX)
                        self.update_video_frame(display_img, "WAITING FOR ARM / UART TRIGGER")
                time.sleep(0.03)
                continue
            
            self.status_var.set("Computing Signals...")
            control_signals = self.adjustor.compute_single_signal(self.cam, target_frames=150, visualize=False)
            
            if control_signals:
                (tx_px, ty_px), rot_angles = control_signals
                tx_cm = tx_px / PX_PER_CM_X
                ty_cm = ty_px / PX_PER_CM_Y
                tx_cm = max(-2.0, min(3.0, tx_cm))
                ty_cm = max(-2.0, min(3.0, ty_cm))

                m1_mm = tx_cm * 10.0
                m2_mm = ty_cm * 10.0
                m3_deg = max(-10.0, min(10.0, rot_angles[rot_index]))

                if moderation_count < cfg['max_moderations']:
                    if not is_m1_m2_modified:
                        m3_deg = 0.0
                        if abs(tx_cm) < TRANSLATION_DEADBAND_CM: m1_mm = 0.0
                        if abs(ty_cm) < TRANSLATION_DEADBAND_CM: m2_mm = 0.0
                        is_m1_m2_modified = True
                    elif not is_m3_modified:
                        m1_mm = m2_mm = 0.0
                        if abs(m3_deg) < ROTATION_DEADBAND_DEG: m3_deg = 0.0
                        is_m3_modified = True
                    
                    if is_m1_m2_modified and is_m3_modified:
                        moderation_count += 1
                        is_m1_m2_modified = False
                        if moderation_count < cfg['max_angle_moderations']:
                            is_m3_modified = False
                else:
                    is_m1_m2_modified = is_m3_modified = False
                    moderation_count = 0
                    time.sleep(1)
                    if self.ser: send_and_receive(self.ser, 'k5')
                    print("Released!")
                    self.trigger_compute = False
                    continue

                cmd = f"{m1_mm}, {m2_mm}, {m3_deg}"
                if self.ser:
                    send_and_receive(self.ser, cmd)
                else:
                    print(f"[DRY-RUN] CMD: {cmd}")
                
            self.trigger_compute = False
            self.status_var.set("Status: Running (Waiting)")

        if self.ser:
            self.ser.close()

    def update_video_frame(self, frame, text):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Resize to fit the label
        h, w = frame.shape[:2]
        target_w = self.video_label.winfo_width()
        target_h = self.video_label.winfo_height()
        if target_w > 10 and target_h > 10:
            scale = min(target_w/w, target_h/h)
            new_w, new_h = int(w*scale), int(h*scale)
            if new_w > 0 and new_h > 0:
                frame = cv2.resize(frame, (new_w, new_h))
        
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        
        # Schedule the UI update on the main thread
        def update_ui():
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        self.root.after(0, update_ui)

    def on_closing(self):
        self.running = False
        self.save_config()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = AI_Sewing_GUI(root)
    root.mainloop()
