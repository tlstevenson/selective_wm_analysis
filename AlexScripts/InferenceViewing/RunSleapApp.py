import os
import sys
import json
import shutil
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import cv2  # Added to read video frame counts for strided inference

# --- CORE UTILITY FUNCTIONS ---

def get_file_paths(directory_path, extension="None"):
    """Returns a list of strings containing the paths of all files in a directory."""
    path_obj = Path(directory_path)
    if extension == "None":
        return [str(file) for file in path_obj.iterdir() if file.is_file()]
    else:
        return [str(file) for file in path_obj.iterdir() if file.is_file() and file.suffix == extension]

def vid_to_slp(path):
    #.mp4 -> .slp
    path_without_ext, ext = os.path.splitext(path)
    return f"{path_without_ext}.slp"

# --- GUI APPLICATION CLASS ---

class SleapApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SLEAP Inference Automator")
        self.root.geometry("850x980")
        
        # Internal state
        self.prefs_file = "sleap_prefs.json"
        self.model_mode = tk.StringVar(value="top_down")       # 'single' or 'top_down' for Standard
        self.port_model_mode = tk.StringVar(value="top_down")  # 'single' or 'top_down' for Port
        self.base_dir = tk.StringVar()
        
        self.models = []       # List of lists: [[model1], [cent1, center1], ...]
        self.port_models = []  # List of lists: [[model1], [cent1, center1], ...]
        self.port_interval = tk.StringVar(value="1800")
        
        self.directories = []  # List of video directories
        self.videos = []       # List of individual videos
        
        self.setup_ui()
        self.auto_load_prefs()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- PREFERENCES & BASE DIRECTORY ---
        pref_frame = ttk.LabelFrame(main_frame, text="Preferences & Settings", padding="5")
        pref_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(pref_frame, text="Load Prefs", command=self.load_prefs).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(pref_frame, text="Save Prefs", command=self.save_prefs).grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(pref_frame, text="Base Project Dir (for saving models):").grid(row=1, column=0, sticky=tk.W, padx=5)
        ttk.Entry(pref_frame, textvariable=self.base_dir, width=50).grid(row=1, column=1, columnspan=2, padx=5)
        ttk.Button(pref_frame, text="Browse", command=self.browse_base_dir).grid(row=1, column=3, padx=5)

        # --- STANDARD MODEL CONFIGURATION ---
        model_frame = ttk.LabelFrame(main_frame, text="Standard Model Configuration", padding="5")
        model_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        mode_subframe = ttk.Frame(model_frame)
        mode_subframe.pack(fill=tk.X, pady=2)
        ttk.Label(mode_subframe, text="Standard Mode:").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mode_subframe, text="Single Animal (1 Model)", variable=self.model_mode, value="single", command=self.update_model_listbox).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mode_subframe, text="Top-Down (Centroid + Centered)", variable=self.model_mode, value="top_down", command=self.update_model_listbox).pack(side=tk.LEFT, padx=5)

        self.model_listbox = tk.Listbox(model_frame, height=3)
        self.model_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        btn_frame = ttk.Frame(model_frame)
        btn_frame.pack(fill=tk.X)
        ttk.Button(btn_frame, text="Add Model", command=self.add_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Clear Models", command=self.clear_models).pack(side=tk.LEFT, padx=5)

        # --- PORT MODELS CONFIGURATION ---
        port_frame = ttk.LabelFrame(main_frame, text="Port Models (Strided Inference)", padding="5")
        port_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        port_mode_subframe = ttk.Frame(port_frame)
        port_mode_subframe.pack(fill=tk.X, pady=2)
        ttk.Label(port_mode_subframe, text="Port Mode:").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(port_mode_subframe, text="Single Animal (1 Model)", variable=self.port_model_mode, value="single", command=self.update_port_listbox).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(port_mode_subframe, text="Top-Down (Centroid + Centered)", variable=self.port_model_mode, value="top_down", command=self.update_port_listbox).pack(side=tk.LEFT, padx=5)

        port_controls = ttk.Frame(port_frame)
        port_controls.pack(fill=tk.X, pady=2)
        ttk.Label(port_controls, text="Prediction Interval (Frames):").pack(side=tk.LEFT, padx=5)
        ttk.Entry(port_controls, textvariable=self.port_interval, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(port_controls, text="(e.g., 1800 for 1 min @ 30fps)").pack(side=tk.LEFT, padx=5)

        self.port_listbox = tk.Listbox(port_frame, height=3)
        self.port_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        port_btn_frame = ttk.Frame(port_frame)
        port_btn_frame.pack(fill=tk.X)
        ttk.Button(port_btn_frame, text="Add Port Model", command=self.add_port_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(port_btn_frame, text="Clear Port Models", command=self.clear_port_models).pack(side=tk.LEFT, padx=5)

        # --- VIDEO CONFIGURATION ---
        vid_frame = ttk.LabelFrame(main_frame, text="Video Sources", padding="5")
        vid_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.vid_listbox = tk.Listbox(vid_frame, height=5)
        self.vid_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        vid_btn_frame = ttk.Frame(vid_frame)
        vid_btn_frame.pack(fill=tk.X)
        ttk.Button(vid_btn_frame, text="Add Directory", command=self.add_directory).pack(side=tk.LEFT, padx=5)
        ttk.Button(vid_btn_frame, text="Add Individual Video(s)", command=self.add_videos).pack(side=tk.LEFT, padx=5)
        ttk.Button(vid_btn_frame, text="Clear Videos/Dirs", command=self.clear_videos).pack(side=tk.LEFT, padx=5)

        # --- EXECUTION & LOGS ---
        run_frame = ttk.Frame(main_frame)
        run_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(run_frame, text="RUN INFERENCE", command=self.start_inference, style="Accent.TButton").pack(fill=tk.X, pady=(0, 10))

        # Progress tracking UI
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(run_frame, textvariable=self.status_var, font=("Helvetica", 10, "bold")).pack(anchor=tk.W)
        
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(run_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=(2, 0))

        log_frame = ttk.LabelFrame(main_frame, text="Console Output", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = tk.Text(log_frame, height=10, state="disabled", bg="#1e1e1e", fg="#d4d4d4")
        self.log_text.pack(fill=tk.BOTH, expand=True)

    # --- UI UPDATE & LOGGING ---
    def log(self, message):
        def append():
            self.log_text.config(state="normal")
            self.log_text.insert(tk.END, str(message) + "\n")
            self.log_text.see(tk.END)
            self.log_text.config(state="disabled")
        self.root.after(0, append)

    def update_progress(self, current, total, status_text=None):
        def update():
            if total > 0:
                percentage = (current / total) * 100
                self.progress_var.set(percentage)
            if status_text:
                self.status_var.set(status_text)
        self.root.after(0, update)

    def browse_base_dir(self):
        dir_path = filedialog.askdirectory(title="Select Base Project Directory")
        if dir_path:
            self.base_dir.set(dir_path)

    # --- PREFERENCES LOGIC ---
    def auto_load_prefs(self):
        if os.path.exists(self.prefs_file):
            self.load_prefs(show_msg=False)

    def load_prefs(self, show_msg=True):
        try:
            with open(self.prefs_file, 'r') as f:
                data = json.load(f)
            
            self.model_mode.set(data.get("model_mode", "top_down"))
            self.port_model_mode.set(data.get("port_model_mode", "top_down"))
            self.base_dir.set(data.get("base_dir", ""))
            
            self.models = data.get("models", [])
            self.port_models = data.get("port_models", [])
            self.port_interval.set(data.get("port_interval", "1800"))
            
            self.directories = data.get("directories", [])
            self.videos = data.get("videos", [])
            
            self.update_model_listbox()
            self.update_port_listbox()
            self.update_vid_listbox()
            if show_msg:
                self.log(f"Preferences loaded from {self.prefs_file}")
                messagebox.showinfo("Success", "Preferences loaded successfully.")
        except Exception as e:
            if show_msg:
                messagebox.showerror("Error", f"Could not load preferences: {e}")

    def save_prefs(self):
        data = {
            "model_mode": self.model_mode.get(),
            "port_model_mode": self.port_model_mode.get(),
            "base_dir": self.base_dir.get(),
            "models": self.models,
            "port_models": self.port_models,
            "port_interval": self.port_interval.get(),
            "directories": self.directories,
            "videos": self.videos
        }
        try:
            with open(self.prefs_file, 'w') as f:
                json.dump(data, f, indent=4)
            self.log(f"Preferences saved to {self.prefs_file}")
            messagebox.showinfo("Success", "Preferences saved successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save preferences: {e}")

    # --- MODEL & VIDEO MANAGEMENT ---
    def add_model(self):
        if self.model_mode.get() == "single":
            model_path = filedialog.askdirectory(title="Select Single Animal Model Directory")
            if model_path:
                self.models.append([model_path])
        else:
            centroid = filedialog.askdirectory(title="Select CENTROID Model Directory")
            if not centroid: return
            centered = filedialog.askdirectory(title="Select CENTERED INSTANCE Model Directory")
            if not centered: return
            self.models.append([centroid, centered])
        self.update_model_listbox()

    def clear_models(self):
        self.models.clear()
        self.update_model_listbox()

    def update_model_listbox(self):
        self.model_listbox.delete(0, tk.END)
        for idx, m in enumerate(self.models):
            if len(m) == 1:
                self.model_listbox.insert(tk.END, f"Standard {idx+1} [Single]: {os.path.basename(m[0])}")
            else:
                self.model_listbox.insert(tk.END, f"Standard {idx+1} [Top-Down]: Centroid: {os.path.basename(m[0])} | Centered: {os.path.basename(m[1])}")

    def add_port_model(self):
        if self.port_model_mode.get() == "single":
            model_path = filedialog.askdirectory(title="Select Port Single Animal Model Directory")
            if model_path:
                self.port_models.append([model_path])
        else:
            centroid = filedialog.askdirectory(title="Select Port CENTROID Model Directory")
            if not centroid: return
            centered = filedialog.askdirectory(title="Select Port CENTERED INSTANCE Model Directory")
            if not centered: return
            self.port_models.append([centroid, centered])
        self.update_port_listbox()

    def clear_port_models(self):
        self.port_models.clear()
        self.update_port_listbox()

    def update_port_listbox(self):
        self.port_listbox.delete(0, tk.END)
        for idx, m in enumerate(self.port_models):
            if isinstance(m, str):
                self.port_models[idx] = [m]
                m = [m]
                
            if len(m) == 1:
                self.port_listbox.insert(tk.END, f"Port {idx+1} [Single]: {os.path.basename(m[0])}")
            else:
                self.port_listbox.insert(tk.END, f"Port {idx+1} [Top-Down]: Centroid: {os.path.basename(m[0])} | Centered: {os.path.basename(m[1])}")

    def add_directory(self):
        dir_path = filedialog.askdirectory(title="Select Directory with MP4s")
        if dir_path:
            self.directories.append(dir_path)
            self.update_vid_listbox()

    def add_videos(self):
        files = filedialog.askopenfilenames(title="Select Video Files", filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
        if files:
            for f in files:
                if f not in self.videos:
                    self.videos.append(f)
            self.update_vid_listbox()

    def clear_videos(self):
        self.directories.clear()
        self.videos.clear()
        self.update_vid_listbox()

    def update_vid_listbox(self):
        self.vid_listbox.delete(0, tk.END)
        for d in self.directories:
            self.vid_listbox.insert(tk.END, f"[DIR] {d}")
        for v in self.videos:
            self.vid_listbox.insert(tk.END, f"[VID] {v}")

    # --- INFERENCE PIPELINE ---

    def create_write_paths(self, models_list, curr_vids, base_dir_val):
        """Copies models to the base_dir and returns corresponding write paths for all videos."""
        all_write_paths = []
        base = base_dir_val
        if not base:
            base = os.getcwd() # fallback

        for model_location_pair in models_list:
            write_paths = []
            
            # Copy Models
            if len(model_location_pair) == 2:
                centroid_loc, centered_loc = model_location_pair
                model_name = os.path.basename(os.path.splitext(os.path.splitext(centroid_loc)[0])[0])
                cent_dest = os.path.join(base, "models", os.path.basename(centroid_loc))
                center_dest = os.path.join(base, "models", os.path.basename(centered_loc))
                try:
                    shutil.copytree(centroid_loc, cent_dest, dirs_exist_ok=True)
                    shutil.copytree(centered_loc, center_dest, dirs_exist_ok=True)
                except Exception as e:
                    pass # Model probably exists
            elif len(model_location_pair) == 1:
                single_loc = model_location_pair[0]
                model_name = os.path.basename(os.path.splitext(os.path.splitext(single_loc)[0])[0])
                single_dest = os.path.join(base, "models", os.path.basename(single_loc))
                try:
                    shutil.copytree(single_loc, single_dest, dirs_exist_ok=True)
                except Exception as e:
                    pass # Model probably exists

            # Define Write Paths
            for video in curr_vids:
                try:
                    out_path = os.path.join(os.path.dirname(video), "predictions", model_name, vid_to_slp(os.path.basename(video)))
                    write_paths.append(out_path)
                except Exception as e:
                    self.log(f"Could not append path for video {video}: {e}")
                    
            all_write_paths.append(write_paths)
            
        return all_write_paths

    def slp_to_analysis_h5(self, slp_path, h5_path):
        self.log(f" -> Exporting to {h5_path} via CLI...")
        command = ["uv", "run", "sleap", "export", str(slp_path), "-o", str(h5_path)]
        try:
            if not os.path.exists(h5_path):
                self.log(f"Converting {slp_path} to {h5_path}")
                subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            self.log(f"H5 Export Error: {e}")

    def run_inference_on_list(self, video_list, write_path_list, model_path, progress_callback, stride=None):
        if not video_list:
            return False
        
        for i in range(len(video_list)):
            progress_callback(i)
            
            if os.path.exists(write_path_list[i]):
                self.log(f"{write_path_list[i]} already exists. Skipping inference.")
                continue

            command = []
            if len(model_path) == 2: # Top Down
                centroid, centered = model_path[0], model_path[1]
                if os.path.exists(centroid) and os.path.exists(centered):
                    command = ["sleap", "track", "-i", video_list[i], "-m", centroid, "-m", centered, "-o", write_path_list[i], "--max_instances", "1"]
                else:
                    self.log(f"Error: Could not find model paths. Skipping {video_list[i]}.")
                    continue
            elif len(model_path) == 1: # Single
                single = model_path[0]
                if os.path.exists(single):
                    command = ["sleap", "track", "-i", video_list[i], "-m", single, "-o", write_path_list[i], "--tracking"]
                else:
                    self.log(f"Error: Could not find model paths. Skipping {video_list[i]}.")
                    continue

            # Append stride argument if a valid interval is provided
            if stride and stride.isdigit() and int(stride) > 1:
                try:
                    # Dynamically extract frame count using OpenCV
                    cap = cv2.VideoCapture(video_list[i])
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    
                    if total_frames > 0:
                        stride_val = int(stride)
                        # Generate comma-separated list of frames to predict
                        frames_to_predict = [str(f) for f in range(0, total_frames, stride_val)]
                        frames_str = ",".join(frames_to_predict)
                        
                        self.log(f"Calculated {len(frames_to_predict)} frames to process at stride {stride_val}.")
                        command.extend(["--frames", frames_str])
                    else:
                        self.log(f"Warning: Could not read total frames for {os.path.basename(video_list[i])}. Ignoring stride.")
                except Exception as e:
                    self.log(f"Warning: Failed to calculate strided frames: {e}. Ignoring stride.")

            try:
                os.makedirs(os.path.dirname(write_path_list[i]), exist_ok=True)
                self.log(f"Running command: {' '.join(command)}")
                
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in process.stdout:
                    self.log(line.strip())
                process.wait()
                
                if process.returncode == 0:
                    self.log("Inference completed successfully!")
                else:
                    self.log(f"Inference failed with exit code {process.returncode}.")
            except Exception as e:
                self.log(f"Failed to launch subprocess: {e}")
                
        return True

    def start_inference(self):
        if not self.models and not self.port_models:
            messagebox.showwarning("Warning", "Please add at least one model (standard or port) before running.")
            return
        if not self.directories and not self.videos:
            messagebox.showwarning("Warning", "Please add at least one directory or video before running.")
            return
        
        # Extract Tkinter variables in the main thread
        base_dir_val = self.base_dir.get()
        port_interval_val = self.port_interval.get()
        
        if not base_dir_val:
            messagebox.showwarning("Warning", "Please define a Base Project Directory.")
            return

        self.update_progress(0, 1, "Initializing...")
        self.log_text.config(state="normal")
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state="disabled")
        
        # Pass the extracted strings into the thread
        thread = threading.Thread(target=self.inference_thread_worker, args=(base_dir_val, port_interval_val))
        thread.daemon = True
        thread.start()

    def inference_thread_worker(self, base_dir_val, port_interval_val):
        self.log("--- Starting Pipeline ---")
        
        # 1. Collect videos
        curr_vids = []
        for d in self.directories:
            curr_vids.extend(get_file_paths(d, ".mp4"))
        for v in self.videos:
            if v not in curr_vids:
                curr_vids.append(v)
                
        total_videos = len(curr_vids)
        total_models = len(self.models) + len(self.port_models)
        total_tasks = total_videos * total_models
        
        self.log(f"Found {total_videos} videos to process across {total_models} total model sets.")

        if total_tasks == 0:
            self.update_progress(0, 1, "Finished: No tasks to run.")
            return

        completed_tasks = 0

        # --- 2. RUN STANDARD MODELS ---
        if self.models:
            self.log("\n[PHASE 1] Processing Standard Models...")
            model_write_paths = self.create_write_paths(self.models, curr_vids, base_dir_val)

            for m_idx in range(len(model_write_paths)):
                self.log(f"\n>>> Inference for Standard Model #{m_idx + 1}")
                
                def progress_callback(vid_idx, m=m_idx):
                    current = completed_tasks + vid_idx
                    self.update_progress(current, total_tasks, f"Standard Model {m + 1}/{len(self.models)} | Video {vid_idx + 1}/{total_videos}")

                self.run_inference_on_list(curr_vids, model_write_paths[m_idx], self.models[m_idx], progress_callback, stride=None)
                completed_tasks += total_videos
                
                self.update_progress(completed_tasks, total_tasks, f"Exporting Standard Model {m_idx + 1} to .h5...")
                for slp_file in model_write_paths[m_idx]:
                    if os.path.exists(slp_file):
                        root, _ = os.path.splitext(slp_file)
                        self.slp_to_analysis_h5(slp_file, f"{root}.h5")

        # --- 3. RUN PORT MODELS ---
        if self.port_models:
            self.log(f"\n[PHASE 2] Processing Port Models (Interval: {port_interval_val})...")
            
            port_write_paths = self.create_write_paths(self.port_models, curr_vids, base_dir_val)

            for p_idx in range(len(port_write_paths)):
                self.log(f"\n>>> Inference for Port Model #{p_idx + 1}")
                
                def port_progress_callback(vid_idx, p=p_idx):
                    current = completed_tasks + vid_idx
                    self.update_progress(current, total_tasks, f"Port Model {p + 1}/{len(self.port_models)} | Video {vid_idx + 1}/{total_videos}")

                self.run_inference_on_list(curr_vids, port_write_paths[p_idx], self.port_models[p_idx], port_progress_callback, stride=port_interval_val)
                completed_tasks += total_videos

                self.update_progress(completed_tasks, total_tasks, f"Exporting Port Model {p_idx + 1} to .h5...")
                for slp_file in port_write_paths[p_idx]:
                    if os.path.exists(slp_file):
                        root, _ = os.path.splitext(slp_file)
                        self.slp_to_analysis_h5(slp_file, f"{root}.h5")
                    
        self.update_progress(total_tasks, total_tasks, "Finished processing all models and videos.")
        self.log("\n=== PIPELINE COMPLETELY FINISHED ===")

if __name__ == "__main__":
    root = tk.Tk()
    app = SleapApp(root)
    root.mainloop()