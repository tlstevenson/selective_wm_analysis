import init
import sys
import os
import cv2
import csv # NEW IMPORT
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QTabWidget, 
                             QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, 
                             QPushButton, QLineEdit, QCheckBox, QMessageBox,
                             QComboBox, QDialog, QFormLayout, QDialogButtonBox) # NEW IMPORTS
from PyQt5.QtCore import Qt, QTimer
import pyqtgraph as pg
import numpy as np
import neurofileread as nfr
import fp_analysis_helpers as fpah
from hankslab_db import db_access
from hankslab_db import basicRLtasks_db as bandit_db
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DataModel:
    """Interface class that holds and manages all data for the UI."""
    def __init__(self):
        self.session_id = None
        self.video_path = None
        self.hdf5_path = None
        self.doric_path = None
        self.cap = None
        
        # 1D Arrays for timestamps
        self.video_timestamps = np.array([])
        self.ts_timestamps = np.array([])
        
        # Current state
        self.current_frame_idx = 0
        self.current_image = None
        
        # List of 5 1D numpy arrays for the 5 plots (1 bottom + 4 side)
        self.time_series = [np.array([]) for _ in range(5)]
        self.plot_titles = [""] * 5  
        
        self.tracks = None
        self.scores = None
        self.node_names = []
        self.fp_data = {}
        
    def find_file_by_session(self, base_dir, session_id, extension):
        if not os.path.isdir(base_dir):
            print(f"Error: Base directory '{base_dir}' does not exist.")
            return None

        if not extension.startswith('.'):
            extension = f".{extension}"

        print(f"Searching for {extension} file for session '{session_id}' in {base_dir}...")
        
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.lower().endswith(extension.lower()) and session_id in file:
                    found_path = os.path.join(root, file)
                    print(f"Found: {found_path}")
                    return found_path
                    
        return None
    
    def load_real_session_data(self):
        """Called after on_submit_session successfully finds the files."""        
        self.time_series = [np.array([]) for _ in range(5)]
        self.plot_titles = [""] * 5
        self.ts_timestamps = np.array([])
            
        loc_db = bandit_db.LocalDB_BasicRLTasks("twoArmBandit")
        
        if self.video_path:
            if self.cap is not None:
                self.cap.release()
            self.cap = cv2.VideoCapture(self.video_path)
            
        if self.doric_path:
            self.video_timestamps = nfr.get_doric_timestamps(self.doric_path)
            
        if self.hdf5_path:
            h5_data = nfr.extract_h5_data(self.hdf5_path)
            self.tracks = h5_data.get("tracks")
            self.scores = h5_data.get("scores")
            self.node_names = h5_data.get("node_names", [])

        if self.session_id:
            try:
                subj_id = db_access.get_sess_subj_id(self.session_id)
                print(f"DB Lookup: Session '{self.session_id}' belongs to Subject '{subj_id}'")
                
                subj_sess_dict = {subj_id: [int(self.session_id)]}
                raw_fp_data, _ = fpah.load_fp_data(loc_db, subj_sess_dict)
                print("Finished loading and continued")
                
                subj_dict = list(raw_fp_data.values())[0]
                fp_data = list(subj_dict.values())[0]
                self.fp_data = fp_data
                
                self.ts_timestamps = fp_data['time']
                
                processed_signals = fp_data.get('processed_signals', {})
                signal_type = 'raw_lig' 
                
                plot_idx = 0
                print(processed_signals.keys())
                for region in processed_signals.keys():
                    print(region)
                    if plot_idx >= 4:
                        break 
                    if region == "DLS":
                        print(processed_signals[region].keys())
                    if signal_type in processed_signals[region]:
                        signal_array = processed_signals[region][signal_type]
                        
                        if type(signal_array).__module__.startswith('cupy'):
                            signal_array = signal_array.get()
                        else:
                            signal_array = np.array(signal_array)
                            
                        print(len(signal_array))
                        print(signal_array)
                        self.time_series[plot_idx] = signal_array
                        
                        self.plot_titles[plot_idx] = f"{region}: {signal_type}"
                        
                        print(f"Mapped '{region}' ({signal_type}) to Graph {plot_idx+1}")
                        plot_idx += 1
                        
            except Exception as e:
                import traceback
                print(f"Failed to fetch FP data: {repr(e)}")
                traceback.print_exc()
                
        if len(self.ts_timestamps) == 0:
            print("Using video timestamps as fallback for UI.")
            self.ts_timestamps = self.video_timestamps

    def load_mock_data(self):
        self.ts_timestamps = np.linspace(0, 100, 10000)
        self.time_series[0] = np.sin(self.ts_timestamps) 
        self.time_series[1] = np.cos(self.ts_timestamps * 0.5) 
        self.time_series[2] = np.sin(self.ts_timestamps * 2) * np.exp(-self.ts_timestamps/20)
        self.time_series[3] = np.random.normal(0, 0.2, 10000) + np.sin(self.ts_timestamps*0.1)
        self.time_series[4] = np.cos(self.ts_timestamps) ** 2
        
        self.video_timestamps = np.linspace(0, 100, 3000)
        self.get_frame(0)

    def get_frame(self, idx):
        self.current_frame_idx = idx
        
        if self.cap is not None and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read()
            if ret:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
        return np.random.randint(0, 255, (300, 400), dtype=np.uint8)

# --- NEW ANNOTATION DIALOG ---
class EventAnnotationDialog(QDialog):
    """Popup window to log a new event."""
    def __init__(self, parent, frame_idx, ts_idx, available_tags):
        super().__init__(parent)
        self.setWindowTitle("Log Event")
        self.resize(300, 200)
        
        self.frame_idx = frame_idx
        self.ts_idx = ts_idx
        self.available_tags = available_tags
        self.selected_tag = None
        
        layout = QFormLayout()
        
        self.frame_label = QLabel(str(frame_idx))
        self.ts_label = QLabel(str(ts_idx))
        layout.addRow("Video Frame:", self.frame_label)
        layout.addRow("Photometry Index:", self.ts_label)
        
        self.tag_dropdown = QComboBox()
        self.tag_dropdown.addItems(self.available_tags)
        self.tag_dropdown.setEditable(True) # Allow user to type new tags
        layout.addRow("Tag:", self.tag_dropdown)
        
        self.sub_id_input = QLineEdit()
        layout.addRow("Sub-identifier:", self.sub_id_input)
        
        self.notes_input = QLineEdit()
        layout.addRow("Notes:", self.notes_input)
        
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)
        
        self.setLayout(layout)
        
    def get_data(self):
        """Returns the data entered in the dialog."""
        return {
            "frame_index": self.frame_idx,
            "photometry_index": self.ts_idx,
            "tag": self.tag_dropdown.currentText().strip(),
            "sub_id": self.sub_id_input.text().strip(),
            "notes": self.notes_input.text().strip()
        }
# -----------------------------

class MainWindow(QMainWindow):
    def __init__(self, base_video_dir, base_hdf5_dir, base_doric_dir):
        super().__init__()
        self.setWindowTitle("App")
        self.resize(1200, 800)

        self.base_video_dir = base_video_dir
        self.base_hdf5_dir = base_hdf5_dir
        self.base_doric_dir = base_doric_dir

        self.data = DataModel()
        self.lines = []
        self._updating_lines = False
        self.ts_curves = [] 
        
        # Playback variables
        self.is_playing = False
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self.next_frame_auto)
        
        # Annotation variables
        self.annotation_tags = ["poke"]
        self.annotations_df = None

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.tab_session = QWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()

        self.tabs.addTab(self.tab_session, "Session")
        self.tabs.addTab(self.tab1, "Main")
        self.tabs.addTab(self.tab2, "Time Series")
        self.tabs.addTab(self.tab3, "Overlay Setting")

        self.bottom_plot = None
        self.side_plots = []

        self.setup_tab_session()
        self.setup_tab1()
        self.setup_tab2()
        self.setup_tab3()

        self.data.load_mock_data()
        self.refresh_plots_from_model()

    def setup_tab_session(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        
        title_label = QLabel("Enter Session ID")
        title_label.setAlignment(Qt.AlignCenter)
        font = title_label.font()
        font.setPointSize(16)
        title_label.setFont(font)
        
        self.session_input = QLineEdit()
        self.session_input.setPlaceholderText("e.g. 116498")
        self.session_input.setFixedWidth(300)
        self.session_input.setAlignment(Qt.AlignCenter)
        self.session_input.returnPressed.connect(self.on_submit_session)
        
        self.btn_submit_session = QPushButton("Submit")
        self.btn_submit_session.setFixedWidth(150)
        self.btn_submit_session.clicked.connect(self.on_submit_session)
        
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_submit_session, alignment=Qt.AlignCenter)

        layout.addWidget(title_label)
        layout.addSpacing(10)
        layout.addWidget(self.session_input, alignment=Qt.AlignCenter)
        layout.addSpacing(10)
        layout.addLayout(btn_layout)
        
        self.tab_session.setLayout(layout)

    def on_submit_session(self):
        session_id = self.session_input.text().strip()
        if not session_id:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid Session ID.")
            return
            
        self.data.session_id = session_id
        
        mp4_path = self.data.find_file_by_session(self.base_video_dir, session_id, ".mp4")
        
        hdf5_path = self.data.find_file_by_session(self.base_hdf5_dir, session_id, ".hdf5")
        if hdf5_path is None:
            hdf5_path = self.data.find_file_by_session(self.base_hdf5_dir, session_id, ".h5")
            
        doric_path = self.data.find_file_by_session(self.base_doric_dir, session_id, ".doric")
        
        missing_files = []
        if mp4_path is None: missing_files.append("• .mp4 (Video file)")
        if hdf5_path is None: missing_files.append("• .hdf5/.h5 (Labels file)")
        if doric_path is None: missing_files.append("• .doric (Photometry file)")
            
        if missing_files:
            error_msg = f"The following files could not be found for session '{session_id}':\n\n"
            error_msg += "\n".join(missing_files)
            QMessageBox.warning(self, "Missing Files", error_msg)
        else:
            self.data.video_path = mp4_path
            self.data.hdf5_path = hdf5_path
            self.data.doric_path = doric_path
            
            # --- NEW ANNOTATION LOAD ---
            self.load_annotations()
            # ---------------------------
            
            QMessageBox.information(self, "Success", f"All files located for session '{session_id}'.")
            
            self.data.load_real_session_data()
            self.refresh_plots_from_model()
            self.setFocus() 

    # --- NEW ANNOTATION LOGIC ---
    def get_annotation_csv_path(self):
        """Generates the save path for the CSV next to the video file."""
        if not self.data.video_path:
            return None
        video_dir = os.path.dirname(self.data.video_path)
        return os.path.join(video_dir, f"{self.data.session_id}_annotations.csv")

    def load_annotations(self):
        """Loads existing annotations from CSV if they exist, to populate dropdown."""
        csv_path = self.get_annotation_csv_path()
        if csv_path and os.path.exists(csv_path):
            try:
                self.annotations_df = pd.read_csv(csv_path)
                # Update available tags based on previous entries
                existing_tags = self.annotations_df["tag"].dropna().unique().tolist()
                for t in existing_tags:
                    if t not in self.annotation_tags:
                        self.annotation_tags.append(t)
            except Exception as e:
                print(f"Error loading annotations: {e}")
                self.annotations_df = pd.DataFrame(columns=["frame_index", "photometry_index", "tag", "sub_id", "notes"])
        else:
            self.annotations_df = pd.DataFrame(columns=["frame_index", "photometry_index", "tag", "sub_id", "notes"])

    def add_annotation(self):
        """Triggers the popup to add an annotation for the current frame."""
        if len(self.data.video_timestamps) == 0:
            QMessageBox.warning(self, "No Data", "Load a session first.")
            return
            
        frame_idx = self.data.current_frame_idx
        
        # Find nearest photometry index
        video_time = self.data.video_timestamps[frame_idx]
        ts_idx = 0
        if len(self.data.ts_timestamps) > 0:
            idx = np.searchsorted(self.data.ts_timestamps, video_time)
            if idx == 0:
                ts_idx = 0
            elif idx == len(self.data.ts_timestamps):
                ts_idx = idx - 1
            else:
                left_diff = video_time - self.data.ts_timestamps[idx - 1]
                right_diff = self.data.ts_timestamps[idx] - video_time
                ts_idx = idx - 1 if left_diff < right_diff else idx
        else:
            ts_idx = frame_idx # Fallback

        dialog = EventAnnotationDialog(self, frame_idx, ts_idx, self.annotation_tags)
        if dialog.exec_():
            new_data = dialog.get_data()
            
            # If the user typed a new tag, add it to our tracking list
            if new_data["tag"] and new_data["tag"] not in self.annotation_tags:
                self.annotation_tags.append(new_data["tag"])
            
            # Save to dataframe
            new_row = pd.DataFrame([new_data])
            if self.annotations_df.empty:
                self.annotations_df = new_row
            else:
                self.annotations_df = pd.concat([self.annotations_df, new_row], ignore_index=True)
                
            # Save to CSV immediately
            csv_path = self.get_annotation_csv_path()
            if csv_path:
                try:
                    self.annotations_df.to_csv(csv_path, index=False)
                    print(f"Saved annotation to {csv_path}")
                except Exception as e:
                    QMessageBox.critical(self, "Save Error", f"Could not save annotations: {e}")
    # ----------------------------

    def setup_tab1(self):
        layout = QVBoxLayout()
        top_layout = QHBoxLayout()
        
        # Video Container Layout
        video_layout = QVBoxLayout()
        
        self.main_plot = pg.PlotWidget(title="Video / Main Display")
        self.main_plot.setAspectLocked(True)
        self.main_plot.hideAxis('bottom')
        self.main_plot.hideAxis('left')
        
        self.video_image_item = pg.ImageItem()
        self.main_plot.addItem(self.video_image_item)
        
        self.pose_scatter = pg.ScatterPlotItem(
            size=6, 
            pen=pg.mkPen(None), 
            brush=pg.mkBrush(255, 0, 0, 150) 
        )
        self.main_plot.addItem(self.pose_scatter)
        
        btn_controls_layout = QHBoxLayout()
        
        self.btn_play_pause = QPushButton("Play")
        self.btn_play_pause.clicked.connect(self.toggle_playback)
        self.btn_play_pause.setToolTip("Press Spacebar to toggle play/pause")
        
        # --- NEW ANNOTATION BUTTON ---
        self.btn_annotate = QPushButton("Add Event Tag")
        self.btn_annotate.clicked.connect(self.add_annotation)
        self.btn_annotate.setToolTip("Press 'A' to quickly add a tag to this frame")
        # -----------------------------
        
        btn_controls_layout.addWidget(self.btn_play_pause)
        btn_controls_layout.addWidget(self.btn_annotate)
        
        video_layout.addWidget(self.main_plot, stretch=1)
        video_layout.addLayout(btn_controls_layout)
        
        top_layout.addLayout(video_layout, stretch=3)
        
        side_plots_layout = QVBoxLayout()
        self.side_plots = []
        for i in range(2, 6):
            small_plot = pg.PlotWidget(title=f"Graph {i}")
            small_plot.hideAxis('bottom')
            small_plot.hideAxis('left')
            
            curve = small_plot.plot(
                pen='c', 
                clipToView=True, 
                autoDownsample=True, 
                downsampleMethod='subsample'
            )
            self.ts_curves.append(curve)
            
            self.side_plots.append(small_plot)
            side_plots_layout.addWidget(small_plot)
            
        top_layout.addLayout(side_plots_layout, stretch=1)
        layout.addLayout(top_layout, stretch=3)
        
        self.bottom_plot = pg.PlotWidget(title="Waveform / Plot (1)")
        bottom_curve = self.bottom_plot.plot(pen='y')
        self.ts_curves.insert(0, bottom_curve)
        layout.addWidget(self.bottom_plot, stretch=1)
        
        self.tab1.setLayout(layout)
        self.setup_indicator_lines()

    def setup_indicator_lines(self):
        all_plots = [self.bottom_plot] + self.side_plots
        for plot in all_plots:
            line = pg.InfiniteLine(angle=90, movable=True, pen='r')
            line.sigPositionChangeFinished.connect(self.on_line_dragged)
            plot.addItem(line)
            self.lines.append(line)

    def setup_tab2(self):
        layout = QVBoxLayout()
        self.key_inputs = []
        for i in range(1, 6):
            row_layout = QHBoxLayout()
            number_label = QLabel(f"({i})")
            key_input = QLineEdit(f"key {i}") if i <= 2 else QLineEdit()
            self.key_inputs.append(key_input)
            row_layout.addWidget(number_label)
            row_layout.addWidget(key_input)
            row_layout.addStretch()
            layout.addLayout(row_layout)
        layout.addStretch()
        self.tab2.setLayout(layout)

    def setup_tab3(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        top_layout = QVBoxLayout()
        top_layout.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        
        self.show_score_cb = QCheckBox("Show Score")
        self.show_score_cb.stateChanged.connect(lambda: self.display_video_frame(self.data.current_frame_idx))
        top_layout.addWidget(self.show_score_cb, alignment=Qt.AlignHCenter)
        
        self.position_text_box = QLineEdit("Position Keys Here")
        self.position_text_box.setAlignment(Qt.AlignCenter)
        top_layout.addWidget(self.position_text_box, alignment=Qt.AlignHCenter)
        
        self.btn_show_qc = QPushButton("Show Tracking QC Summary")
        self.btn_show_qc.clicked.connect(self.show_tracking_summary)
        top_layout.addWidget(self.btn_show_qc, alignment=Qt.AlignHCenter)
        
        self.btn_reset_views = QPushButton("Reset Plot Views")
        self.btn_reset_views.clicked.connect(self.reset_plot_views)
        top_layout.addWidget(self.btn_reset_views, alignment=Qt.AlignHCenter)
        
        layout.addLayout(top_layout)
        layout.addSpacing(30)
        
        nodes_label = QLabel("Nodes")
        layout.addWidget(nodes_label)
        
        self.nodes_grid_layout = QGridLayout()
        self.nodes_grid_layout.setAlignment(Qt.AlignLeft)
        
        self.node_checkboxes = []
        self.node_text_items = []
            
        layout.addLayout(self.nodes_grid_layout)
        layout.addStretch()
        self.tab3.setLayout(layout)
        
    def reset_plot_views(self):
        if self.bottom_plot is None:
            return
            
        all_plots = [self.bottom_plot] + self.side_plots
        for plot in all_plots:
            plot.autoRange()
    
    def build_dynamic_node_ui(self):
        if not self.data.node_names:
            return

        for i in reversed(range(self.nodes_grid_layout.count())): 
            widget = self.nodes_grid_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
                
        for text_item in self.node_text_items:
            self.main_plot.removeItem(text_item)
            
        self.node_checkboxes = []
        self.node_text_items = []
        
        for i, name in enumerate(self.data.node_names):
            cb = QCheckBox(name)
            cb.setChecked(True) 
            cb.stateChanged.connect(lambda state, idx=i: self.display_video_frame(self.data.current_frame_idx))
            self.node_checkboxes.append(cb)
            
            row = i // 4
            col = i % 4
            self.nodes_grid_layout.addWidget(cb, row, col)
            
            text_item = pg.TextItem(text="", color=(0, 255, 255), anchor=(0, 1)) 
            text_item.hide()
            self.main_plot.addItem(text_item)
            self.node_text_items.append(text_item)
        
    def refresh_plots_from_model(self):
        self.build_dynamic_node_ui()
        
        if len(self.data.ts_timestamps) == 0:
            for curve in self.ts_curves:
                curve.setData([], [])
            return
            
        all_plot_widgets = [self.bottom_plot] + self.side_plots
        
        for i, curve in enumerate(self.ts_curves):
            if i < len(self.data.time_series) and len(self.data.time_series[i]) > 0:
                y_data = self.data.time_series[i]
                x_data = self.data.ts_timestamps
                
                if len(x_data) == len(y_data):
                    curve.setData(x_data, y_data)
                    
                    if hasattr(self.data, 'plot_titles') and i < len(self.data.plot_titles) and self.data.plot_titles[i]:
                        all_plot_widgets[i].setTitle(self.data.plot_titles[i])
                    else:
                        all_plot_widgets[i].setTitle(f"Graph {i + 1}" if i > 0 else "Waveform / Plot (1)")
                else:
                    print(f"Warning: Graph {i+1} mismatch! X: {len(x_data)}, Y: {len(y_data)}")
                    curve.setData([], []) 
            else:
                curve.setData([], []) 
                
        self.display_video_frame(self.data.current_frame_idx)

    # --- PLAYBACK METHODS ---
    def toggle_playback(self):
        if len(self.data.video_timestamps) == 0:
            return
            
        self.is_playing = not self.is_playing
        
        if self.is_playing:
            self.btn_play_pause.setText("Pause")
            fps = 30
            if self.data.cap is not None and self.data.cap.isOpened():
                cap_fps = self.data.cap.get(cv2.CAP_PROP_FPS)
                if cap_fps > 0 and not np.isnan(cap_fps):
                    fps = cap_fps
            self.playback_timer.start(int(1000 / fps))
        else:
            self.btn_play_pause.setText("Play")
            self.playback_timer.stop()

    def next_frame_auto(self):
        if self.data.current_frame_idx >= len(self.data.video_timestamps) - 1:
            self.toggle_playback()
            return
        self.step_frame(1)
    # ------------------------

    def on_line_dragged(self, source_line):
        if self._updating_lines or len(self.data.video_timestamps) == 0:
            return

        target_time = source_line.value()

        self._updating_lines = True
        for line in self.lines:
            if line != source_line:
                line.setValue(target_time)
        self._updating_lines = False

        idx = np.searchsorted(self.data.video_timestamps, target_time)
        
        if idx == 0:
            closest_idx = 0
        elif idx == len(self.data.video_timestamps):
            closest_idx = idx - 1
        else:
            left_diff = target_time - self.data.video_timestamps[idx - 1]
            right_diff = self.data.video_timestamps[idx] - target_time
            closest_idx = idx - 1 if left_diff < right_diff else idx
            
        self.display_video_frame(closest_idx)

    def display_video_frame(self, idx):
        if len(self.data.video_timestamps) == 0:
            return
            
        img = self.data.get_frame(idx)
        self.video_image_item.setImage(np.swapaxes(img, 0, 1))
        
        if self.data.tracks is not None and idx < self.data.tracks.shape[0]:
            frame_coords = self.data.tracks[idx, :, :, 0] 
            
            frame_scores = None
            if self.data.scores is not None and idx < self.data.scores.shape[0]:
                frame_scores = self.data.scores[idx, :, 0] 
                
            valid_scatter_pts = []
            show_score = self.show_score_cb.isChecked()
            
            for i, name in enumerate(self.data.node_names):
                if i >= len(self.node_checkboxes):
                    break
                    
                if not self.node_checkboxes[i].isChecked():
                    self.node_text_items[i].hide()
                    continue
                    
                x, y = frame_coords[i, 0], frame_coords[i, 1]
                
                if not np.isnan(x) and not np.isnan(y):
                    valid_scatter_pts.append({'pos': (x, y)})
                    
                    if show_score and frame_scores is not None:
                        score_val = frame_scores[i]
                        self.node_text_items[i].setText(f"{name}: {score_val:.2f}")
                        self.node_text_items[i].setPos(x, y)
                        self.node_text_items[i].show()
                    else:
                        self.node_text_items[i].hide()
                else:
                    self.node_text_items[i].hide()
                    
            self.pose_scatter.setData(valid_scatter_pts)

    def keyPressEvent(self, event):
        if len(self.data.video_timestamps) == 0:
            super().keyPressEvent(event) 
            return

        if event.key() == Qt.Key_Right:
            self.step_frame(1)
        elif event.key() == Qt.Key_Left:
            self.step_frame(-1)
        elif event.key() == Qt.Key_Space:
            self.toggle_playback()
        # --- NEW ANNOTATION HOTKEY ---
        elif event.key() == Qt.Key_A:
            self.add_annotation()
        # -----------------------------
        else:
            super().keyPressEvent(event)

    def step_frame(self, step):
        new_idx = self.data.current_frame_idx + step
        max_idx = len(self.data.video_timestamps) - 1
        new_idx = max(0, min(new_idx, max_idx))
        
        self.display_video_frame(new_idx)
        video_time = self.data.video_timestamps[new_idx]
        
        if len(self.data.ts_timestamps) > 0:
            idx = np.searchsorted(self.data.ts_timestamps, video_time)
            
            if idx == 0:
                closest_ts_idx = 0
            elif idx == len(self.data.ts_timestamps):
                closest_ts_idx = idx - 1
            else:
                left_diff = video_time - self.data.ts_timestamps[idx - 1]
                right_diff = self.data.ts_timestamps[idx] - video_time
                closest_ts_idx = idx - 1 if left_diff < right_diff else idx
                
            snapped_time = self.data.ts_timestamps[closest_ts_idx]
        else:
            snapped_time = video_time 
        
        self._updating_lines = True
        for line in self.lines:
            line.setValue(snapped_time)
        self._updating_lines = False

    def show_tracking_summary(self):
        if self.data.tracks is None or self.data.scores is None:
            QMessageBox.warning(self, "No Data", "Please load a valid session first.")
            return

        coords = self.data.tracks[:, :, :, 0]
        scores = self.data.scores[:, :, 0]
        nodes = self.data.node_names

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Tracking QC Summary: {self.data.session_id}", fontsize=16, fontweight='bold')

        nan_counts = np.sum(np.isnan(coords[:, :, 0]), axis=0)
        prop_nans = nan_counts / coords.shape[0]
        
        axes[0].bar(nodes, prop_nans, color="steelblue")
        axes[0].set_title("Proportion of Missing Frames")
        axes[0].set_ylabel("Proportion (NaNs / Total Frames)")
        axes[0].tick_params(axis='x', rotation=45)

        all_gap_lengths = []
        for i in range(len(nodes)):
            is_nan = np.isnan(coords[:, i, 0])
            starts = np.where(is_nan & ~np.roll(is_nan, 1))[0]
            stops = np.where(~is_nan & np.roll(is_nan, 1))[0]
            
            if len(is_nan) > 0:
                if is_nan[0]: starts = np.insert(starts, 0, 0)
                if is_nan[-1]: stops = np.append(stops, len(is_nan))
            
            all_gap_lengths.extend(stops - starts)

        if all_gap_lengths:
            axes[1].hist(all_gap_lengths, bins=30, color="indianred")
            axes[1].set_yscale('log')
        axes[1].set_title("Distribution of Gap Lengths (All Nodes)")
        axes[1].set_xlabel("Gap Length (Frames)")
        axes[1].set_ylabel("Frequency (Log Scale)")

        score_records = []
        for i, node in enumerate(nodes):
            valid_scores = scores[:, i][~np.isnan(scores[:, i])]
            for s in valid_scores:
                score_records.append({"Node": node, "Score": s})

        if score_records:
            score_df = pd.DataFrame(score_records)
            sns.violinplot(
                data=score_df, x="Node", y="Score", ax=axes[2], 
                inner="quartile", color="mediumseagreen", density_norm="width"
            )
        axes[2].set_title("Prediction Confidence Score Distributions")
        axes[2].set_ylabel("Confidence Score")
        axes[2].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    video_dir = os.path.expanduser(r"C:\Users\cns-th-lab\TannerVidsRenamed")
    #Multiple models so exact modelfolders must be specified in a list
    hdf5_dir = os.path.expanduser(r"C:\Users\cns-th-lab\TannerVidsRenamed\198\Videos\predictions\260731_198_199x_234x_235x_237x_238x_274x_400x_402x_419x_421x_422x_424x_483x_occin")
    doric_dir = os.path.expanduser(r"C:\Users\cns-th-lab\TannerVidsRenamed")
    
    window = MainWindow(video_dir, hdf5_dir, doric_dir)
    window.show()
    sys.exit(app.exec())