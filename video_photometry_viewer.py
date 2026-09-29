# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 14:34:37 2026

@author: cns-th-lab
"""

import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QTabWidget, 
                             QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, 
                             QPushButton, QLineEdit, QCheckBox, QMessageBox)
from PyQt6.QtCore import Qt
import pyqtgraph as pg
import numpy as np
import neurofileread as nfr
import init
from hankslab_db import db_access
from hankslab_db import basicRLtasks_db as bandit_db
import cv2


class DataModel:
    """Interface class that holds and manages all data for the UI."""
    def __init__(self):
        self.session_id = None
        self.video_path = None
        self.hdf5_path = None
        self.doric_path = None
        
        # 1D Arrays for timestamps
        self.video_timestamps = np.array([])
        self.ts_timestamps = np.array([])
        
        # Current state
        self.current_frame_idx = 0
        self.current_image = None
        
        # List of 5 1D numpy arrays for the 5 plots (1 bottom + 4 side)
        self.time_series = [np.array([]) for _ in range(5)]
        self.tracks = None
        self.scores = None
        self.node_names = []
        self.fp_data = {}
        self.cap = None
        
    def find_file_by_session(self, base_dir, session_id, extension):
        """
        Recursively searches a base directory for a file containing the session ID 
        and matching the specified extension.
        """
        if not os.path.isdir(base_dir):
            print(f"Error: Base directory '{base_dir}' does not exist.")
            return None

        # Ensure extension starts with a dot for accurate comparison
        if not extension.startswith('.'):
            extension = f".{extension}"

        print(f"Searching for {extension} file for session '{session_id}' in {base_dir}...")
        
        for root, _, files in os.walk(base_dir):
            for file in files:
                # Check if the file has the correct extension and contains the session ID
                if file.lower().endswith(extension.lower()) and session_id in file:
                    found_path = os.path.join(root, file)
                    print(f"Found: {found_path}")
                    return found_path
                    
        return None
    
    def load_real_session_data(self):
        """Called after on_submit_session successfully finds the files."""        
        
        loc_db = bandit_db.LocalDB_BasicRLTasks("twoArmBandit")
        
        if self.video_path:
            if self.cap is not None:
                self.cap.release()
            self.cap = cv2.VideoCapture(self.video_path)
        
        if self.doric_path:
            # We still load this to sync the video frames
            self.video_timestamps = nfr.get_doric_timestamps(self.doric_path)
            
        subj_id = None
        if self.hdf5_path:
            h5_data = nfr.extract_h5_data(self.hdf5_path)
            self.tracks = h5_data.get("tracks")
            self.scores = h5_data.get("scores")
            self.node_names = h5_data.get("node_names", [])
            
            # Extract subj_id to build the dictionary for FP data
            #TODO: Remove the reliance on h5 naming conventions
            #h5_data itself gets the subj_id from the name ofthe file
            subj_id = h5_data.get("subj_id") 

        if subj_id and self.session_id:
            print(f"Requesting FP data for Subject: {subj_id}, Session: {self.session_id}")
            try:
                # 1. Load FP data using the dictionary structure and unpack the tuple
                # Note: We are ignoring the second return value (usually metadata/configs) with '_'
                raw_fp_data, _ = nfr.load_fp(loc_db, {subj_id: [self.session_id]})
                
                # 2. Drill down into the nested output dictionary
                fp_data = raw_fp_data[subj_id][self.session_id]
                
                # 3. Extract the high-frequency FP timeline
                self.ts_timestamps = fp_data['time']
                
                # 4. Extract regions and populate the 5 UI time series plots
                processed_signals = fp_data.get('processed_signals', {})
                
                # Define the specific signal you want to plot (e.g., 'dFF', 'zscore', 'raw')
                # Change this variable to match your pipeline's exact naming convention!
                signal_type = 'dFF' 
                
                plot_idx = 0
                for region in processed_signals.keys():
                    if plot_idx >= 5:
                        break # We only have 5 UI plots available
                        
                    if signal_type in processed_signals[region]:
                        # Extract the biological signal
                        signal_array = processed_signals[region][signal_type]
                        
                        # SAFETY CHECK: Your snippet uses fp_utils.to_cupy()
                        # PyQtGraph requires standard NumPy (CPU) arrays. If the data 
                        # is a CuPy array, we must bring it back to the CPU using .get()
                        if type(signal_array).__module__.startswith('cupy'):
                            signal_array = signal_array.get()
                        else:
                            signal_array = np.array(signal_array)
                            
                        # Assign to the UI graphs
                        self.time_series[plot_idx] = signal_array
                        print(f"Mapped '{region}' ({signal_type}) to Graph {plot_idx+1}")
                        plot_idx += 1
                        
            except Exception as e:
                print(f"Failed to load FP data into viewer: {e}")
                # Fallback so the video UI doesn't crash if FP data fails
                self.ts_timestamps = self.video_timestamps

    def load_mock_data(self):
        """Generates dummy data to demonstrate the UI plotting and syncing."""
        self.ts_timestamps = np.linspace(0, 100, 10000)
        self.time_series[0] = np.sin(self.ts_timestamps) 
        self.time_series[1] = np.cos(self.ts_timestamps * 0.5) 
        self.time_series[2] = np.sin(self.ts_timestamps * 2) * np.exp(-self.ts_timestamps/20)
        self.time_series[3] = np.random.normal(0, 0.2, 10000) + np.sin(self.ts_timestamps*0.1)
        self.time_series[4] = np.cos(self.ts_timestamps) ** 2
        
        self.video_timestamps = np.linspace(0, 100, 3000)
        self.get_frame(0)

    def get_frame(self, idx):
        """Retrieves a real video frame, or falls back to static placeholder."""
        self.current_frame_idx = idx
        
        # Attempt to read the real video frame
        if self.cap is not None and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read()
            if ret:
                # Convert OpenCV's BGR to PyQtGraph's expected RGB format
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
        # Fallback to static placeholder if no video is loaded
        return np.random.randint(0, 255, (300, 400), dtype=np.uint8)


class MainWindow(QMainWindow):
    def __init__(self, base_video_dir, base_hdf5_dir, base_doric_dir):
        super().__init__()
        self.setWindowTitle("App")
        self.resize(1200, 800)

        # Store the directories passed from the main block
        self.base_video_dir = base_video_dir
        self.base_hdf5_dir = base_hdf5_dir
        self.base_doric_dir = base_doric_dir

        self.data = DataModel()
        self.lines = []
        self._updating_lines = False
        self.ts_curves = [] 

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Create the four tabs
        self.tab_session = QWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()

        # Add tabs in order
        self.tabs.addTab(self.tab_session, "Session")
        self.tabs.addTab(self.tab1, "Main")
        self.tabs.addTab(self.tab2, "Time Series")
        self.tabs.addTab(self.tab3, "Overlay Setting")

        # Setup individual tab UI
        self.setup_tab_session()
        self.setup_tab1()
        self.setup_tab2()
        self.setup_tab3()

        # Load mock data for demonstration
        self.data.load_mock_data()
        self.refresh_plots_from_model()

    def setup_tab_session(self):
        """Builds the Session ID input tab."""
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        title_label = QLabel("Enter Session ID")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = title_label.font()
        font.setPointSize(16)
        title_label.setFont(font)
        
        self.session_input = QLineEdit()
        self.session_input.setPlaceholderText("e.g. Mouse1_Session3")
        self.session_input.setFixedWidth(300)
        self.session_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.session_input.returnPressed.connect(self.on_submit_session)
        
        self.btn_submit_session = QPushButton("Submit")
        self.btn_submit_session.setFixedWidth(150)
        self.btn_submit_session.clicked.connect(self.on_submit_session)
        
        # Center the submit button
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_submit_session, alignment=Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(title_label)
        layout.addSpacing(10)
        layout.addWidget(self.session_input, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addSpacing(10)
        layout.addLayout(btn_layout)
        
        self.tab_session.setLayout(layout)

    def on_submit_session(self):
        """Saves the Session ID and triggers the file search."""
        session_id = self.session_input.text().strip()
        if not session_id:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid Session ID.")
            return
            
        self.data.session_id = session_id
        print(f"Session ID saved: {self.data.session_id}")
        
        # Trigger the file searches and capture the returned paths
        mp4_path = self.data.find_file_by_session(self.base_video_dir, session_id, ".mp4")
        hdf5_path = self.data.find_file_by_session(self.base_hdf5_dir, session_id, ".hdf5")
        doric_path = self.data.find_file_by_session(self.base_doric_dir, session_id, ".doric")
        
        # Check for missing files
        missing_files = []
        if mp4_path is None:
            missing_files.append("• .mp4 (Video file)")
        if hdf5_path is None:
            missing_files.append("• .hdf5 (Labels file)")
        if doric_path is None:
            missing_files.append("• .doric (Photometry file)")
            
        if missing_files:
            error_msg = f"The following files could not be found for session '{session_id}':\n\n"
            error_msg += "\n".join(missing_files)
            error_msg += "\n\nPlease check the directory structure or session ID."
            
            QMessageBox.warning(self, "Missing Files", error_msg)
        else:
            self.data.video_path = mp4_path
            self.data.hdf5_path = hdf5_path
            self.data.doric_path = doric_path
            
            QMessageBox.information(self, "Success", f"All files located for session '{session_id}'.")
            
            # Load the data and refresh the UI
            self.data.load_real_session_data()
            self.refresh_plots_from_model()
            self.setFocus() # Returns arrow-key focus to the main window

    def setup_tab1(self):
        layout = QVBoxLayout()
        top_layout = QHBoxLayout()
        
        self.main_plot = pg.PlotWidget(title="Video / Main Display")
        self.main_plot.setAspectLocked(True)
        self.main_plot.hideAxis('bottom')
        self.main_plot.hideAxis('left')
        
        # 1. Add the video image layer
        self.video_image_item = pg.ImageItem()
        self.main_plot.addItem(self.video_image_item)
        
        # 2. Add the pose tracking scatter layer on top
        self.pose_scatter = pg.ScatterPlotItem(
            size=6, 
            pen=pg.mkPen(None), 
            brush=pg.mkBrush(255, 0, 0, 150) # Red dots with slight transparency
        )
        self.main_plot.addItem(self.pose_scatter)
        
        top_layout.addWidget(self.main_plot, stretch=3)
        
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
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        top_layout = QVBoxLayout()
        top_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        
        # Checkbox (Connect state change to force a video redraw)
        self.show_score_cb = QCheckBox("Show Score")
        self.show_score_cb.stateChanged.connect(lambda: self.display_video_frame(self.data.current_frame_idx))
        top_layout.addWidget(self.show_score_cb, alignment=Qt.AlignmentFlag.AlignHCenter)
        
        # Text box
        self.position_text_box = QLineEdit("Position Keys Here")
        self.position_text_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top_layout.addWidget(self.position_text_box, alignment=Qt.AlignmentFlag.AlignHCenter)
        
        layout.addLayout(top_layout)
        layout.addSpacing(30)
        
        # Nodes Grid - Leave empty, will populate dynamically
        nodes_label = QLabel("Nodes")
        layout.addWidget(nodes_label)
        
        self.nodes_grid_layout = QGridLayout()
        self.nodes_grid_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        
        # We store the checkboxes and text items as lists
        self.node_checkboxes = []
        self.node_text_items = []
            
        layout.addLayout(self.nodes_grid_layout)
        layout.addStretch()
        self.tab3.setLayout(layout)
        
    def build_dynamic_node_ui(self):
        """Generates checkboxes and text overlays based on loaded node names."""
        if not self.data.node_names:
            return

        # 1. Clear old checkboxes
        for i in reversed(range(self.nodes_grid_layout.count())): 
            widget = self.nodes_grid_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
                
        # 2. Clear old text overlays
        for text_item in self.node_text_items:
            self.main_plot.removeItem(text_item)
            
        self.node_checkboxes = []
        self.node_text_items = []
        
        # 3. Build new UI elements
        for i, name in enumerate(self.data.node_names):
            # Create Checkbox
            cb = QCheckBox(name)
            cb.setChecked(True) # Checked by default
            # Force redraw when toggled
            cb.stateChanged.connect(lambda state, idx=i: self.display_video_frame(self.data.current_frame_idx))
            self.node_checkboxes.append(cb)
            
            # Place in a 4-column grid
            row = i // 4
            col = i % 4
            self.nodes_grid_layout.addWidget(cb, row, col)
            
            # Create hidden text item for the video
            text_item = pg.TextItem(text="", color=(0, 255, 255), anchor=(0, 1)) # Cyan text
            text_item.hide()
            self.main_plot.addItem(text_item)
            self.node_text_items.append(text_item)
        
    def refresh_plots_from_model(self):
        
        self.build_dynamic_node_ui()
        
        if len(self.data.ts_timestamps) == 0:
            return
            
        for i, curve in enumerate(self.ts_curves):
            if i < len(self.data.time_series) and len(self.data.time_series[i]) > 0:
                curve.setData(self.data.ts_timestamps, self.data.time_series[i])
                
        for plot in self.side_plots:
            plot.setXLink(self.bottom_plot)
            
        self.display_video_frame(self.data.current_frame_idx)

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
        # Use swapaxes instead of transpose to handle both 2D and 3D (Color) arrays
        self.video_image_item.setImage(np.swapaxes(img, 0, 1))
        
        if self.data.tracks is not None and idx < self.data.tracks.shape[0]:
            # Extract coordinates and scores for the current frame
            frame_coords = self.data.tracks[idx, :, :, 0] # shape: (nodes, 2)
            
            # Only try to grab scores if they exist
            frame_scores = None
            if self.data.scores is not None and idx < self.data.scores.shape[0]:
                frame_scores = self.data.scores[idx, :, 0] # shape: (nodes)
                
            valid_scatter_pts = []
            show_score = self.show_score_cb.isChecked()
            
            # Iterate through each node dynamically
            for i, name in enumerate(self.data.node_names):
                # Ensure we have a matching checkbox and text item
                if i >= len(self.node_checkboxes):
                    break
                    
                # If node is toggled OFF by the user, hide its text and skip
                if not self.node_checkboxes[i].isChecked():
                    self.node_text_items[i].hide()
                    continue
                    
                x, y = frame_coords[i, 0], frame_coords[i, 1]
                
                # Check for NaN (occluded/untracked node)
                if not np.isnan(x) and not np.isnan(y):
                    valid_scatter_pts.append({'pos': (x, y)})
                    
                    # Update text overlay if toggle is on
                    if show_score and frame_scores is not None:
                        score_val = frame_scores[i]
                        self.node_text_items[i].setText(f"{name}: {score_val:.2f}")
                        self.node_text_items[i].setPos(x, y)
                        self.node_text_items[i].show()
                    else:
                        self.node_text_items[i].hide()
                else:
                    self.node_text_items[i].hide()
                    
            # Push the valid coordinates to the scatter plot layer
            self.pose_scatter.setData(valid_scatter_pts)

    def keyPressEvent(self, event):
        if len(self.data.video_timestamps) == 0:
            super().keyPressEvent(event) 
            return

        if event.key() == Qt.Key.Key_Right:
            self.step_frame(1)
        elif event.key() == Qt.Key.Key_Left:
            self.step_frame(-1)
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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Define the base directories at the execution entry point
    video_dir = os.path.expanduser("~/Documents/HanksLab/Videos")
    hdf5_dir = os.path.expanduser("~/Documents/HanksLab/PoseTracking")
    doric_dir = os.path.expanduser("~/Documents/HanksLab/Photometry")
    
    # Pass the directories into the application window
    window = MainWindow(video_dir, hdf5_dir, doric_dir)
    window.show()
    sys.exit(app.exec())