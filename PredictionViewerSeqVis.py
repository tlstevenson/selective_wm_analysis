# -*- coding: utf-8 -*-
"""
Updated Prediction Viewer (Fallback & Transformation Hierarchy + Segment Export)
"""
import cv2
import numpy as np
import os

def RunApp(video_path, tracks_coords, node_names, scores, output_window, fps, bad_sequences=None, fallbacks=None, transformations=None):
    """
    Args:
        video_path: Path to the video file.
        
        tracks_coords: Numpy array of base raw coordinates.
        
        node_names: List of strings representing the names of the tracked nodes.
        
        scores: Numpy array of prediction scores.
        
        output_window: String name of the OpenCV window.
        
        fps: Frames per second of the video.
        
        bad_sequences: A list of tuples containing (start_frame, end_frame).
        
        fallbacks: A list of numpy arrays representing node interpolation 
                   that is not present in the raw data.
                   
        transformations: A list of numpy arrays representing the time series 
                         to overlay regardless of raw/fallback status.
    """
    if bad_sequences is None:
        bad_sequences = []
    if fallbacks is None:
        fallbacks = []
    if transformations is None:
        transformations = []
        
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frame_idx = 0
    frame_interval = 1
    seq_idx = -1 
    
    # State trackers for visibility
    active_fallbacks = [True] * len(fallbacks)
    active_transformations = [True] * len(transformations)
    
    # Pre-defined list of distinct BGR colors 
    palette = [
        (255, 0, 0),    # Blue
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 165, 255),  # Orange
        (255, 255, 0),  # Cyan
        (255, 255, 255),# White
        (128, 0, 128),  # Purple
        (0, 0, 255),    # Red
        (128, 128, 0),  # Teal
        (0, 128, 255)   # Amber
    ]

    # --- HELPER FUNCTION TO DRAW OVERLAYS ---
    def draw_overlays(frame_to_draw, current_f_idx, current_seq_idx):
        # Get frame data safely
        try:
            current_pos = tracks_coords[current_f_idx]
            current_scores = scores[current_f_idx]
            num_nodes = np.shape(current_pos)[0]
        except IndexError:
            current_pos = []
            current_scores = []
            num_nodes = len(node_names)
    
        for i in range(num_nodes):
            node_drawn = False
            
            # 1. Try Base Raw Tracking (Green) First
            try:
                x, y = current_pos[i, 0], current_pos[i, 1]
                if isinstance(x, np.ndarray): 
                    x, y = x[0], y[0]
                    
                if not np.isnan(x) and not np.isnan(y) and x > 0 and y > 0:
                    x, y = int(x), int(y)
                    cv2.circle(frame_to_draw, center=(x, y), radius=5, color=(0, 255, 0), thickness=-1)
                    
                    score_val = current_scores[i][0] if isinstance(current_scores[i], (list, np.ndarray)) else current_scores[i]
                    cv2.putText(frame_to_draw, f"{node_names[i]}: {round(score_val,2)}", (x + 8, y - 8), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    node_drawn = True
            except IndexError:
                pass 
                
            # 2. If Raw Tracking is invalid/missing, fallback in order
            if not node_drawn:
                for f_idx, fb_series in enumerate(fallbacks):
                    if active_fallbacks[f_idx]:
                        try:
                            curr_fb = fb_series[current_f_idx]
                            x, y = curr_fb[i, 0], curr_fb[i, 1]
                            
                            if isinstance(x, np.ndarray):
                                x, y = x[0], y[0]
                                
                            if not np.isnan(x) and not np.isnan(y) and x > 0 and y > 0:
                                x, y = int(x), int(y)
                                color = palette[f_idx % len(palette)]
                                cv2.circle(frame_to_draw, center=(x, y), radius=5, color=color, thickness=-1)
                                
                                cv2.putText(frame_to_draw, f"{node_names[i]}", (x + 8, y - 8), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                                            
                                node_drawn = True
                                break # Stop searching! We found the first valid fallback.
                        except IndexError:
                            pass
                            
            # 3. Independent Transformations (Drawn regardless of raw/fallback status)
            for t_idx, trans_series in enumerate(transformations):
                if active_transformations[t_idx]:
                    try:
                        curr_trans = trans_series[current_f_idx]
                        x, y = curr_trans[i, 0], curr_trans[i, 1]
                        
                        if isinstance(x, np.ndarray):
                            x, y = x[0], y[0]
                            
                        if not np.isnan(x) and not np.isnan(y) and x > 0 and y > 0:
                            x, y = int(x), int(y)
                            
                            # Shift the color slightly or use the same palette, drawn as a hollow square
                            color = palette[(t_idx + 3) % len(palette)] 
                            cv2.drawMarker(frame_to_draw, position=(x, y), color=color, 
                                           markerType=cv2.MARKER_SQUARE, markerSize=10, thickness=2)
                    except IndexError:
                        pass
    
        # 4. Overlay frame, time, and sequence info
        current_time = current_f_idx / fps
        cv2.putText(frame_to_draw, f"Frame: {current_f_idx}/{total_frames - 1} | Time: {current_time:.2f}s", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if bad_sequences and current_seq_idx != -1:
            curr_seq = bad_sequences[current_seq_idx]
            cv2.putText(frame_to_draw, f"Seq {current_seq_idx + 1}/{len(bad_sequences)}: Frames {curr_seq[0]}-{curr_seq[1]}", 
                        (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return frame_to_draw
    # ----------------------------------------
    
    print("--- Controls ---")
    print("[D] / Right Arrow : Next Frame")
    print("[A] / Left Arrow  : Previous Frame")
    print("[I]               : Change Interval")
    print("[F]               : Jump to specific Frame")
    print("[S]               : Jump to specific Second")
    print("[N]               : Jump to Next Sequence")
    print("[P]               : Jump to Previous Sequence")
    print("[T]               : Toggle a Transformation overlay")
    print("[W]               : Export Video Segment (Writes current active overlays)")
    print("[Q]               : Quit")
    
    if len(fallbacks) > 0:
        print("\n--- Fallback Toggles ---")
        for i in range(min(len(fallbacks), 10)):
            print(f"[{i}] : Toggle Fallback Level {i} (Color index {i})")
            
    if len(transformations) > 0:
        print("\n--- Transformations ---")
        print(f"Loaded {len(transformations)} transformation overlays. Use [T] to toggle.")
        
    print("----------------")
    
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame.")
            break
            
        # Draw all elements using the helper function
        frame = draw_overlays(frame, frame_idx, seq_idx)
        cv2.imshow(output_window, frame)
        
        # 5. Keyboard Navigation
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('d'): 
            if frame_idx < total_frames - frame_interval:
                frame_idx += frame_interval
                
        elif key == ord('a'): 
            if frame_idx > frame_interval:
                frame_idx -= frame_interval
                
        elif key == ord('n'): 
            if len(bad_sequences) > 0:
                seq_idx = (seq_idx + 1) % len(bad_sequences)
                frame_idx = bad_sequences[seq_idx][0]
                print(f"Jumped to Sequence {seq_idx + 1}: Frame {frame_idx}")

        elif key == ord('p'): 
            if len(bad_sequences) > 0:
                if seq_idx == -1: 
                    seq_idx = len(bad_sequences) - 1
                else:
                    seq_idx = (seq_idx - 1) % len(bad_sequences)
                frame_idx = bad_sequences[seq_idx][0]
                print(f"Jumped to Sequence {seq_idx + 1}: Frame {frame_idx}")
                
        elif key == ord('i'):
            try:
                target_interval = int(input("\nEnter new amount of frames to skip by (1-300): "))
                if 1 <= target_interval < 300:
                    frame_interval = target_interval
            except ValueError:
                pass
                
        elif key == ord('f'):
            try:
                target_frame = int(input(f"\nEnter target frame (0 to {total_frames - 1}): "))
                if 0 <= target_frame < total_frames:
                    frame_idx = target_frame
                    seq_idx = -1
            except ValueError:
                pass
                
        elif key == ord('s'):
            try:
                target_sec = float(input(f"\nEnter target second (0 to {(total_frames - 1) / fps:.2f}): "))
                target_frame = int(target_sec * fps)
                if 0 <= target_frame < total_frames:
                    frame_idx = target_frame
                    seq_idx = -1
            except ValueError:
                pass
                
        elif key == ord('t'):
            if len(transformations) > 0:
                try:
                    t_idx = int(input(f"\nEnter Transformation index to toggle (0 to {len(transformations) - 1}): "))
                    if 0 <= t_idx < len(transformations):
                        active_transformations[t_idx] = not active_transformations[t_idx]
                        print(f"Transformation [{t_idx}] toggled: {active_transformations[t_idx]}")
                except ValueError:
                    pass

        elif key == ord('w'): # WRITE/EXPORT VIDEO SEGMENT
            print("\n--- Exporting Video Segment ---")
            try:
                start_f = int(input(f"Enter start frame (0 to {total_frames - 1}): "))
                end_f = int(input(f"Enter end frame ({start_f + 1} to {total_frames}): "))
                
                # Basic validation to keep inputs within video bounds
                start_f = max(0, min(start_f, total_frames - 1))
                end_f = max(start_f + 1, min(end_f, total_frames))
                
                out_name = input("Enter output filename (default: output_labeled.mp4): ")
                if not out_name.strip():
                    out_name = "output_labeled.mp4"
                    
                fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4
                out_writer = cv2.VideoWriter(out_name, fourcc, fps, (vid_width, vid_height))
                
                print(f"Writing frames {start_f} to {end_f-1} to {out_name}... Please wait.")
                total_export_frames = end_f - start_f
                
                for export_f in range(start_f, end_f):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, export_f)
                    ret, export_frame = cap.read()
                    if not ret: 
                        break
                    
                    # Apply overlays to the clean frame
                    export_frame = draw_overlays(export_frame, export_f, -1)
                    out_writer.write(export_frame)
                    
                    # Progress indicator in console
                    frames_done = export_f - start_f + 1
                    if frames_done % max(1, (total_export_frames // 10)) == 0:
                        print(f"Exported {frames_done}/{total_export_frames} frames ({(frames_done/total_export_frames)*100:.1f}%)")
                        
                out_writer.release()
                print(f"Done! Video saved to {os.path.abspath(out_name)}\n")
                
            except ValueError:
                print("Invalid frame input. Video export aborted.")
            
            # Put the video capture back to where the user was looking
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        elif key == ord('q'): 
            break
            
        # Dynamically map keys '0' through '9' to the fallback list
        elif ord('0') <= key <= ord('9'):
            f_idx = key - ord('0')
            if f_idx < len(fallbacks):
                active_fallbacks[f_idx] = not active_fallbacks[f_idx]
                print(f"Fallback Level [{f_idx}] toggled: {active_fallbacks[f_idx]}")
    
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)