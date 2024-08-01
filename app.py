import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from datetime import datetime

class CameraError(Exception):
    pass

def open_camera(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise CameraError(f"Camera index {index} not available.")
    return cap

def enhance_sensitivity(diff_frame, scale=20):
    enhanced_frame = cv2.convertScaleAbs(diff_frame, alpha=scale, beta=0)
    return enhanced_frame

def enhance_sensitivity_high(diff_frame, scale=40):
    enhanced_frame = cv2.convertScaleAbs(diff_frame, alpha=scale, beta=0)
    return enhanced_frame

def resize_frame(frame, target_width, target_height):
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

def temporal_smoothing(frame, prev_frames, num_frames=5):
    prev_frames.append(frame)
    if len(prev_frames) > num_frames:
        prev_frames.pop(0)
    avg_frame = np.mean(prev_frames, axis=0).astype(np.uint8)
    return avg_frame, prev_frames

def select_video():
    filepath = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
    video_path.set(filepath)
    video_path_entry.update()  # Update the entry widget with the selected path

def apply_filter(frame, filter_type):
    if filter_type == "Gaussian Blur":
        return cv2.GaussianBlur(frame, (5, 5), 0)
    elif filter_type == "Median Blur":
        return cv2.medianBlur(frame, 5)
    elif filter_type == "Bilateral Filter":
        return cv2.bilateralFilter(frame, 9, 75, 75)
    elif filter_type == "Histogram Equalization":
        if len(frame.shape) == 2:  # Grayscale image
            return cv2.equalizeHist(frame)
        elif len(frame.shape) == 3:  # Color image
            ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        return frame

def process_video(preview=False, live=False):
    global recording, out, start_time

    if live:
        try:
            cap = open_camera(1)
        except CameraError:
            cap = open_camera(0)
        input_video_path = "webcam_feed"
    else:
        input_video_path = video_path.get()
        if not input_video_path:
            messagebox.showerror("Error", "No video file selected.")
            return
        cap = cv2.VideoCapture(input_video_path)

    target_width = int(width.get())
    target_height = int(height.get())
    fps = int(fps_slider.get())
    frame_type = frame_type_var.get()
    sensitivity_type = sensitivity_var.get()
    filter_type = filter_var.get()
    playback_speed = float(playback_speed_slider.get())
    playback_delay = int(1000 / (fps * playback_speed))

    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open video.")
        return

    ret, prev_frame = cap.read()
    if not ret:
        messagebox.showerror("Error", "Could not read frame.")
        cap.release()
        return
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    threshold_value = 20  
    blur_kernel_size = (3, 3)
    prev_frames = []
    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_blurred = cv2.GaussianBlur(gray, blur_kernel_size, 0)
            prev_gray_blurred = cv2.GaussianBlur(prev_gray, blur_kernel_size, 0)
            diff_frame = cv2.absdiff(gray_blurred, prev_gray_blurred)

            if sensitivity_type == "High Sensitivity":
                sensitive_frame = enhance_sensitivity_high(diff_frame)
            else:
                sensitive_frame = enhance_sensitivity(diff_frame)

            _, thresholded_frame = cv2.threshold(sensitive_frame, threshold_value, 255, cv2.THRESH_BINARY)
            smoothed_frame, prev_frames = temporal_smoothing(thresholded_frame, prev_frames)
            color_mapped_frame = cv2.applyColorMap(sensitive_frame, cv2.COLORMAP_MAGMA)

            filtered_frame = apply_filter(frame, filter_type)
            resized_filtered_frame = resize_frame(filtered_frame, target_width, target_height)
            resized_sensitive_frame = resize_frame(sensitive_frame, target_width, target_height)
            resized_smoothed_frame = resize_frame(smoothed_frame, target_width, target_height)
            resized_color_mapped_frame = resize_frame(color_mapped_frame, target_width, target_height)

            if frame_type == "Difference":
                display_frame = resize_frame(sensitive_frame, int(target_width * 0.2), int(target_height * 0.2))
            elif frame_type == "Threshold":
                display_frame = resize_frame(smoothed_frame, int(target_width * 0.2), int(target_height * 0.2))
            elif frame_type == "Colormap":
                display_frame = resize_frame(color_mapped_frame, int(target_width * 0.2), int(target_height * 0.2))
            elif frame_type == "Filtered":
                display_frame = resize_frame(filtered_frame, int(target_width * 0.2), int(target_height * 0.2))

            if recording:
                timer = (datetime.now() - start_time).seconds
                cv2.putText(display_frame, f"Recording: {timer}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                out.write(frame)

            cv2.imshow('Output Frame', display_frame)

            prev_gray = gray

        key = cv2.waitKey(playback_delay) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord('r'):
            if not recording:
                recording = True
                start_time = datetime.now()
                output_path = os.path.join(os.path.expanduser("~"), "Downloads", "live_feed_recording.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                out = cv2.VideoWriter(output_path, fourcc, fps, (display_frame.shape[1], display_frame.shape[0]))
                if not out.isOpened():
                    print("Error: Could not open video writer.")
                    recording = False
                else:
                    print(f"Recording started: {output_path}")
            else:
                recording = False
                out.release()
                print("Recording stopped and file saved.")
        if recording:
            save_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            out.write(save_frame)

    cap.release()
    cv2.destroyAllWindows()

def save_video(input_video_path, target_width, target_height, fps, frame_type, prev_frames, threshold_value, blur_kernel_size, filter_type):
    output_video_path_diff = 'output_difference_video.mp4'
    output_video_path_threshold = 'output_threshold_video.mp4'
    output_video_path_colormap = 'output_colormap_video.mp4'
    output_video_path_filtered = 'output_filtered_video.mp4'

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open video.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out_diff = cv2.VideoWriter(output_video_path_diff, fourcc, fps, (target_width, target_height), isColor=False)
    out_threshold = cv2.VideoWriter(output_video_path_threshold, fourcc, fps, (target_width, target_height), isColor=False)
    out_colormap = cv2.VideoWriter(output_video_path_colormap, fourcc, fps, (target_width, target_height), isColor=True)
    out_filtered = cv2.VideoWriter(output_video_path_filtered, fourcc, fps, (target_width, target_height), isColor=True)

    ret, prev_frame = cap.read()
    if not ret:
        messagebox.showerror("Error", "Could not read frame.")
        cap.release()
        out_diff.release()
        out_threshold.release()
        out_colormap.release()
        out_filtered.release()
        return
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, blur_kernel_size, 0)
        prev_gray_blurred = cv2.GaussianBlur(prev_gray, blur_kernel_size, 0)
        diff_frame = cv2.absdiff(gray_blurred, prev_gray_blurred)

        if sensitivity_type == "High Sensitivity":
            sensitive_frame = enhance_sensitivity_high(diff_frame)
        else:
            sensitive_frame = enhance_sensitivity(diff_frame)

        _, thresholded_frame = cv2.threshold(sensitive_frame, threshold_value, 255, cv2.THRESH_BINARY)
        smoothed_frame, prev_frames = temporal_smoothing(thresholded_frame, prev_frames)
        color_mapped_frame = cv2.applyColorMap(sensitive_frame, cv2.COLORMAP_MAGMA)

        filtered_frame = apply_filter(frame, filter_type)
        resized_filtered_frame = resize_frame(filtered_frame, target_width, target_height)
        resized_sensitive_frame = resize_frame(sensitive_frame, target_width, target_height)
        resized_smoothed_frame = resize_frame(smoothed_frame, target_width, target_height)
        resized_color_mapped_frame = resize_frame(color_mapped_frame, target_width, target_height)

        if frame_type == "Difference":
            out_diff.write(resized_sensitive_frame)
        elif frame_type == "Threshold":
            out_threshold.write(resized_smoothed_frame)
        elif frame_type == "Colormap":
            out_colormap.write(resized_color_mapped_frame)
        elif frame_type == "Filtered":
            out_filtered.write(resized_filtered_frame)

    cap.release()
    out_diff.release()
    out_threshold.release()
    out_colormap.release()
    out_filtered.release()

def preview_video():
    process_video(preview=True)

def process_and_save_video():
    process_video(preview=False)

def start_live_feed():
    process_video(preview=True, live=True)

root = tk.Tk()
root.title("Video Processing")

video_path = tk.StringVar()
width = tk.StringVar(value="3000")
height = tk.StringVar(value="1500")

tk.Label(root, text="Select Video:").grid(row=0, column=0, padx=5, pady=5)
video_path_entry = tk.Entry(root, textvariable=video_path, width=50)
video_path_entry.grid(row=0, column=1, padx=5, pady=5)
tk.Button(root, text="Browse", command=select_video).grid(row=0, column=2, padx=5, pady=5)

tk.Label(root, text="Width:").grid(row=1, column=0, padx=5, pady=5)
tk.Entry(root, textvariable=width).grid(row=1, column=1, padx=5, pady=5)

tk.Label(root, text="Height:").grid(row=2, column=0, padx=5, pady=5)
tk.Entry(root, textvariable=height).grid(row=2, column=1, padx=5, pady=5)

tk.Label(root, text="Target FPS:").grid(row=3, column=0, padx=5, pady=5)
fps_slider = tk.Scale(root, from_=1, to_=240, orient=tk.HORIZONTAL)
fps_slider.set(120)
fps_slider.grid(row=3, column=1, padx=5, pady=5)

tk.Label(root, text="Playback Speed:").grid(row=4, column=0, padx=5, pady=5)
playback_speed_slider = tk.Scale(root, from_=0.1, to_=3.0, orient=tk.HORIZONTAL, resolution=0.1)
playback_speed_slider.set(1.0)
playback_speed_slider.grid(row=4, column=1, padx=5, pady=5)

tk.Label(root, text="Frame Type:").grid(row=5, column=0, padx=5, pady=5)
frame_type_var = tk.StringVar(value="Difference")
frame_type_menu = ttk.Combobox(root, textvariable=frame_type_var)
frame_type_menu['values'] = ("Difference", "Threshold", "Colormap", "Filtered")
frame_type_menu.grid(row=5, column=1, padx=5, pady=5)

tk.Label(root, text="Sensitivity:").grid(row=6, column=0, padx=5, pady=5)
sensitivity_var = tk.StringVar(value="Normal Sensitivity")
sensitivity_menu = ttk.Combobox(root, textvariable=sensitivity_var)
sensitivity_menu['values'] = ("Normal Sensitivity", "High Sensitivity")
sensitivity_menu.grid(row=6, column=1, padx=5, pady=5)

tk.Label(root, text="Filter Type:").grid(row=7, column=0, padx=5, pady=5)
filter_var = tk.StringVar(value="None")
filter_menu = ttk.Combobox(root, textvariable=filter_var)
filter_menu['values'] = ("None", "Gaussian Blur", "Median Blur", "Bilateral Filter", "Histogram Equalization")
filter_menu.grid(row=7, column=1, padx=5, pady=5)

tk.Button(root, text="Preview Video", command=preview_video).grid(row=8, column=0, columnspan=3, pady=5)
tk.Button(root, text="Process and Save Video", command=process_and_save_video).grid(row=9, column=0, columnspan=3, pady=10)
tk.Button(root, text="Start Live Feed", command=start_live_feed).grid(row=10, column=0, columnspan=3, pady=10)

recording = False
out = None
start_time = None

root.mainloop()