import os
if os.name == "nt":
    # Comment this out after permanently fixing your environment
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from deep_sort_realtime.deepsort_tracker import DeepSort

st.set_page_config(layout="wide")
st.subheader("Video Analytics")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    model_choice = st.selectbox('Choose YOLO Model', ['YOLOv4', 'YOLOv4-tiny'])
    uploaded_file = st.file_uploader('Upload a video file (.mp4, .avi, .mov)', type=['mp4', 'avi', 'mov'])
    analytics_option = st.radio(
        "Select Analytics Option:",
        ["Heatmap", "Dwell analysis", "Person detection"]
    )

    st.header("Model params")
    conf_thresh = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)
    nms_thresh = st.slider("NMS threshold", 0.1, 1.0, 0.45, 0.05)
    use_cuda = st.checkbox("Use CUDA (if available)", value=False)

    st.header("Zone params (right-side)")
    ZONE_OFFSET_FROM_RIGHT = st.number_input("Offset from right (px)", min_value=0, max_value=2000, value=100, step=10)
    ZONE_WIDTH = st.number_input("Zone width (px)", min_value=50, max_value=2000, value=300, step=10)
    ZONE_HEIGHT = st.number_input("Zone height (px)", min_value=50, max_value=2000, value=400, step=10)

# Model selection
if model_choice == 'YOLOv4':
    weights_path = 'models/yolov4.weights'
    config_path = 'models/yolov4.cfg'
    input_size = (416, 416)
else:
    weights_path = 'models/yolov4-tiny.weights'
    config_path = 'models/yolov4-tiny.cfg'
    input_size = (320, 320)

@st.cache_resource
def load_network(weights, config, use_cuda=False):
    # Darknet loader with backend selection
    net = cv2.dnn.readNet(weights, config)
    try:
        if use_cuda:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        else:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    except Exception:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Output layers as in original code
    layer_names = net.getLayerNames()
    out_layer_idxs = net.getUnconnectedOutLayers().flatten()
    output_layers = [layer_names[i - 1] for i in out_layer_idxs]
    return net, output_layers

# Dwell state and tracker
dwell_times_ms = dict()   # store milliseconds now
tracker = DeepSort(max_age=30)

def in_zone(bbox, zone):
    x1, y1, x2, y2 = zone
    cx = int((bbox[0] + bbox[2]) / 2)
    cy = int((bbox[1] + bbox[3]) / 2)
    return x1 <= cx <= x2 and y1 <= cy <= y2

def process_frame(frame, detections_xyxy_conf, zone, dt_ms):
    # detections_xyxy_conf: list of [x1,y1,x2,y2,conf]
    global dwell_times_ms, tracker

    # DeepSort expects [[bbox, conf], ...] where bbox is [x1,y1,x2,y2]
    detections_list = []
    for det in detections_xyxy_conf:
        bbox = det[:4].tolist() if isinstance(det, np.ndarray) else list(det[:4])
        conf = float(det[4])
        detections_list.append([bbox, conf])

    results = tracker.update_tracks(detections_list, frame=frame)

    # draw zone
    cv2.rectangle(frame, (zone[0], zone[1]), (zone[2], zone[3]), (0, 0, 255), 2)

    # annotate tracks, update dwell
    for track in results:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, r, b = track.to_ltrb()
        if in_zone((l, t, r, b), zone) and dt_ms > 0:
            dwell_times_ms[track_id] = dwell_times_ms.get(track_id, 0.0) + dt_ms
        in_zone_now = in_zone((l, t, r, b), zone)
        color = (0, 200, 0) if in_zone_now else (255, 0, 0)
        cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), color, 2)
        dwell_sec = dwell_times_ms.get(track_id, 0.0) / 1000.0
        cv2.putText(frame, f'ID {track_id} | {dwell_sec:.1f}s',
                    (int(l), max(15, int(t) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return frame

def yolo_person_detections(frame, net, output_layers, input_size, conf_thresh, nms_thresh):
    # Forward pass
    H, W = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, input_size, swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Collect boxes/confidences/class_ids
    boxes_xywh = []
    confidences = []
    class_ids = []
    for out in outs:
        for det in out:
            scores = det[5:]
            cid = int(np.argmax(scores))
            conf = float(scores[cid])
            if conf < conf_thresh:
                continue
            # filter to COCO person class id 0
            if cid != 0:
                continue
            cx, cy, w, h = det[0:4]
            x = int((cx - w / 2) * W)
            y = int((cy - h / 2) * H)
            w_px = int(w * W)
            h_px = int(h * H)
            boxes_xywh.append([x, y, w_px, h_px])
            confidences.append(conf)
            class_ids.append(cid)

    # Apply NMS
    detected_xyxy_conf = []
    if len(boxes_xywh) > 0:
        indices = cv2.dnn.NMSBoxes(boxes_xywh, confidences, conf_thresh, nms_thresh)
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w_px, h_px = boxes_xywh[i]
                x1, y1 = x, y
                x2, y2 = x + w_px, y + h_px
                # clip
                x1 = max(0, min(x1, W - 1))
                y1 = max(0, min(y1, H - 1))
                x2 = max(0, min(x2, W - 1))
                y2 = max(0, min(y2, H - 1))
                detected_xyxy_conf.append([x1, y1, x2, y2, float(confidences[i])])

    return detected_xyxy_conf

if analytics_option == "Heatmap":
    st.info("This page is under maintenance")

elif analytics_option == "Dwell analysis":
    if uploaded_file is None:
        st.warning("Please upload a video file from the sidebar to start dwell time analysis.")
    else:
        temp_path = 'temp_video.mp4'
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.read())

        net, output_layers = load_network(weights_path, config_path, use_cuda=use_cuda)
        cap = cv2.VideoCapture(temp_path)
        frame_placeholder = st.empty()
        scale_factor = 0.7

        # Geometry
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Right-side zone centered vertically
        zone_left = max(0, video_width - ZONE_OFFSET_FROM_RIGHT - ZONE_WIDTH)
        zone_top = max(0, (video_height - ZONE_HEIGHT) // 2)
        zone_right = min(video_width - 1, zone_left + ZONE_WIDTH)
        zone_bottom = min(video_height - 1, zone_top + ZONE_HEIGHT)
        ZONE = (zone_left, zone_top, zone_right, zone_bottom)

        # FPS and timestamp handling
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0
        last_pos_ms = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Timestamp in ms; fallback if unavailable
            pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if (pos_ms is None) or (pos_ms <= 0):
                if last_pos_ms is None:
                    pos_ms = 0.0
                else:
                    pos_ms = last_pos_ms + (1000.0 / fps)
            dt_ms = 0.0
            if last_pos_ms is not None:
                dt_ms = max(0.0, pos_ms - last_pos_ms)
                if dt_ms == 0.0:
                    dt_ms = (1000.0 / fps)
            last_pos_ms = pos_ms

            # YOLO person detections with NMS
            detected_boxes = yolo_person_detections(
                frame, net, output_layers, input_size, conf_thresh, nms_thresh
            )

            # Process with DeepSort tracker and dwell logic
            processed_frame = process_frame(frame, detected_boxes, ZONE, dt_ms)

            # Display
            display_frame = cv2.resize(processed_frame, None, fx=scale_factor, fy=scale_factor)
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(display_frame, channels='RGB')

        cap.release()

        # After video processing: save dwell times and show download button
        rows = []
        for tid, ms in dwell_times_ms.items():
            rows.append((tid, ms / 1000.0))
        df_dwell = pd.DataFrame(rows, columns=['Track ID', 'Dwell_Time_Secs'])

        st.write("Dwell Times (per Track):")
        st.dataframe(df_dwell)

        csv_bytes = df_dwell.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download dwell time CSV",
            data=csv_bytes,
            file_name='dwell_time_output.csv',
            mime='text/csv'
        )

else:
    st.subheader("Person detection using YOLO")
    if uploaded_file is None:
        st.warning("Please upload a video file from the sidebar to start detection.")
    else:
        temp_path = 'temp_video.mp4'
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.read())

        net, output_layers = load_network(weights_path, config_path, use_cuda=use_cuda)
        cap = cv2.VideoCapture(temp_path)
        frame_placeholder = st.empty()
        scale_factor = 0.7

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO person detections with NMS
            detections = yolo_person_detections(
                frame, net, output_layers, input_size, conf_thresh, nms_thresh
            )

            # Draw boxes
            for x1, y1, x2, y2, conf in detections:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{conf:.2f}", (int(x1), max(15, int(y1) - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            display_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(display_frame, channels='RGB')

        cap.release()
