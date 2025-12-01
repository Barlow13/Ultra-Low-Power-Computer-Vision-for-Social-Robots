"""
Brady Barlow
Oklahoma State University
11/28/2025

TensorFlow Lite YouTube Video Inference Script
This script downloads and processes a YouTube video stream using a TensorFlow Lite model.
It supports both quantized and float models, displays results with confidence bars,
and can save the output video with overlaid predictions.

Usage Example:
python -u Youtube.py "https://www.youtube.com/watch?v=gcUHp8Wm7D0" --max-frames 300

"""

import cv2
import numpy as np
import tensorflow as tf
import yt_dlp
import argparse
import os
import json
import time

# Constants
EXPORT_DIR = './export'
MODEL_NAME = 'BSB-PicoVision'
MODEL_PATH = os.path.join(EXPORT_DIR, f'{MODEL_NAME}.tflite')
CLASSES = ['person', 'dog', 'cat', 'none']
IMG_SIZE = 96
NUM_CLASSES = len(CLASSES)

def get_youtube_stream_url(youtube_url):
    ydl_opts = {
        'format': '18/best[ext=mp4][protocol^=http]/best[protocol^=http]',
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info['url']

def main():
    parser = argparse.ArgumentParser(description='Run TFLite model on YouTube video')
    parser.add_argument('url', type=str, help='YouTube Video URL')
    parser.add_argument('--model', type=str, default=MODEL_PATH, help='Path to TFLite model')
    parser.add_argument('--threshold', type=float, default=None, help='Global confidence threshold (overrides metadata)')
    parser.add_argument('--output', type=str, default='./export/output.mp4', help='Output video file')
    parser.add_argument('--show', action='store_true', help='Show video window (requires GUI)')
    parser.add_argument('--max-frames', type=int, default=None, help='Maximum frames to process')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        return

    # Load Metadata
    meta_path = os.path.join(os.path.dirname(args.model), 'metadata.json')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        thresholds = np.array(meta.get('thresholds', [0.5]*NUM_CLASSES), dtype=np.float32)
        quant_mode = meta.get('quantization', 'float16')
        print(f"Loaded metadata: quantization={quant_mode}, thresholds={thresholds}")
    else:
        thresholds = np.array([0.5]*NUM_CLASSES, dtype=np.float32)
        quant_mode = 'float16'
        print("No metadata found, using defaults.")

    if args.threshold is not None:
        thresholds = np.array([args.threshold]*NUM_CLASSES, dtype=np.float32)
        print(f"Overriding thresholds to {args.threshold}")

    # Load Model
    interpreter = tf.lite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    in_scale = input_details[0]['quantization'][0]
    in_zero = input_details[0]['quantization'][1]
    out_scale = output_details[0]['quantization'][0]
    out_zero = output_details[0]['quantization'][1]

    print(f"Input details: {input_details[0]['dtype']}, scale={in_scale}, zero={in_zero}")

    # Open Video
    print(f"Resolving stream for {args.url}...")
    try:
        stream_url = get_youtube_stream_url(args.url)
    except Exception as e:
        print(f"Error getting stream URL: {e}")
        return

    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("Error opening video stream")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_cap = cap.get(cv2.CAP_PROP_FPS)
    if fps_cap == 0: fps_cap = 30.0

    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps_cap, (width, height))
        print(f"Saving output to {args.output}")

    print("Starting processing...")
    if args.show:
        print("Press 'q' to quit.")
    
    total_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        total_frames += 1
        if args.max_frames and total_frames > args.max_frames:
            print("Max frames reached.")
            break

        # Preprocess
        img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_f = img.astype(np.float32) / 255.0

        if quant_mode == 'int8' and in_scale > 0:
            input_data = np.round(img_f / in_scale + in_zero).astype(np.int8)
            input_data = np.expand_dims(input_data, axis=0)
        else:
            input_data = np.expand_dims(img_f, axis=0)

        # Inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]

        # Dequantize output if needed
        if quant_mode == 'int8' and out_scale > 0:
            probs = (output_data.astype(np.float32) - out_zero) * out_scale
        else:
            probs = output_data

        # Find top prediction
        top_idx = np.argmax(probs)
        top_label = CLASSES[top_idx]
        
        # Draw results - Bottom Left with Background
        margin = 20
        line_height = 25
        panel_width = 220
        header_height = 30
        panel_height = (len(CLASSES) * line_height) + header_height + 15
        
        # Panel position (Bottom Left)
        x_start = margin
        y_start = height - panel_height - margin
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_start, y_start), (x_start + panel_width, y_start + panel_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # Draw Top Prediction
        cv2.putText(frame, f"Best: {top_label}", (x_start + 10, y_start + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw text and bars
        text_y = y_start + header_height + 25
        
        for i, score in enumerate(probs):
            label = CLASSES[i]
            thresh = thresholds[i]
            
            # Determine color based on threshold
            color = (0, 255, 0) if score > thresh else (100, 100, 255) # Green or Light Red
            
            # Label text
            label_text = f"{label}"
            cv2.putText(frame, label_text, (x_start + 10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Score text
            score_text = f"{score:.2f}"
            cv2.putText(frame, score_text, (x_start + 80, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Bar
            bar_x = x_start + 130
            bar_w = 80
            bar_h = 8
            bar_y = text_y - 8
            
            # Background bar
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
            # Filled bar
            fill_w = int(score * bar_w)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)
            # Threshold marker
            thresh_x = bar_x + int(thresh * bar_w)
            cv2.line(frame, (thresh_x, bar_y - 2), (thresh_x, bar_y + bar_h + 2), (255, 255, 0), 1)

            text_y += line_height

        if out:
            out.write(frame)

        if args.show:
            cv2.imshow('TFLite Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            if total_frames % 30 == 0:
                print(f"Processed {total_frames} frames", end='\r')

    cap.release()
    if out:
        out.release()
    if args.show:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
