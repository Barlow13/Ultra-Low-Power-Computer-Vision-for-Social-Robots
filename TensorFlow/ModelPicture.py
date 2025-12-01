"""
Brady Barlow
Oklahoma State University
12/01/2025

TensorFlow Model Visualization Script
This script generates visualizations of the model architecture and an example input/output inference.
It exports the model architecture diagram (PNG/SVG) and an inference example figure to the export directory.

Usage Example:
python -u ModelPicture.py

"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import plot_model
from PIL import Image
import matplotlib.pyplot as plt

# ─── CONFIG ─────────────────────────────────────────────────────────
CONFIG = {
    'EXPORT_DIR': 'export',
    'MODEL_FILENAME': 'best.keras',
    'TFLITE_FILENAME': 'BSB-PicoVision.tflite',
    'METADATA_FILENAME': 'metadata.json',
    'EXAMPLE_IMAGE_PATH': '../datasets/coco/images/train2017/000000380252.jpg',
    'ARCH_PNG_FILENAME': 'model_architecture.png',
    'ARCH_SVG_FILENAME': 'model_architecture.svg',
    'IO_FIGURE_FILENAME': 'model_io_example.png',
    'DPI': None,
    'DEFAULT_CLASSES': ["class0", "class1", "class2"]
}

# ─── SETUP PATHS ────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
export_dir = os.path.join(script_dir, CONFIG['EXPORT_DIR'])

# Input paths
model_path = os.path.join(export_dir, CONFIG['MODEL_FILENAME'])
tflite_model_path = os.path.join(export_dir, CONFIG['TFLITE_FILENAME'])
metadata_path = os.path.join(export_dir, CONFIG['METADATA_FILENAME'])
example_image_path = os.path.join(script_dir, CONFIG['EXAMPLE_IMAGE_PATH'])

# Output paths
arch_png_path = os.path.join(export_dir, CONFIG['ARCH_PNG_FILENAME'])
arch_svg_path = os.path.join(export_dir, CONFIG['ARCH_SVG_FILENAME'])
io_figure_path = os.path.join(export_dir, CONFIG['IO_FIGURE_FILENAME'])

# Ensure export directory exists
os.makedirs(export_dir, exist_ok=True)

# ─── LOAD METADATA ──────────────────────────────────────────────────
CLASS_NAMES = CONFIG['DEFAULT_CLASSES']
THRESHOLDS = None

if os.path.exists(metadata_path):
    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
            if "classes" in metadata:
                CLASS_NAMES = metadata["classes"]
                print(f"Loaded class names from metadata: {CLASS_NAMES}")
            if "thresholds" in metadata:
                THRESHOLDS = metadata["thresholds"]
                print(f"Loaded thresholds from metadata: {THRESHOLDS}")
    except Exception as e:
        print(f"Could not load metadata: {e}")

def get_size_str(path):
    """Returns a human-readable file size string."""
    if not os.path.exists(path): return "N/A"
    sz = os.path.getsize(path)
    if sz < 1024*1024: return f"{sz/1024:.0f} KB"
    return f"{sz/(1024*1024):.1f} MB"

def main():
    # ─── 1. LOAD MODEL & SAVE ARCHITECTURE ──────────────────────────
    print(f"Loading model from {model_path}...")
    model = keras.models.load_model(model_path)

    print("Generating model architecture diagrams...")
    # PNG version
    plot_model(
        model,
        to_file=arch_png_path,
        show_shapes=True,
        show_layer_names=True,
        expand_nested=True,
        dpi=CONFIG['DPI']
    )
    # SVG version (vector graphic, Visio-friendly)
    plot_model(
        model,
        to_file=arch_svg_path,
        show_shapes=True,
        show_layer_names=True,
        expand_nested=True,
        dpi=CONFIG['DPI']
    )

    print(f"Model diagram saved to:\n  {arch_png_path}\n  {arch_svg_path}")

    # ─── 2. EXAMPLE INPUT + PREDICTED OUTPUT FIGURE ─────────────────
    if os.path.exists(example_image_path):
        print(f"Processing example image: {example_image_path}")
        # get model input size (H, W)
        _, h, w, _ = model.input_shape

        img = Image.open(example_image_path).convert("RGB")
        img_resized = img.resize((w, h))

        x = np.array(img_resized, dtype=np.float32) / 255.0
        x = np.expand_dims(x, axis=0)

        # --- Keras Inference ---
        preds_keras = model.predict(x, verbose=0)[0]

        # --- TFLite Inference ---
        tflite_preds = None
        if os.path.exists(tflite_model_path):
            try:
                interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
                interpreter.allocate_tensors()

                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()

                # Check if input needs quantization
                input_scale, input_zero_point = input_details[0]['quantization']
                if input_scale != 0.0:
                    # Quantize input: (val / scale) + zero_point
                    x_input = (x / input_scale + input_zero_point).astype(np.int8)
                else:
                    x_input = x

                interpreter.set_tensor(input_details[0]['index'], x_input)
                interpreter.invoke()
                
                output_data = interpreter.get_tensor(output_details[0]['index'])[0]

                # Dequantize output if needed
                output_scale, output_zero_point = output_details[0]['quantization']
                if output_scale != 0.0:
                    tflite_preds = (output_data.astype(np.float32) - output_zero_point) * output_scale
                else:
                    tflite_preds = output_data
                    
            except Exception as e:
                print(f"TFLite inference failed: {e}")

        # labels for the bars
        if len(CLASS_NAMES) == preds_keras.shape[-1]:
            labels = CLASS_NAMES
        else:
            labels = [f"c{i}" for i in range(preds_keras.shape[-1])]

        keras_size = get_size_str(model_path)
        tflite_size = get_size_str(tflite_model_path)

        # Plotting
        num_plots = 4 if tflite_preds is not None else 3
        plt.figure(figsize=(5 * num_plots, 5))

        # 1. Original image
        plt.subplot(1, num_plots, 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Original ({img.width}x{img.height})")

        # 2. Model input (resized)
        plt.subplot(1, num_plots, 2)
        plt.imshow(img_resized)
        plt.axis("off")
        plt.title(f"Model Input ({w}x{h})")

        # 3. Keras Predictions
        plt.subplot(1, num_plots, 3)
        y_pos = np.arange(len(preds_keras))
        plt.barh(y_pos, preds_keras, color='orange')
        plt.yticks(y_pos, labels)
        plt.xlabel("Probability")
        plt.title(f"Keras Model Output ({keras_size})")
        plt.xlim(0, 1.0)
        
        # Add thresholds
        if THRESHOLDS and len(THRESHOLDS) == len(preds_keras):
             for i, thresh in enumerate(THRESHOLDS):
                plt.plot([thresh, thresh], [i - 0.4, i + 0.4], "k-", linewidth=2, label="Threshold" if i == 0 else "")
        
        plt.gca().invert_yaxis()

        # 4. TFLite Predictions (if available)
        if tflite_preds is not None:
            plt.subplot(1, num_plots, 4)
            plt.barh(y_pos, tflite_preds, color='orange')
            plt.yticks(y_pos, labels)
            plt.xlabel("Probability")
            plt.title(f"TFLite Model Output ({tflite_size})")
            plt.xlim(0, 1.0)
            
            # Add thresholds
            if THRESHOLDS and len(THRESHOLDS) == len(tflite_preds):
                 for i, thresh in enumerate(THRESHOLDS):
                    plt.plot([thresh, thresh], [i - 0.4, i + 0.4], "k-", linewidth=2)

            plt.gca().invert_yaxis()

        plt.tight_layout()

        plt.savefig(io_figure_path, dpi=CONFIG['DPI'])
        plt.close()

        print(f"Example input/output figure saved to:\n  {io_figure_path}")
    else:
        print(f"Skipping IO example: {example_image_path} not found.")

if __name__ == "__main__":
    main()
