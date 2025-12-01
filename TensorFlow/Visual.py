"""
Brady Barlow
Oklahoma State University
11/28/2025

TensorFlow Lite Visual Inference Script
This script processes images using a TensorFlow Lite model.
It can create a grid visualization of random validation images with predictions,
and evaluate the model on the full validation set, saving metrics and confusion plots.

Usage Example:
python -u Visual.py --grid 48 --rows 6 --cols 8 --csv --confusion --seed 42

"""

import os, json, random, argparse
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except:
    _HAS_TQDM = False

EXPORT_DIR = './export'
DATASET_ROOT = '../datasets/coco'
MODEL_NAME = 'BSB-PicoVision'  
TFLITE_PATH = os.path.join(EXPORT_DIR, f'{MODEL_NAME}.tflite')
VAL_CSV = os.path.join(DATASET_ROOT, 'balanced_multilabel_val.csv')
CLASSES = ['person', 'dog', 'cat', 'none']
IMG_SIZE = 96
NUM_CLASSES = len(CLASSES)

# Load metadata thresholds (fallback to 0.5)
meta_path = os.path.join(EXPORT_DIR, 'metadata.json')
if os.path.exists(meta_path):
    with open(meta_path) as f:
        meta = json.load(f)
    THRESHOLDS = np.array(meta.get('thresholds', [0.5]*len(CLASSES)), dtype=np.float32)
    QUANT_MODE = meta.get('quantization', 'float16')
else:
    THRESHOLDS = np.array([0.5]*len(CLASSES), dtype=np.float32)
    QUANT_MODE = 'float16'

df_val = pd.read_csv(VAL_CSV)

def fix_path(p):
    p = str(p).replace('\\', '/')
    if 'images/' in p:
        p = p[p.index('images/'):]
    return p

df_val['image'] = df_val['image'].apply(fix_path)

interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
in_scale = input_details[0].get('quantization', (0.0, 0))[0]
in_zero = input_details[0].get('quantization', (0.0, 0))[1]
out_scale = output_details[0].get('quantization', (0.0, 0))[0]
out_zero = output_details[0].get('quantization', (0.0, 0))[1]

def preprocess(img_bgr):
    img = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_f = img.astype(np.float32) / 255.0
    if QUANT_MODE == 'int8' and in_scale > 0:
        img_q = np.round(img_f / in_scale + in_zero).astype(np.int8)
        return img_q
    return img_f

def infer(img_proc):
    tensor = np.expand_dims(img_proc, 0)
    interpreter.set_tensor(input_details[0]['index'], tensor)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index'])[0]
    if QUANT_MODE == 'int8' and out_scale > 0:
        out = (out.astype(np.float32) - out_zero) * out_scale
    return out

def visualize(num_images=24, rows=4, cols=6, seed=None, show_bars=False):
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(df_val), size=min(num_images, len(df_val)), replace=False)
    subset = df_val.iloc[indices]
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    axes = axes.flatten()
    for i, (_, row) in enumerate(subset.iterrows()):
        if i >= rows*cols:
            break
        path = os.path.join(DATASET_ROOT, row['image'])
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            axes[i].axis('off'); continue
        img_proc = preprocess(img_bgr)
        probs = infer(img_proc)
        pred_labels = [CLASSES[j] for j, p in enumerate(probs) if p >= THRESHOLDS[j]]
        true_labels = [c for c in CLASSES if row[c] == 1]
        show_img = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), (IMG_SIZE, IMG_SIZE))
        if show_bars:
            bar_h = 14
            canvas = np.ones((IMG_SIZE + bar_h, IMG_SIZE, 3), dtype=np.uint8) * 255
            canvas[:IMG_SIZE] = show_img
            for ci, p in enumerate(probs):
                x0 = int(ci * IMG_SIZE / NUM_CLASSES)
                x1 = int((ci+1) * IMG_SIZE / NUM_CLASSES)
                h = int(bar_h * max(0.0, min(1.0, p)))
                color = (0, 180, 0) if p >= THRESHOLDS[ci] else (200, 0, 0)
                cv2.rectangle(canvas, (x0+1, bar_h - h), (x1-2, bar_h-1), color, -1)
            show_img = canvas
        axes[i].imshow(show_img)
        axes[i].axis('off')
        axes[i].set_title(
            f"T:{','.join(true_labels) or '-'}\nP:{','.join(pred_labels) or '-'}\n"
            f"[{', '.join(f'{p:.2f}' for p in probs)}]",
            fontsize=7
        )
    for j in range(i+1, rows*cols):
        axes[j].axis('off')
    plt.tight_layout()
    out_path = os.path.join(EXPORT_DIR, 'tflite_sample_grid.png')
    plt.savefig(out_path, dpi=140)
    print(f"Saved visualization: {out_path}")
    plt.close()

def evaluate_full(save_csv=False, make_confusion_plot=False, limit=None):
    indices = np.arange(len(df_val))
    if limit:
        indices = indices[:limit]
    preds = []
    iterator = indices
    if _HAS_TQDM:
        iterator = tqdm(indices, desc="Eval", unit="img")
    for idx in iterator:
        row = df_val.iloc[idx]
        path = os.path.join(DATASET_ROOT, row['image'])
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            preds.append([np.nan]*NUM_CLASSES); continue
        probs = infer(preprocess(img_bgr))
        preds.append(probs)
    preds = np.array(preds, dtype=np.float32)
    y_true = df_val.iloc[indices][CLASSES].values.astype(int)
    y_bin = (preds >= THRESHOLDS).astype(int)

    # Per-class metrics
    stats = []
    for ci, cname in enumerate(CLASSES):
        yt = y_true[:, ci]
        yp = y_bin[:, ci]
        TP = int(((yt==1)&(yp==1)).sum())
        FP = int(((yt==0)&(yp==1)).sum())
        TN = int(((yt==0)&(yp==0)).sum())
        FN = int(((yt==1)&(yp==0)).sum())
        prec = TP / (TP + FP + 1e-9)
        rec  = TP / (TP + FN + 1e-9)
        f1   = 2*prec*rec/(prec+rec+1e-9)
        stats.append({
            'class': cname,
            'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN,
            'precision': prec, 'recall': rec, 'f1': f1,
            'threshold': float(THRESHOLDS[ci])
        })
    stats_df = pd.DataFrame(stats)
    print(stats_df)

    if save_csv:
        stats_df.to_csv(os.path.join(EXPORT_DIR, 'val_class_metrics.csv'), index=False)
        out_pred = df_val.iloc[indices][['image']].copy()
        for ci, cname in enumerate(CLASSES):
            out_pred[f'prob_{cname}'] = preds[:, ci]
        out_pred.to_csv(os.path.join(EXPORT_DIR, 'val_predictions.csv'), index=False)
        print("Saved val_class_metrics.csv & val_predictions.csv")

    if make_confusion_plot:
        plt.figure(figsize=(8,3))
        x = np.arange(NUM_CLASSES)
        plt.bar(x-0.25, stats_df['precision'], 0.25, label='Precision')
        plt.bar(x,       stats_df['recall'],    0.25, label='Recall')
        plt.bar(x+0.25,  stats_df['f1'],        0.25, label='F1')
        plt.xticks(x, CLASSES, rotation=20)
        plt.ylim(0,1)
        plt.grid(axis='y', alpha=0.3)
        plt.title('Per-class metrics')
        plt.legend(fontsize=8)
        outp = os.path.join(EXPORT_DIR, 'per_class_metrics.png')
        plt.tight_layout()
        plt.savefig(outp, dpi=140)
        plt.close()
        print(f"Saved per-class metrics plot: {outp}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid', type=int, default=24, help='Number of images in grid')
    parser.add_argument('--rows', type=int, default=4)
    parser.add_argument('--cols', type=int, default=6)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--bars', action='store_true', help='Show class probability bar strips')
    parser.add_argument('--csv', action='store_true', help='Save prediction CSV & metrics')
    parser.add_argument('--confusion', action='store_true', help='Generate per-class metrics plot')
    parser.add_argument('--limit', type=int, default=None, help='Limit evaluation images (speed)')
    args = parser.parse_args()

    visualize(num_images=args.grid, rows=args.rows, cols=args.cols,
              seed=args.seed, show_bars=args.bars)
    if args.csv or args.confusion:
        evaluate_full(save_csv=args.csv, make_confusion_plot=args.confusion, limit=args.limit)