"""
Brady Barlow
Oklahoma State University
11/28/2025

TensorFlow Multi-Label Classification Training Script
This script trains a multi-label image classification model using TensorFlow.
It includes dynamic threshold adjustment, visualization callbacks, and exports the trained model.

Usage Example:
python -u Train.py

"""

import os, random, gc, json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from datetime import datetime
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── CONFIG ─────────────────────────────────────────────────────────
CONFIG = {
    'BATCH_SIZE': 32, # Batch size for training and validation
    'EPOCHS_FROZEN': 32, # Number of epochs with frozen backbone
    'EPOCHS_FINETUNE': 224, # Number of epochs for fine-tuning
    'INITIAL_LR': 1e-3, # Initial learning rate
    'FINETUNE_LR': 1e-5, # Learning rate for fine-tuning
    'IMG_SIZE': 96, # Input image size (height and width)
    'ALPHA': 0.35, # MobileNetV2 width multiplier
    'BACKBONE_CUTOFF': 'block_6_expand', # MobileNetV2 layer to cut at
    'DROPOUT': 0.5, # Dropout rate before final dense layers
    'SEED': 42, # Random seed for reproducibility
    'DATASET_ROOT': '../datasets/coco', # Root directory of the dataset
    'EXPORT_DIR': './export', # Directory to save exports
    'TENSORBOARD_DIR': './logs', # Directory for TensorBoard logs
    'CHECKPOINT': './export/best.keras', # Path to save the best model checkpoint
    'NICKNAME': 'BSB-PicoVision', # Model nickname
    'THRESH_FREQ': 4, # Frequency (in epochs) to update dynamic thresholds
    'THRESH_GRID': np.linspace(0.05, 0.95, 19).tolist(), # Threshold search grid
    'THRESH_EMA_ALPHA': 0.5,   # 0 → use raw thresholds; closer to 1.0 → smoother/slower updates
    'THRESH_MINMAX': (0.05, 0.9),  # clip thresholds to this range per class
    'VIS_FREQ': 4, # Frequency (in epochs) to generate visualization grids
    'VIS_SAMPLES': 16, # Number of samples to visualize
    'VIS_ROWS': 4, # Number of rows in visualization grid
    'VIS_COLS': 4, # Number of columns in visualization grid
    'QUANT_MODE': 'int8',  # 'float16', 'int8', or None
    'REP_SAMPLES': 300, # Number of samples for representative dataset (int8 quantization)
}

CLASSES = ['person', 'dog', 'cat', 'none']
NUM_CLASSES = len(CLASSES)
os.makedirs(CONFIG['EXPORT_DIR'], exist_ok=True)
random.seed(CONFIG['SEED']); np.random.seed(CONFIG['SEED']); tf.random.set_seed(CONFIG['SEED'])

# ─── DATA ───────────────────────────────────────────────────────────
def load_csv_dataset():
    train_csv = os.path.join(CONFIG['DATASET_ROOT'], 'balanced_multilabel_train.csv')
    val_csv = os.path.join(CONFIG['DATASET_ROOT'], 'balanced_multilabel_val.csv')
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
   # Fix paths in case of Windows backslashes
    def fix_path(p):
        p = str(p).replace('\\', '/')
        if 'images/' in p:
            p = p[p.index('images/'):]
        return p

    df_train['image'] = df_train['image'].apply(fix_path)
    df_val['image'] = df_val['image'].apply(fix_path)

    print(f"Train samples: {len(df_train)}, Val samples: {len(df_val)}")
    return df_train, df_val

def _parse(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, (CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE']))
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

def _augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.2)
    img = tf.image.random_contrast(img, 0.85, 1.15)
    img = tf.image.random_saturation(img, 0.85, 1.15)
    img = tf.image.random_hue(img, 0.2)
    return img, label

def create_dataset(df, training=False):
    paths = [os.path.join(CONFIG['DATASET_ROOT'], p) for p in df['image']]
    labels = df[CLASSES].values.astype(np.float32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(2048).map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(CONFIG['BATCH_SIZE']).prefetch(tf.data.AUTOTUNE)
    return ds

# ─── METRICS FACTORY ────────────────────────────────────────────────
def get_metrics():
    return [
        tf.keras.metrics.BinaryAccuracy(name='bin_acc'),
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]

# ─── MODEL ──────────────────────────────────────────────────────────
def build_streamlined_model(cut_layer_name):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'], 3),
        include_top=False,
        alpha=CONFIG['ALPHA'],
        weights='imagenet'
    )
    try:
        cut_output = base.get_layer(cut_layer_name).output
    except ValueError:
        raise ValueError(f"Cut layer {cut_layer_name} not found.")
    truncated = models.Model(base.input, cut_output, name='truncated_mobilenetv2')
    truncated.trainable = False
    
    # Build full model architecture
    inputs = layers.Input((CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'], 3))
    x = truncated(inputs)
    x = layers.Conv2D(192, 1, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.DepthwiseConv2D(3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 1, padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(CONFIG['DROPOUT'])(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(NUM_CLASSES, activation='sigmoid')(x)

    model = models.Model(inputs, outputs, name=CONFIG['NICKNAME'])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(CONFIG['INITIAL_LR']),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.03),
        metrics=get_metrics()
    )
    model.best_thresholds = np.array([0.5]*NUM_CLASSES, dtype=np.float32)
    model.best_macro_f1 = -1.0
    model.best_confusion = {}
    return model, truncated

def unfreeze_for_finetune(model, truncated, train_last_n_blocks=3):
    truncated.trainable = True
    block_layers = [l for l in truncated.layers if 'block_' in l.name and 'expand' in l.name]
    keep_frozen = block_layers[:-train_last_n_blocks]
    for l in keep_frozen:
        l.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(CONFIG['FINETUNE_LR']),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.0),
        metrics=get_metrics()
    )

# ─── DYNAMIC THRESHOLD CALLBACK ─────────────────────────────────────
class DynamicThresholdCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_ds, freq=1, grid=None,
                 ema_alpha=None, minmax=None):
        super().__init__()
        self.val_ds = val_ds
        self.freq = freq
        self.grid = grid if grid is not None else np.linspace(0.05,0.95,19)
        self.history = []  # store dict entries
        self.ema_alpha = ema_alpha
        self.minmax = minmax

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.freq != 0:
            return
        # collect predictions
        y_true, y_pred = [], []
        for x, y in self.val_ds:
            y_true.append(y.numpy())
            y_pred.append(self.model(x, training=False).numpy())
        y_true = np.vstack(y_true); y_pred = np.vstack(y_pred)

        from sklearn.metrics import f1_score
        thresholds = []
        for c in range(NUM_CLASSES):
            best_f1 = -1; best_t = 0.5
            for t in self.grid:
                f1 = f1_score(y_true[:, c], (y_pred[:, c] >= t).astype(int), zero_division=0)
                if f1 > best_f1:
                    best_f1, best_t = f1, t
            thresholds.append(best_t)
        thresholds = np.array(thresholds, dtype=np.float32)

        # Smooth and clip thresholds to prevent wild swings between epochs
        if self.ema_alpha is None:
            self.ema_alpha = CONFIG.get('THRESH_EMA_ALPHA', 0.0)
        if self.minmax is None:
            self.minmax = CONFIG.get('THRESH_MINMAX', None)

        prev = getattr(self.model, 'best_thresholds', None)
        if prev is not None and self.ema_alpha and self.ema_alpha > 0.0:
            thresholds = (self.ema_alpha * prev) + ((1.0 - self.ema_alpha) * thresholds)
        if self.minmax is not None:
            lo, hi = self.minmax
            thresholds = np.clip(thresholds, lo, hi)
        y_bin = (y_pred >= thresholds).astype(int)

        macro_f1 = f1_score(y_true, y_bin, average='macro')
        micro_f1 = f1_score(y_true, y_bin, average='micro')

        # confusion stats per class
        confusion = {}
        for idx, cls in enumerate(CLASSES):
            yt = y_true[:, idx].astype(int)
            yp = y_bin[:, idx].astype(int)
            TP = int(np.sum((yt == 1) & (yp == 1)))
            TN = int(np.sum((yt == 0) & (yp == 0)))
            FP = int(np.sum((yt == 0) & (yp == 1)))
            FN = int(np.sum((yt == 1) & (yp == 0)))
            confusion[cls] = {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN}

        improved = macro_f1 > self.model.best_macro_f1
        if improved:
            self.model.best_macro_f1 = macro_f1
            self.model.best_thresholds = thresholds
            self.model.best_confusion = confusion

        print(f"\n[DynamicThreshold] Epoch {epoch+1} macro_f1={macro_f1:.4f} micro_f1={micro_f1:.4f} "
              f"{'**improved**' if improved else ''}")
        print("[DynamicThreshold] thresholds:", {c: np.float32(round(float(t),3)) for c,t in zip(CLASSES, thresholds)})
        if improved:
            print("[DynamicThreshold] Updated best thresholds & confusion stats.")
        self.history.append({
            'epoch': epoch+1,
            'macro_f1': macro_f1,
            'micro_f1': micro_f1,
            'thresholds': thresholds.tolist(),
            'improved': improved,
            'confusion': confusion
        })

# ─── VISUALIZATION CALLBACK ────────────────────────────────────────
class VisualizationCallback(tf.keras.callbacks.Callback):
    def __init__(self, df_val, export_dir, freq=4):
        super().__init__()
        self.df_val = df_val
        self.export_dir = export_dir
        self.freq = freq
        self.sample_cache = None

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.freq != 0:
            return
        samples = self.df_val.sample(
            min(CONFIG['VIS_SAMPLES'], len(self.df_val)),
            random_state=random.randint(0, 10_000)
        )
        thresholds = getattr(self.model, 'best_thresholds',
                             np.array([0.5]*NUM_CLASSES, dtype=np.float32))
        nrows, ncols = CONFIG['VIS_ROWS'], CONFIG['VIS_COLS']
        fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))
        axes = axes.flatten()
        for idx, (_, row) in enumerate(samples.iterrows()):
            if idx >= nrows * ncols:
                break
            img_path = os.path.join(CONFIG['DATASET_ROOT'], row['image'])
            
            try:
                # Use TF to read image to avoid cv2/threading issues
                img_raw = tf.io.read_file(img_path)
                img_tensor = tf.image.decode_jpeg(img_raw, 3)
                img_tensor = tf.image.resize(img_tensor, (CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE']))
                inp = tf.cast(img_tensor, tf.float32) / 255.0
                
                # Use model() directly instead of predict()
                pred = self.model(tf.expand_dims(inp, 0), training=False).numpy()[0]
                
                # Convert for display
                img_vis = (inp.numpy() * 255).astype(np.uint8)
            except Exception as e:
                print(f"[Visualization] Error processing {img_path}: {e}")
                axes[idx].axis('off')
                continue

            true_labels = [CLASSES[i] for i in range(NUM_CLASSES) if row[CLASSES[i]] == 1]
            pred_labels = [CLASSES[i] for i in range(NUM_CLASSES) if pred[i] >= thresholds[i]]
            axes[idx].imshow(img_vis)
            axes[idx].axis('off')
            axes[idx].set_title(
                f"T: {','.join(true_labels) or 'None'}\nP: {','.join(pred_labels) or 'None'}\n"
                f"[{', '.join(f'{p:.2f}' for p in pred)}]",
                fontsize=7
            )
        for j in range(idx+1, nrows*ncols):
            axes[j].axis('off')
        plt.tight_layout()
        
        # Save to disk
        out_path = os.path.join(self.export_dir, f'viz_epoch_{epoch+1}.png')
        try:
            plt.savefig(out_path, dpi=140)
            print(f"[Visualization] Saved grid to {out_path}")
        except Exception as e:
            print(f"[Visualization] Save failed: {e}")
        plt.close()

# ─── TRAINING HISTORY PLOTTING ─────────────────────────────────────
def plot_training_history(history_dict, export_dir):
    metrics = ['loss', 'auc', 'bin_acc', 'precision', 'recall']
    plt.figure(figsize=(12, 10))
    rows = 3
    cols = 2
    for i, m in enumerate(metrics):
        if m not in history_dict:
            continue
        plt.subplot(rows, cols, i+1)
        plt.plot(history_dict[m], label=f'train_{m}')
        val_key = f'val_{m}'
        if val_key in history_dict:
            plt.plot(history_dict[val_key], label=val_key)
        plt.title(m)
        plt.xlabel('Epoch')
        plt.grid(alpha=0.3)
        plt.legend(fontsize=8)
    plt.tight_layout()
    out = os.path.join(export_dir, 'training_history.png')
    plt.savefig(out, dpi=150)
    plt.close()
    # CSV
    import pandas as pd
    pd.DataFrame(history_dict).to_csv(os.path.join(export_dir, 'training_history.csv'), index=False)
    print(f"[History] Saved plots and CSV to {export_dir}")

def _merge_histories(h1, h2, offset):
    merged = {}
    for k, v in h1.history.items():
        merged[k] = list(v)
    for k, v in h2.history.items():
        if k not in merged:
            merged[k] = [None]*offset
        merged[k].extend(v)
    return merged

# ─── TRAIN ──────────────────────────────────────────────────────────
def train():
    df_train, df_val = load_csv_dataset()
    train_ds = create_dataset(df_train, training=True)
    val_ds = create_dataset(df_val, training=False)

    model, truncated = build_streamlined_model(CONFIG['BACKBONE_CUTOFF'])
    model.summary()

# ─── CALLBACKS ─────────────────────────────────────────────────────
    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        CONFIG['CHECKPOINT'], monitor='val_auc', mode='max',
        save_best_only=True, verbose=1
    )
    es_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc', mode='max', patience=32,
        restore_best_weights=True, verbose=1
    )
    lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_auc', mode='max', factor=0.5, patience=4,
        min_lr=1e-10, verbose=1
    )
    dyn_cb_stage1 = DynamicThresholdCallback(
        val_ds,
        freq=CONFIG['THRESH_FREQ'],
        grid=CONFIG['THRESH_GRID'],
        ema_alpha=CONFIG.get('THRESH_EMA_ALPHA', 0.0),
        minmax=CONFIG.get('THRESH_MINMAX', None)
    )
    vis_cb = VisualizationCallback(df_val, CONFIG['EXPORT_DIR'], freq=CONFIG['VIS_FREQ'])

    print("\nStage 1 (frozen backbone)")
    hist1 = model.fit(
        train_ds,
        epochs=CONFIG['EPOCHS_FROZEN'],
        validation_data=val_ds,
        callbacks=[ckpt_cb, es_cb, lr_cb, dyn_cb_stage1, vis_cb],
        verbose=1
    )

    print("\nStage 2 (selective fine-tune)")
    unfreeze_for_finetune(model, truncated, train_last_n_blocks=4)
    dyn_cb_stage2 = DynamicThresholdCallback(
        val_ds,
        freq=CONFIG['THRESH_FREQ'],
        grid=CONFIG['THRESH_GRID'],
        ema_alpha=CONFIG.get('THRESH_EMA_ALPHA', 0.0),
        minmax=CONFIG.get('THRESH_MINMAX', None)
    )
    hist2 = model.fit(
        train_ds,
        initial_epoch=CONFIG['EPOCHS_FROZEN'],
        epochs=CONFIG['EPOCHS_FROZEN'] + CONFIG['EPOCHS_FINETUNE'],
        validation_data=val_ds,
        callbacks=[ckpt_cb, es_cb, lr_cb, dyn_cb_stage2, vis_cb],
        verbose=1
    )

    # Combine and plot histories
    merged_history = _merge_histories(hist1, hist2, CONFIG['EPOCHS_FROZEN'])
    plot_training_history(merged_history, CONFIG['EXPORT_DIR'])

    # Save best stats before clearing session
    best_thresholds = getattr(model, 'best_thresholds', None)
    best_macro_f1 = getattr(model, 'best_macro_f1', None)
    best_confusion = getattr(model, 'best_confusion', None)

    # Clear memory to prevent Segfault/OOM when loading best model
    del model, truncated
    tf.keras.backend.clear_session()
    gc.collect()

    print("Loading best model...")
    best = tf.keras.models.load_model(CONFIG['CHECKPOINT'])
    if best_thresholds is not None:
        best.best_thresholds = best_thresholds
        best.best_macro_f1 = best_macro_f1
        best.best_confusion = best_confusion
        
    evaluate_and_export(best, val_ds, df_val)
    return best

# ─── EVAL + EXPORT (UPDATED) ───────────────────────────────────────
def evaluate_and_export(model, val_ds, df_val):
    print("\nEvaluating with best thresholds (if available)...")
    
    # Use model.predict directly on the dataset for efficiency and stability
    y_pred = model.predict(val_ds, verbose=1)
    y_true = df_val[CLASSES].values.astype(np.float32)
    
    # Ensure shapes match
    if len(y_true) != len(y_pred):
        print(f"Warning: Label count {len(y_true)} != Prediction count {len(y_pred)}")
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]

    thresholds = getattr(model, 'best_thresholds', np.array([0.5]*NUM_CLASSES, dtype=np.float32))
    from sklearn.metrics import f1_score
    y_bin = (y_pred >= thresholds).astype(int)

    macro_f1 = f1_score(y_true, y_bin, average='macro')
    micro_f1 = f1_score(y_true, y_bin, average='micro')
    print(f"Final Macro F1: {macro_f1:.4f}")
    print(f"Final Micro F1: {micro_f1:.4f}")
    print("Final thresholds:", {c: round(t,3) for c,t in zip(CLASSES, thresholds)})

    # Confusion stats
    confusion = {}
    for idx, cls in enumerate(CLASSES):
        yt = y_true[:, idx].astype(int)
        yp = y_bin[:, idx].astype(int)
        TP = int(np.sum((yt == 1) & (yp == 1)))
        TN = int(np.sum((yt == 0) & (yp == 0)))
        FP = int(np.sum((yt == 0) & (yp == 1)))
        FN = int(np.sum((yt == 1) & (yp == 0)))
        confusion[cls] = {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN}

    # Save thresholds & confusion
    np.save(os.path.join(CONFIG['EXPORT_DIR'], 'best_thresholds.npy'), thresholds)
    with open(os.path.join(CONFIG['EXPORT_DIR'], 'confusion_stats.json'), 'w') as f:
        json.dump(confusion, f, indent=2)


    def representative_dataset():
        if CONFIG['QUANT_MODE'] != 'int8':
            return
        csv_train = os.path.join(CONFIG['DATASET_ROOT'], 'balanced_multilabel_train.csv')
        csv_val = os.path.join(CONFIG['DATASET_ROOT'], 'balanced_multilabel_val.csv')
        df_tr = pd.read_csv(csv_train)
        df_vl = pd.read_csv(csv_val)

        def fix_path(p):
            p = str(p).replace('\\', '/')
            if 'images/' in p:
                p = p[p.index('images/'):]
            return p

        df_tr['image'] = df_tr['image'].apply(fix_path)
        df_vl['image'] = df_vl['image'].apply(fix_path)

        paths = [os.path.join(CONFIG['DATASET_ROOT'], p)
                 for p in (df_tr['image'].tolist() + df_vl['image'].tolist())]
        random.shuffle(paths)
        for p in paths[:CONFIG['REP_SAMPLES']]:
            img = tf.io.read_file(p)
            img = tf.image.decode_jpeg(img, 3)
            img = tf.image.resize(img, (CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE']))
            img = tf.cast(img, tf.float32) / 255.0
            yield [tf.expand_dims(img, 0)]

    print(f"Exporting TFLite ({CONFIG['QUANT_MODE']}) ...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if CONFIG['QUANT_MODE'] == 'int8':
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    else:
        converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()
    tflite_path = os.path.join(CONFIG['EXPORT_DIR'], f"{CONFIG['NICKNAME']}.tflite")
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite saved: {tflite_path} ({len(tflite_model)/1024:.1f} KB)")

    # Save model summary
    summary_path = os.path.join(CONFIG['EXPORT_DIR'], 'model_summary.txt')
    with open(summary_path, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    print(f"Model summary saved: {summary_path}")

    meta = {
        'model': CONFIG['NICKNAME'],
        'classes': CLASSES,
        'thresholds': thresholds.tolist(),
        'macro_f1': float(macro_f1),
        'micro_f1': float(micro_f1),
        'input_shape': [CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'], 3],
        'quantization': CONFIG['QUANT_MODE'],
        'created': datetime.now().isoformat()
    }
    with open(os.path.join(CONFIG['EXPORT_DIR'], 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    # NEW: generate C header for deployment
    header_path = os.path.join(CONFIG['EXPORT_DIR'], 'model_data.h')
    with open(tflite_path, 'rb') as f:
        raw = f.read()
    with open(header_path, 'w') as f:
        f.write("// Auto-generated neural network model\n")
        f.write(f"// Model: {CONFIG['NICKNAME']}\n")
        f.write(f"// Quantization: {CONFIG['QUANT_MODE']}\n")
        f.write(f"// Size: {len(raw)} bytes\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"const unsigned int model_data_len = {len(raw)};\n")
        f.write("const unsigned char model_data[] = {")
        for i, b in enumerate(raw):
            if i % 16 == 0:
                f.write("\n ")
            f.write(f"0x{b:02x}, ")
        f.write("\n};\n\n")
        f.write(f"const int NUM_CLASSES = {NUM_CLASSES};\n")
        f.write("const char* const CLASS_NAMES[] = {")
        f.write(", ".join([f'\"{c}\"' for c in CLASSES]))
        f.write("};\n")
        f.write("const float CLASS_THRESHOLDS[] = {")
        f.write(", ".join([f"{t:.6f}f" for t in thresholds]))
        f.write("};\n")
    print(f"C header created: {header_path}")
def main():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for g in gpus: tf.config.experimental.set_memory_growth(g, True)
        print(f"Using GPU: {gpus[0].name}")
    else:
        print("Using CPU")
    train()
    gc.collect()

if __name__ == "__main__":
    main()