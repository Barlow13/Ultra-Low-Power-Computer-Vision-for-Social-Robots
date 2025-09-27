import os
import random
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from datetime import datetime
import json
import gc

# ─── SUPPRESS LOGGING ─────────────────────────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
CONFIG = {
    # Training parameters
    'BATCH_SIZE': 16,  # Optimized batch size
    'FROZEN_EPOCHS': 32,  # Initial frozen training
    'FINE_TUNE_EPOCHS': 128,  # Fine-tuning epochs
    'INITIAL_LR': 1e-3,
    'FINE_TUNE_LR': 1e-4,
    'MIN_LR': 1e-8,

    # Model parameters
    'IMG_HEIGHT': 128,
    'IMG_WIDTH': 128,
    'DROPOUT_RATE': 0.5,
    'MOBILENET_ALPHA': 0.35,

    # Augmentation parameters
    'MIXUP_ALPHA': 0.2,
    'AUGMENTATION_STRENGTH': 0.3,

    # System
    'SEED': 42,
    'NICKNAME': 'AdvancedMobileNetV2',
    
    # Directories
    'DATASET_ROOT': '../datasets/coco',
    'EXPORT_DIR': './export',
    'CHECKPOINT_DIR': './checkpoints'
}

# Object classes to detect
CLASSES = ['person', 'dog', 'cat', 'none']
NUM_CLASSES = len(CLASSES)

# Create necessary directories
for key, path in CONFIG.items():
    if key.endswith('_DIR') and isinstance(path, str):
        os.makedirs(path, exist_ok=True)

# Set seeds for reproducibility
random.seed(CONFIG['SEED'])
np.random.seed(CONFIG['SEED'])
tf.random.set_seed(CONFIG['SEED'])

# ─── MEMORY MANAGEMENT ────────────────────────────────────────────────────────
def clear_memory():
    """Clear GPU and system memory"""
    try:
        tf.keras.backend.clear_session()
        gc.collect()
    except:
        pass

# ─── ENHANCED FOCAL LOSS ──────────────────────────────────────────────────────
@tf.keras.utils.register_keras_serializable(package="custom_losses")
class EnhancedFocalLoss(tf.keras.losses.Loss):
    """Enhanced Focal Loss with class balancing and label smoothing"""
    
    def __init__(self, gamma=2.0, alpha=0.25, label_smoothing=0.05, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.class_weights = class_weights
    
    def call(self, y_true, y_pred):
        # Apply label smoothing
        if self.label_smoothing > 0:
            y_true = y_true * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # Clip predictions for numerical stability
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate focal loss components
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        focal_weight = alpha_t * tf.pow((1 - p_t), self.gamma)
        
        # Binary crossentropy
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        
        # Apply class weights if provided
        if self.class_weights is not None:
            bce = bce * self.class_weights
        
        # Focal loss
        focal_loss = focal_weight * bce
        
        return tf.reduce_mean(focal_loss)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "gamma": self.gamma,
            "alpha": self.alpha,
            "label_smoothing": self.label_smoothing,
            "class_weights": self.class_weights
        })
        return config

# ─── DATA LOADING FROM CSV ────────────────────────────────────────────────────
def load_csv_dataset():
    """Load dataset from CSV files"""
    train_csv = os.path.join(CONFIG['DATASET_ROOT'], 'balanced_multilabel_train.csv')
    val_csv = os.path.join(CONFIG['DATASET_ROOT'], 'balanced_multilabel_val.csv')
    
    print("Loading CSV datasets...")
    
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    
    print(f"Loaded {len(df_train)} training and {len(df_val)} validation samples")
    
    # Print detailed class distribution
    for df, name in [(df_train, 'Train'), (df_val, 'Validation')]:
        print(f"\n{name} Set Statistics:")
        print(f"  Total samples: {len(df)}")
        for cls in CLASSES:
            count = df[cls].sum()
            print(f"  {cls}: {count} ({count/len(df)*100:.1f}%)")
    
    return df_train, df_val

# ─── ENHANCED IMAGE PROCESSING ────────────────────────────────────────────────
@tf.function
def load_and_preprocess_image(path, label):
    """Load and preprocess image with proper normalization"""
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [CONFIG['IMG_HEIGHT'], CONFIG['IMG_WIDTH']])
    img = tf.cast(img, tf.float32) / 255.0  # Normalize to [0,1]
    return img, label

@tf.function
def enhanced_augment_image(image, label):
    """Enhanced augmentation pipeline"""
    # Geometric augmentations
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    
    # Random rotation (0, 90, 180, 270 degrees)
    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k=k)
    
    # Color augmentations
    image = tf.image.random_brightness(image, CONFIG['AUGMENTATION_STRENGTH'])
    image = tf.image.random_contrast(image, 1-CONFIG['AUGMENTATION_STRENGTH'], 1+CONFIG['AUGMENTATION_STRENGTH'])
    image = tf.image.random_hue(image, 0.1)
    image = tf.image.random_saturation(image, 0.7, 1.3)
    
    # Random crop and resize
    if tf.random.uniform([]) < 0.5:
        scale = tf.random.uniform([], minval=0.8, maxval=1.0)
        h, w = tf.shape(image)[0], tf.shape(image)[1]
        new_h = tf.cast(tf.cast(h, tf.float32) * scale, tf.int32)
        new_w = tf.cast(tf.cast(w, tf.float32) * scale, tf.int32)
        offset_h = tf.random.uniform([], maxval=h - new_h + 1, dtype=tf.int32)
        offset_w = tf.random.uniform([], maxval=w - new_w + 1, dtype=tf.int32)
        image = tf.image.crop_to_bounding_box(image, offset_h, offset_w, new_h, new_w)
        image = tf.image.resize(image, [h, w])
    
    # Add noise occasionally
    if tf.random.uniform([]) < 0.3:
        noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=0.02)
        image = image + noise
    
    # Ensure values stay in [0, 1] range
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image, label

@tf.function
def mixup(images, labels, alpha=0.2):
    """Apply mixup augmentation"""
    batch_size = tf.shape(images)[0]
    indices = tf.random.shuffle(tf.range(batch_size))
    lam = tf.random.uniform([], minval=0, maxval=alpha)
    
    mixed_images = lam * images + (1 - lam) * tf.gather(images, indices)
    mixed_labels = lam * labels + (1 - lam) * tf.gather(labels, indices)
    
    return mixed_images, mixed_labels

def create_dataset(df, batch_size, training=False, apply_mixup=False):
    """Create optimized TensorFlow dataset"""
    paths = [os.path.join(CONFIG['DATASET_ROOT'], path) for path in df['image']]
    labels = df[CLASSES].values.astype(np.float32)
    
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    if training:
        dataset = dataset.shuffle(2048, reshuffle_each_iteration=True)
        dataset = dataset.map(enhanced_augment_image, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        
        if apply_mixup:
            dataset = dataset.map(
                lambda x, y: mixup(x, y, CONFIG['MIXUP_ALPHA']),
                num_parallel_calls=tf.data.AUTOTUNE
            )
    else:
        dataset = dataset.cache()
        dataset = dataset.batch(batch_size)
    
    return dataset.prefetch(tf.data.AUTOTUNE)

# ─── ADVANCED MODEL ARCHITECTURE ──────────────────────────────────────────────
def build_advanced_mobilenetv2(input_shape, trainable_base=False, fine_tuning=False):
    """Build advanced MobileNetV2 with improved architecture"""
    
    # Load pre-trained MobileNetV2
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        alpha=CONFIG['MOBILENET_ALPHA']
    )
    
    # Configure trainability
    base_model.trainable = trainable_base
    
    if fine_tuning and trainable_base:
        # Only train the last few layers during fine-tuning
        for layer in base_model.layers[:-30]:
            layer.trainable = False
    
    # Build the complete model
    inputs = layers.Input(shape=input_shape, dtype='float32')
    
    # Data augmentation layer (applied during training)
    x = inputs
    
    # Base model
    x = base_model(x, training=trainable_base)
    
    # Advanced head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(CONFIG['DROPOUT_RATE'])(x)
    
    # First dense layer
    x = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(CONFIG['DROPOUT_RATE'] * 0.8)(x)
    
    # Second dense layer
    x = layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(CONFIG['DROPOUT_RATE'] * 0.6)(x)
    
    # Output layer
    outputs = layers.Dense(NUM_CLASSES, activation='sigmoid')(x)
    
    model = models.Model(inputs, outputs, name=CONFIG['NICKNAME'])
    
    # Compile with appropriate learning rate
    lr = CONFIG['FINE_TUNE_LR'] if fine_tuning else CONFIG['INITIAL_LR']
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr, clipnorm=1.0),
        loss=EnhancedFocalLoss(gamma=2.0, alpha=0.25, label_smoothing=0.05),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    return model

# ─── ADVANCED CALLBACKS ───────────────────────────────────────────────────────
class AdvancedThresholdOptimizer(tf.keras.callbacks.Callback):
    """Advanced threshold optimization with class-specific strategies"""
    
    def __init__(self, validation_data, classes, viz_callback=None):
        super().__init__()
        self.validation_data = validation_data
        self.classes = classes
        self.viz_callback = viz_callback
        self.thresholds = np.array([0.5] * len(classes))
        self.best_thresholds = self.thresholds.copy()
        self.best_f1 = 0
        self.class_weights = np.ones(len(classes))
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # Get predictions and true labels
        y_true, y_pred = [], []
        for x, y in self.validation_data:
            y_true.append(y.numpy())
            y_pred.append(self.model.predict(x, verbose=0))
        
        y_true = np.vstack(y_true)
        y_pred = np.vstack(y_pred)
        
        # Calculate class distribution for weighting
        class_counts = np.sum(y_true, axis=0)
        total_positives = np.sum(class_counts)
        
        if total_positives > 0:
            for i, count in enumerate(class_counts):
                if count > 0:
                    self.class_weights[i] = min(5.0, total_positives / (count * len(self.classes)))
        
        # Optimize thresholds every 4 epochs
        if (epoch + 1) % 4 == 0:
            for i, class_name in enumerate(self.classes):
                best_f1 = 0
                best_t = 0.5
                
                # Use adaptive threshold ranges based on class frequency
                if class_counts[i] < 10:  # Rare class
                    thresholds = np.linspace(0.05, 0.7, 50)
                else:  # Common class
                    thresholds = np.linspace(0.1, 0.8, 30)
                
                for t in thresholds:
                    y_pred_bin = (y_pred[:, i] >= t)
                    
                    # Use F2 score for rare classes to emphasize recall
                    if class_counts[i] < 10:
                        precision = precision_score(y_true[:, i], y_pred_bin, zero_division=0)
                        recall = recall_score(y_true[:, i], y_pred_bin, zero_division=0)
                        if precision > 0 and recall > 0:
                            f_score = (5 * precision * recall) / (4 * precision + recall)
                        else:
                            f_score = 0
                    else:
                        f_score = f1_score(y_true[:, i], y_pred_bin, zero_division=0)
                    
                    if f_score > best_f1:
                        best_f1 = f_score
                        best_t = t
                
                self.thresholds[i] = best_t
        
        # Calculate overall metrics
        y_pred_bin = (y_pred >= self.thresholds)
        f1_macro = f1_score(y_true, y_pred_bin, average='macro', zero_division=0)
        
        if f1_macro > self.best_f1:
            self.best_f1 = f1_macro
            self.best_thresholds = self.thresholds.copy()
        
        # Add metrics to logs
        logs['val_f1_macro'] = f1_macro
        
        # Update visualization callback
        if self.viz_callback:
            self.viz_callback.set_thresholds(self.thresholds)
        
        # Print progress
        if (epoch + 1) % 4 == 0:
            print(f"\nThresholds: {self.thresholds.round(3)}, F1: {f1_macro:.4f}")
            
            # Per-class F1 scores
            class_f1 = {}
            for i, class_name in enumerate(self.classes):
                class_f1[class_name] = f1_score(y_true[:, i], y_pred_bin[:, i], zero_division=0)
            print(f"Class F1: {', '.join([f'{c}={v:.3f}' for c, v in class_f1.items()])}")

class EnhancedVisualizationCallback(tf.keras.callbacks.Callback):
    """Enhanced visualization with better sample selection"""
    
    def __init__(self, df_val, export_dir, thresholds):
        super().__init__()
        self.df_val = df_val
        self.export_dir = export_dir
        self.thresholds = thresholds
    
    def set_thresholds(self, thresholds):
        self.thresholds = thresholds
    
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 8 == 0 or epoch == 0:
            self._create_visualization(epoch)
    
    def _create_visualization(self, epoch):
        """Create comprehensive visualization"""
        num_images = 16
        nrows, ncols = 4, 4
        
        # Stratified sampling for better representation
        samples_per_class = {}
        for cls in CLASSES:
            if cls != 'none':
                class_samples = self.df_val[self.df_val[cls] == 1]
                if len(class_samples) > 0:
                    samples_per_class[cls] = class_samples.sample(
                        min(3, len(class_samples)), 
                        random_state=np.random.randint(0, 10000)
                    )
        
        # Add some 'none' samples
        none_samples = self.df_val[self.df_val['none'] == 1]
        if len(none_samples) > 0:
            samples_per_class['none'] = none_samples.sample(
                min(4, len(none_samples)), 
                random_state=np.random.randint(0, 10000)
            )
        
        # Combine all samples
        all_samples = []
        for samples in samples_per_class.values():
            all_samples.extend([row for _, row in samples.iterrows()])
        
        # Pad with random samples if needed
        while len(all_samples) < num_images:
            random_sample = self.df_val.sample(1, random_state=np.random.randint(0, 10000)).iloc[0]
            all_samples.append(random_sample)
        
        # Limit to num_images
        all_samples = all_samples[:num_images]
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
        axes = axes.flatten()
        
        for idx, row in enumerate(all_samples):
            if idx >= num_images:
                break
            
            # Load and preprocess image
            img_path = os.path.join(CONFIG['DATASET_ROOT'], row['image'])
            try:
                img = cv2.imread(img_path)
                if img is None:
                    raise FileNotFoundError(f"Image not found: {img_path}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img, (CONFIG['IMG_WIDTH'], CONFIG['IMG_HEIGHT']))
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                axes[idx].axis('off')
                continue
            
            # Predict
            img_input = np.expand_dims(img_resized.astype(np.float32) / 255.0, axis=0)
            pred = self.model.predict(img_input, verbose=0)[0]
            
            # Get labels
            true_labels = [CLASSES[i] for i, cls in enumerate(CLASSES) if row[cls] == 1]
            pred_labels = [CLASSES[i] for i, p in enumerate(pred) if p >= self.thresholds[i]]
            
            # Display
            axes[idx].imshow(img_resized)
            axes[idx].axis('off')
            
            title = f"True: {', '.join(true_labels) or 'None'}\n"
            title += f"Pred: {', '.join(pred_labels) or 'None'}\n"
            title += f"Conf: [{', '.join(f'{p:.2f}' for p in pred)}]"
            axes[idx].set_title(title, fontsize=9)
        
        # Hide unused axes
        for j in range(len(all_samples), nrows*ncols):
            axes[j].axis('off')
        
        plt.tight_layout()
        try:
            plt.savefig(os.path.join(self.export_dir, f'predictions_epoch_{epoch+1}.png'), dpi=100)
        except Exception as e:
            print(f"Error saving visualization: {e}")
        plt.close()
        
        # Memory cleanup
        gc.collect()

# ─── TWO-STAGE TRAINING FUNCTION ──────────────────────────────────────────────
def train_two_stage_model(df_train, df_val):
    """Advanced two-stage training with frozen base then fine-tuning"""
    
    print("\n" + "="*60)
    print("Starting Two-Stage Training")
    print(f"Stage 1: {CONFIG['FROZEN_EPOCHS']} epochs (frozen base)")
    print(f"Stage 2: {CONFIG['FINE_TUNE_EPOCHS']} epochs (fine-tuning)")
    print("="*60 + "\n")
    
    # Create datasets
    train_ds = create_dataset(df_train, CONFIG['BATCH_SIZE'], training=True, apply_mixup=True)
    val_ds = create_dataset(df_val, CONFIG['BATCH_SIZE'], training=False)
    
    # ─── STAGE 1: FROZEN BASE TRAINING ───────────────────────────────────────
    print("\n=== Stage 1: Training with Frozen Base ===")
    
    model_stage1 = build_advanced_mobilenetv2(
        (CONFIG['IMG_HEIGHT'], CONFIG['IMG_WIDTH'], 3),
        trainable_base=False
    )
    
    print("Stage 1 Model Architecture:")
    model_stage1.summary()
    
    # Stage 1 callbacks
    viz_callback = EnhancedVisualizationCallback(df_val, CONFIG['EXPORT_DIR'], np.array([0.5] * len(CLASSES)))
    threshold_callback = AdvancedThresholdOptimizer(val_ds, CLASSES, viz_callback)
    
    stage1_callbacks = [
        threshold_callback,
        viz_callback,
        ModelCheckpoint(
            os.path.join(CONFIG['CHECKPOINT_DIR'], 'best_model_stage1.keras'),
            monitor='val_f1_macro',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_f1_macro',
            mode='max',
            patience=12,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_f1_macro',
            mode='max',
            factor=0.5,
            patience=6,
            min_lr=CONFIG['MIN_LR'],
            verbose=1
        )
    ]
    
    # Train stage 1
    history_stage1 = model_stage1.fit(
        train_ds,
        epochs=CONFIG['FROZEN_EPOCHS'],
        validation_data=val_ds,
        callbacks=stage1_callbacks,
        verbose=1
    )
    
    # ─── STAGE 2: FINE-TUNING ────────────────────────────────────────────────
    print("\n=== Stage 2: Fine-tuning ===")
    
    # Load best stage 1 model
    best_stage1_model = tf.keras.models.load_model(
        os.path.join(CONFIG['CHECKPOINT_DIR'], 'best_model_stage1.keras'),
        custom_objects={'EnhancedFocalLoss': EnhancedFocalLoss}
    )
    
    # Create fine-tuning model
    model_stage2 = build_advanced_mobilenetv2(
        (CONFIG['IMG_HEIGHT'], CONFIG['IMG_WIDTH'], 3),
        trainable_base=True,
        fine_tuning=True
    )
    
    # Transfer weights
    model_stage2.set_weights(best_stage1_model.get_weights())
    
    # Stage 2 callbacks with updated thresholds
    threshold_callback_stage2 = AdvancedThresholdOptimizer(val_ds, CLASSES, viz_callback)
    threshold_callback_stage2.thresholds = threshold_callback.best_thresholds.copy()
    viz_callback.set_thresholds(threshold_callback_stage2.thresholds)
    
    stage2_callbacks = [
        threshold_callback_stage2,
        viz_callback,
        ModelCheckpoint(
            os.path.join(CONFIG['CHECKPOINT_DIR'], 'best_model_final.keras'),
            monitor='val_f1_macro',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_f1_macro',
            mode='max',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_f1_macro',
            mode='max',
            factor=0.5,
            patience=8,
            min_lr=CONFIG['MIN_LR'],
            verbose=1
        )
    ]
    
    # Train stage 2
    history_stage2 = model_stage2.fit(
        train_ds,
        epochs=CONFIG['FINE_TUNE_EPOCHS'],
        validation_data=val_ds,
        callbacks=stage2_callbacks,
        verbose=1
    )
    
    return model_stage2, threshold_callback_stage2.best_thresholds, history_stage1, history_stage2

# ─── ENHANCED MODEL EVALUATION ────────────────────────────────────────────────
def evaluate_model_comprehensive(model, test_ds, thresholds):
    """Comprehensive model evaluation"""
    print("\n" + "="*60)
    print("Comprehensive Model Evaluation")
    print("="*60 + "\n")
    
    # Get predictions
    y_true, y_pred = [], []
    for x, y in test_ds:
        y_true.append(y.numpy())
        y_pred.append(model.predict(x, verbose=0))
    
    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)
    y_pred_bin = (y_pred >= thresholds)
    
    # Overall metrics
    metrics = {
        'f1_macro': f1_score(y_true, y_pred_bin, average='macro', zero_division=0),
        'f1_micro': f1_score(y_true, y_pred_bin, average='micro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred_bin, average='weighted', zero_division=0),
        'precision_macro': precision_score(y_true, y_pred_bin, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred_bin, average='macro', zero_division=0),
    }
    
    # Per-class metrics
    class_metrics = {}
    for i, cls in enumerate(CLASSES):
        class_metrics[cls] = {
            'f1': f1_score(y_true[:, i], y_pred_bin[:, i], zero_division=0),
            'precision': precision_score(y_true[:, i], y_pred_bin[:, i], zero_division=0),
            'recall': recall_score(y_true[:, i], y_pred_bin[:, i], zero_division=0),
            'support': int(np.sum(y_true[:, i])),
            'threshold': float(thresholds[i]),
            'avg_confidence': float(np.mean(y_pred[:, i]))
        }
    
    # Print results
    print("Overall Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\nPer-class Performance:")
    for cls, m in class_metrics.items():
        print(f"  {cls:8s}: F1={m['f1']:.3f}, P={m['precision']:.3f}, "
              f"R={m['recall']:.3f}, Support={m['support']}, "
              f"Threshold={m['threshold']:.3f}, AvgConf={m['avg_confidence']:.3f}")
    
    return metrics, class_metrics

# ─── ENHANCED TFLITE EXPORT ───────────────────────────────────────────────────
def export_to_tflite_advanced(model, thresholds, df_train):
    """Advanced TFLite export with better quantization"""
    print("\n" + "="*60)
    print("Advanced TFLite Conversion")
    print("="*60 + "\n")
    
    # Create comprehensive representative dataset
    def representative_dataset():
        # Ensure all classes are represented
        class_samples = {}
        for cls in CLASSES:
            if cls != 'none':
                samples = df_train[df_train[cls] == 1]
                if len(samples) > 0:
                    class_samples[cls] = samples.sample(min(25, len(samples)))
        
        # Add none samples
        none_samples = df_train[df_train['none'] == 1]
        if len(none_samples) > 0:
            class_samples['none'] = none_samples.sample(min(25, len(none_samples)))
        
        # Generate calibration data
        for cls, samples in class_samples.items():
            for _, row in samples.iterrows():
                img_path = os.path.join(CONFIG['DATASET_ROOT'], row['image'])
                try:
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, (CONFIG['IMG_WIDTH'], CONFIG['IMG_HEIGHT']))
                    img = img.astype(np.float32) / 255.0
                    yield [np.expand_dims(img, axis=0)]
                except:
                    continue
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_model = converter.convert()
    
    # Save TFLite model
    tflite_path = os.path.join(CONFIG['EXPORT_DIR'], f"{CONFIG['NICKNAME']}.tflite")
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved: {tflite_path}")
    print(f"Model size: {len(tflite_model) / 1024:.2f} KB")
    
    # Generate comprehensive C header
    header_path = os.path.join(CONFIG['EXPORT_DIR'], 'model_data.h')
    with open(header_path, 'w') as f:
        f.write('#ifndef MODEL_DATA_H\n')
        f.write('#define MODEL_DATA_H\n\n')
        f.write('#include <stdint.h>\n\n')
        f.write(f'// Model: {CONFIG["NICKNAME"]}\n')
        f.write(f'// Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'// Input size: {CONFIG["IMG_WIDTH"]}x{CONFIG["IMG_HEIGHT"]}x3\n')
        f.write(f'// Classes: {", ".join(CLASSES)}\n\n')
        
        # Model data
        f.write('const uint8_t model_data[] = {\n')
        for i, byte in enumerate(tflite_model):
            if i % 12 == 0:
                f.write('\n    ')
            f.write(f'0x{byte:02x}, ')
        f.write('\n};\n\n')
        
        f.write(f'const uint32_t model_data_len = {len(tflite_model)};\n\n')
        
        # Configuration
        f.write('// Model configuration\n')
        f.write(f'#define MODEL_INPUT_WIDTH {CONFIG["IMG_WIDTH"]}\n')
        f.write(f'#define MODEL_INPUT_HEIGHT {CONFIG["IMG_HEIGHT"]}\n')
        f.write(f'#define MODEL_NUM_CLASSES {NUM_CLASSES}\n\n')
        
        # Class names
        f.write('const char* model_classes[MODEL_NUM_CLASSES] = {\n')
        for cls in CLASSES:
            f.write(f'    "{cls}",\n')
        f.write('};\n\n')
        
        # Thresholds
        f.write('// Detection thresholds (float)\n')
        f.write('const float model_thresholds[MODEL_NUM_CLASSES] = {\n')
        for i, t in enumerate(thresholds):
            f.write(f'    {t:.6f}f,  // {CLASSES[i]}\n')
        f.write('};\n\n')
        
        # Quantized thresholds
        f.write('// Detection thresholds (int8, -128 to 127)\n')
        f.write('const int8_t model_thresholds_int8[MODEL_NUM_CLASSES] = {\n')
        for i, t in enumerate(thresholds):
            # Convert threshold to int8 range
            int8_threshold = int((t - 0.5) * 255)  # Map [0,1] to [-127,128]
            int8_threshold = max(-128, min(127, int8_threshold))
            f.write(f'    {int8_threshold},  // {CLASSES[i]} ({t:.3f})\n')
        f.write('};\n\n')
        
        f.write('#endif // MODEL_DATA_H\n')
    
    print(f"C header generated: {header_path}")
    
    return tflite_path

# ─── ENHANCED VISUALIZATION ───────────────────────────────────────────────────
def plot_comprehensive_training_history(history1, history2):
    """Plot comprehensive training history for both stages"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    
    # Combine histories
    combined_loss = history1.history['loss'] + history2.history['loss']
    combined_val_loss = history1.history['val_loss'] + history2.history['val_loss']
    combined_acc = history1.history['accuracy'] + history2.history['accuracy']
    combined_val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    combined_f1 = history1.history.get('val_f1_macro', []) + history2.history.get('val_f1_macro', [])
    
    stage1_epochs = len(history1.history['loss'])
    total_epochs = range(1, len(combined_loss) + 1)
    
    # Loss
    axes[0, 0].plot(total_epochs, combined_loss, label='Train Loss', color='blue')
    axes[0, 0].plot(total_epochs, combined_val_loss, label='Val Loss', color='red')
    axes[0, 0].axvline(x=stage1_epochs, color='green', linestyle='--', alpha=0.7, label='Fine-tuning starts')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(total_epochs, combined_acc, label='Train Accuracy', color='blue')
    axes[0, 1].plot(total_epochs, combined_val_acc, label='Val Accuracy', color='red')
    axes[0, 1].axvline(x=stage1_epochs, color='green', linestyle='--', alpha=0.7, label='Fine-tuning starts')
    axes[0, 1].set_title('Training Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 Score
    if combined_f1:
        f1_epochs = range(1, len(combined_f1) + 1)
        axes[1, 0].plot(f1_epochs, combined_f1, label='Val F1 Macro', color='purple', marker='o')
        axes[1, 0].axvline(x=stage1_epochs, color='green', linestyle='--', alpha=0.7, label='Fine-tuning starts')
        axes[1, 0].set_title('F1 Score (Macro)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Learning Rate
    lr_history = []
    for hist in [history1, history2]:
        if 'lr' in hist.history:
            lr_history.extend(hist.history['lr'])
    
    if lr_history:
        axes[1, 1].plot(range(1, len(lr_history) + 1), lr_history, label='Learning Rate', color='orange')
        axes[1, 1].axvline(x=stage1_epochs, color='green', linestyle='--', alpha=0.7, label='Fine-tuning starts')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # Precision and Recall
    combined_precision = history1.history.get('precision', []) + history2.history.get('precision', [])
    combined_recall = history1.history.get('recall', []) + history2.history.get('recall', [])
    combined_val_precision = history1.history.get('val_precision', []) + history2.history.get('val_precision', [])
    combined_val_recall = history1.history.get('val_recall', []) + history2.history.get('val_recall', [])
    
    if combined_precision:
        axes[2, 0].plot(total_epochs, combined_precision, label='Train Precision', color='blue')
        axes[2, 0].plot(total_epochs, combined_val_precision, label='Val Precision', color='red')
        axes[2, 0].axvline(x=stage1_epochs, color='green', linestyle='--', alpha=0.7, label='Fine-tuning starts')
        axes[2, 0].set_title('Precision')
        axes[2, 0].set_xlabel('Epoch')
        axes[2, 0].set_ylabel('Precision')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
    
    if combined_recall:
        axes[2, 1].plot(total_epochs, combined_recall, label='Train Recall', color='blue')
        axes[2, 1].plot(total_epochs, combined_val_recall, label='Val Recall', color='red')
        axes[2, 1].axvline(x=stage1_epochs, color='green', linestyle='--', alpha=0.7, label='Fine-tuning starts')
        axes[2, 1].set_title('Recall')
        axes[2, 1].set_xlabel('Epoch')
        axes[2, 1].set_ylabel('Recall')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['EXPORT_DIR'], 'comprehensive_training_history.png'), dpi=150)
    plt.close()

def save_comprehensive_report(model, val_ds, thresholds, metrics, class_metrics, history1, history2):
    """Save comprehensive training and evaluation report"""
    
    # Classification report
    y_true, y_pred = [], []
    for x, y in val_ds:
        y_true.append(y.numpy())
        y_pred.append(model.predict(x, verbose=0))
    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)
    y_pred_bin = (y_pred >= thresholds)
    
    report = classification_report(y_true, y_pred_bin, target_names=CLASSES, zero_division=0)
    
    # Save comprehensive report
    report_path = os.path.join(CONFIG['EXPORT_DIR'], 'comprehensive_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"=== {CONFIG['NICKNAME']} Training Report ===\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("=== Configuration ===\n")
        for key, value in CONFIG.items():
            f.write(f"{key}: {value}\n")
        f.write(f"Classes: {', '.join(CLASSES)}\n\n")
        
        f.write("=== Training Summary ===\n")
        f.write(f"Stage 1 (Frozen): {len(history1.history['loss'])} epochs\n")
        f.write(f"Stage 2 (Fine-tune): {len(history2.history['loss'])} epochs\n")
        f.write(f"Total epochs: {len(history1.history['loss']) + len(history2.history['loss'])}\n\n")
        
        f.write("=== Final Performance ===\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
        f.write("\n")
        
        f.write("=== Per-Class Performance ===\n")
        for cls, m in class_metrics.items():
            f.write(f"\n{cls}:\n")
            for metric, value in m.items():
                f.write(f"  {metric}: {value}\n")
        
        f.write("\n=== Optimized Thresholds ===\n")
        for i, cls in enumerate(CLASSES):
            f.write(f"{cls}: {thresholds[i]:.6f}\n")
        
        f.write("\n=== Classification Report ===\n")
        f.write(report)
    
    print(f"Comprehensive report saved: {report_path}")

# ─── MAIN EXECUTION ───────────────────────────────────────────────────────────
def main():
    """Main execution function"""
    print(f"\n{'='*70}")
    print(f"Advanced Multi-Label Classification Training")
    print(f"Model: {CONFIG['NICKNAME']}")
    print(f"Classes: {', '.join(CLASSES)}")
    print(f"Input Size: {CONFIG['IMG_WIDTH']}x{CONFIG['IMG_HEIGHT']}")
    print(f"Two-Stage Training: Frozen + Fine-tuning")
    print(f"{'='*70}\n")
    
    # Configure GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU configured: {gpus[0].name}")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("No GPU available, using CPU")
    
    # Load dataset
    df_train, df_val = load_csv_dataset()
    
    # Train model with two-stage approach
    model, thresholds, history1, history2 = train_two_stage_model(df_train, df_val)
    
    # Comprehensive evaluation
    val_ds = create_dataset(df_val, CONFIG['BATCH_SIZE'], training=False)
    metrics, class_metrics = evaluate_model_comprehensive(model, val_ds, thresholds)
    
    # Export to TFLite
    tflite_path = export_to_tflite_advanced(model, thresholds, df_train)
    
    # Generate comprehensive visualizations and reports
    plot_comprehensive_training_history(history1, history2)
    save_comprehensive_report(model, val_ds, thresholds, metrics, class_metrics, history1, history2)
    
    # Save training histories
    hist1_df = pd.DataFrame(history1.history)
    hist2_df = pd.DataFrame(history2.history)
    hist1_df.to_csv(os.path.join(CONFIG['EXPORT_DIR'], 'stage1_history.csv'), index=False)
    hist2_df.to_csv(os.path.join(CONFIG['EXPORT_DIR'], 'stage2_history.csv'), index=False)
    
    # Final summary
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"\nFinal Performance:")
    print(f"  F1 Score (Macro): {metrics['f1_macro']:.4f}")
    print(f"  F1 Score (Micro): {metrics['f1_micro']:.4f}")
    print(f"  F1 Score (Weighted): {metrics['f1_weighted']:.4f}")
    print(f"  Model Size: {os.path.getsize(tflite_path) / 1024:.2f} KB")
    
    print(f"\nOptimized Thresholds:")
    for i, cls in enumerate(CLASSES):
        print(f"  {cls}: {thresholds[i]:.4f}")
    
    print(f"\nPer-Class F1 Scores:")
    for cls, m in class_metrics.items():
        print(f"  {cls}: {m['f1']:.4f} (support: {m['support']})")
    
    # Save final metadata
    metadata = {
        'model_name': CONFIG['NICKNAME'],
        'model_size_kb': os.path.getsize(tflite_path) / 1024,
        'input_shape': [CONFIG['IMG_HEIGHT'], CONFIG['IMG_WIDTH'], 3],
        'classes': CLASSES,
        'thresholds': thresholds.tolist(),
        'metrics': metrics,
        'class_metrics': class_metrics,
        'config': CONFIG,
        'training_summary': {
            'stage1_epochs': len(history1.history['loss']),
            'stage2_epochs': len(history2.history['loss']),
            'total_epochs': len(history1.history['loss']) + len(history2.history['loss'])
        },
        'created': datetime.now().isoformat()
    }
    
    with open(os.path.join(CONFIG['EXPORT_DIR'], 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nAll files exported to: {CONFIG['EXPORT_DIR']}")
    print("="*70)
    
    # Clear memory
    clear_memory()

if __name__ == "__main__":
    main()