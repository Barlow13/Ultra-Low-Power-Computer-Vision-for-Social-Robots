import os
import csv
import random
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count
import time
from functools import partial
import numpy as np

# === CONFIGURATION ===
dataset_root = '../datasets/coco'
splits = ['train2017', 'val2017']
target_classes = {0: 0, 16: 1, 15: 2}  # COCO class_id → [person, dog, cat]
class_names = ['person', 'dog', 'cat', 'none']
image_extensions = ('.jpg', '.jpeg', '.png')

# Balance settings
BALANCE_STRATEGY = 'mixed'  # 'undersample', 'oversample', 'mixed'
TARGET_RATIO = [1.0, 1.0, 1.0]  # Target ratio for [person, dog, cat]
MAX_SAMPLES_PER_CLASS = 5000  # Maximum samples per class
MIN_SAMPLES_PER_CLASS = 2000  # Minimum samples per class
MAX_NEG_RATIO = 0.2  # 20% negative samples
OVERSAMPLE_FACTOR = 3  # How many times to duplicate rare samples

train_val_split = 0.8

# Performance settings
num_processes = min(cpu_count() - 1, 24)
chunk_size = 100

# === WORKER FUNCTION ===
def process_single_file(args):
    """Process a single label file"""
    label_file, label_dir, image_dir, dataset_root = args
    
    if not label_file.endswith('.txt'):
        return None
    
    base_name = label_file[:-4]
    
    # Find image file
    image_path = None
    for ext in image_extensions:
        test_path = os.path.join(image_dir, base_name + ext)
        if os.path.isfile(test_path):
            image_path = test_path
            break
    
    if image_path is None:
        return None
    
    # Read labels
    label_path = os.path.join(label_dir, label_file)
    label = [0, 0, 0]  # person, dog, cat
    
    try:
        with open(label_path, 'r') as f:
            for line in f:
                if line.strip():
                    cls = int(line.split()[0])
                    if cls in target_classes:
                        label[target_classes[cls]] = 1
    except:
        return None
    
    # Add 'none' class: set to 1 if all other classes are 0
    none_label = 1 if sum(label) == 0 else 0
    label.append(none_label)  # [person, dog, cat, none]

    # Return relative path and label
    rel_path = os.path.relpath(image_path, dataset_root)
    return (rel_path, label)

# === BALANCING FUNCTIONS ===
def analyze_dataset(samples):
    """Analyze dataset composition"""
    class_samples = defaultdict(list)
    negative_samples = []
    multi_label_samples = []
    
    for path, label in samples:
        label_sum = sum(label)
        
        if label_sum == 0:
            negative_samples.append((path, label))
        elif label_sum > 1:
            multi_label_samples.append((path, label))
        
        # Track samples for each class
        for i, present in enumerate(label):
            if present:
                class_samples[class_names[i]].append((path, label))
    
    return class_samples, negative_samples, multi_label_samples

def balance_dataset(samples, strategy='mixed'):
    """Balance dataset using specified strategy"""
    class_samples, negative_samples, multi_label_samples = analyze_dataset(samples)
    
    print("\n📊 Original class distribution:")
    for cls in class_names:
        print(f"   {cls}: {len(class_samples[cls])}")
    print(f"   negative: {len(negative_samples)}")
    print(f"   multi-label: {len(multi_label_samples)}")
    
    balanced_samples = []
    
    if strategy == 'undersample':
        # Find minimum class count
        min_count = min(len(class_samples[cls]) for cls in class_names)
        target_count = min(min_count, MAX_SAMPLES_PER_CLASS)
        
        print(f"\n⬇️  Undersampling to {target_count} samples per class")
        
        for cls in class_names:
            sampled = random.sample(class_samples[cls], 
                                  min(target_count, len(class_samples[cls])))
            balanced_samples.extend(sampled)
    
    elif strategy == 'oversample':
        # Find maximum class count
        max_count = max(len(class_samples[cls]) for cls in class_names)
        target_count = min(max_count, MAX_SAMPLES_PER_CLASS)
        
        print(f"\n⬆️  Oversampling to {target_count} samples per class")
        
        for cls in class_names:
            samples_list = class_samples[cls]
            current_count = len(samples_list)
            
            if current_count < target_count:
                # Oversample by duplicating
                factor = target_count // current_count
                remainder = target_count % current_count
                
                balanced_samples.extend(samples_list * factor)
                balanced_samples.extend(random.sample(samples_list, remainder))
            else:
                balanced_samples.extend(samples_list)
    
    elif strategy == 'mixed':
        # Mixed strategy: undersample majority, oversample minority
        counts = [len(class_samples[cls]) for cls in class_names]
        median_count = int(np.median(counts))
        target_count = min(median_count * 2, MAX_SAMPLES_PER_CLASS)
        target_count = max(target_count, MIN_SAMPLES_PER_CLASS)
        
        print(f"\n🔄 Mixed strategy: target {target_count} samples per class")
        
        for cls in class_names:
            samples_list = class_samples[cls]
            current_count = len(samples_list)
            
            if current_count > target_count:
                # Undersample
                print(f"   ⬇️  Undersampling {cls}: {current_count} → {target_count}")
                sampled = random.sample(samples_list, target_count)
                balanced_samples.extend(sampled)
            elif current_count < target_count:
                # Oversample
                factor = min(OVERSAMPLE_FACTOR, target_count // current_count)
                new_count = min(current_count * factor, target_count)
                print(f"   ⬆️  Oversampling {cls}: {current_count} → {new_count}")
                
                balanced_samples.extend(samples_list)
                if factor > 1:
                    additional = random.choices(samples_list, 
                                              k=new_count - current_count)
                    balanced_samples.extend(additional)
            else:
                balanced_samples.extend(samples_list)
    
    # Remove duplicates while preserving multi-label samples
    seen = set()
    unique_balanced = []
    
    # First, add all multi-label samples (they're valuable)
    for sample in multi_label_samples:
        if sample[0] not in seen:
            unique_balanced.append(sample)
            seen.add(sample[0])
    
    # Then add single-label samples
    for sample in balanced_samples:
        if sample[0] not in seen:
            unique_balanced.append(sample)
            seen.add(sample[0])
    
    # Add negative samples (limited)
    num_positives = len(unique_balanced)
    max_negatives = int(num_positives * MAX_NEG_RATIO)
    
    if negative_samples:
        neg_to_add = min(len(negative_samples), max_negatives)
        print(f"\n➕ Adding {neg_to_add} negative samples ({MAX_NEG_RATIO*100:.0f}% of positives)")
        unique_balanced.extend(random.sample(negative_samples, neg_to_add))
    
    return unique_balanced

def augment_minority_classes(samples, minority_threshold=0.1):
    """Create synthetic variations for minority classes"""
    class_counts = Counter()
    for _, label in samples:
        for i, present in enumerate(label):
            if present:
                class_counts[class_names[i]] += 1
    
    total_positive = sum(class_counts.values())
    minority_classes = [cls for cls, count in class_counts.items() 
                       if count / total_positive < minority_threshold]
    
    if minority_classes:
        print(f"\n🔧 Augmenting minority classes: {minority_classes}")
        # In practice, you'd apply transformations here
        # For now, we'll just duplicate some samples
        
    return samples

# === MAIN PROCESSING ===
if __name__ == '__main__':
    print(f"🚀 Balanced Dataset Generator for Multi-Label Classification")
    print(f"🔧 Using {num_processes} processes")
    print(f"⚖️  Balance strategy: {BALANCE_STRATEGY}")
    start_time = time.time()
    
    all_results = []
    
    for split in splits:
        split_start = time.time()
        label_dir = os.path.join(dataset_root, 'labels', split)
        image_dir = os.path.join(dataset_root, 'images', split)
        
        if not os.path.exists(label_dir):
            print(f"⚠️  Warning: {label_dir} not found, skipping...")
            continue
        
        print(f"\n📁 Processing {split}...")
        label_files = os.listdir(label_dir)
        print(f"📊 Found {len(label_files)} files to process")
        
        # Prepare arguments
        args_list = [(f, label_dir, image_dir, dataset_root) for f in label_files]
        
        # Process with pool
        with Pool(processes=num_processes) as pool:
            results = []
            processed = 0
            
            for result in pool.imap_unordered(process_single_file, args_list, chunksize=chunk_size):
                if result is not None:
                    results.append(result)
                
                processed += 1
                if processed % 5000 == 0:
                    print(f"   ⚡ Processed: {processed}/{len(label_files)} files")
            
            all_results.extend(results)
        
        print(f"✅ {split} completed in {time.time() - split_start:.2f} seconds")
    
    print(f"\n⏱️  Total processing time: {time.time() - start_time:.2f} seconds")
    print(f"📊 Total samples found: {len(all_results)}")
    
    # Balance the dataset
    print("\n" + "="*50)
    print("BALANCING DATASET")
    print("="*50)
    
    balanced_dataset = balance_dataset(all_results, strategy=BALANCE_STRATEGY)
    
    # Optionally augment minority classes
    balanced_dataset = augment_minority_classes(balanced_dataset)
    
    # Shuffle
    random.shuffle(balanced_dataset)
    
    # Final statistics
    final_class_counts = Counter()
    for _, label in balanced_dataset:
        for i, present in enumerate(label):
            if present:
                final_class_counts[class_names[i]] += 1
    
    print("\n📊 Final balanced distribution:")
    total_balanced = sum(final_class_counts.values())
    for cls in class_names:
        count = final_class_counts[cls]
        percentage = (count / total_balanced * 100) if total_balanced > 0 else 0
        print(f"   {cls}: {count} ({percentage:.1f}%)")
    
    # Write CSVs
    print("\n💾 Writing CSV files...")
    
    def write_csv(filename, data):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image', 'person', 'dog', 'cat', 'none'])  # Add 'none'
            for path, label in data:
                writer.writerow([path] + label)
    
    # Write all CSVs
    output_csv = os.path.join(dataset_root, 'balanced_multilabel_dataset.csv')
    write_csv(output_csv, balanced_dataset)
    
    split_idx = int(len(balanced_dataset) * train_val_split)
    train_csv = os.path.join(dataset_root, 'balanced_multilabel_train.csv')
    val_csv = os.path.join(dataset_root, 'balanced_multilabel_val.csv')
    
    write_csv(train_csv, balanced_dataset[:split_idx])
    write_csv(val_csv, balanced_dataset[split_idx:])
    
    print(f"\n✅ Complete! Total time: {time.time() - start_time:.2f} seconds")
    print(f"📄 Created:")
    print(f"   - {os.path.basename(output_csv)} ({len(balanced_dataset)} samples)")
    print(f"   - {os.path.basename(train_csv)} ({split_idx} samples)")
    print(f"   - {os.path.basename(val_csv)} ({len(balanced_dataset) - split_idx} samples)")
    
    # Save balance report
    report_path = os.path.join(dataset_root, 'balance_report.txt')
    with open(report_path, 'w') as f:
        f.write("Dataset Balance Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Strategy: {BALANCE_STRATEGY}\n")
        f.write(f"Total samples: {len(balanced_dataset)}\n\n")
        f.write("Class distribution:\n")
        for cls in class_names:
            count = final_class_counts[cls]
            percentage = (count / total_balanced * 100) if total_balanced > 0 else 0
            f.write(f"  {cls}: {count} ({percentage:.1f}%)\n")
    
    print(f"\n📄 Balance report saved to: {report_path}")