# =====================================================================
# SOIL CLASSIFICATION PIPELINE — MEMORY-SAFE VERSION FOR COLAB (T4 GPU)
# =====================================================================
# WHY THE ORIGINAL CRASHED
# -------------------------------------------------------------------
# `SoilDataLoader.load_image_directory()` reads every image into ONE
# giant numpy array, all at once, in system RAM (not GPU VRAM).
# For N images at 224x224x3 float32 that's N * 224*224*3*4 bytes.
# 10,000 images alone = ~6 GB. Then the pipeline immediately makes
# several MORE full copies of that array:
#   - preprocessor.normalize()      -> new array (~2x)
#   - train_test_split() x2         -> new arrays (~another 1x-2x)
#   - EDA plotting on raw `images`  -> keeps the original alive too
# Peak RAM usage ends up at 3-5x the raw dataset size, which blows
# past Colab's free-tier ~12-13 GB system RAM ceiling. This is a RAM
# problem, not a GPU memory problem — switching GPU type won't fix it.
#
# THE FIX
# -------------------------------------------------------------------
# Never materialize the full dataset as one numpy array. Instead,
# stream images from disk in batches using tf.data / Keras utilities,
# and explicitly release memory (clear_session + gc.collect + del)
# between models so training model #2 doesn't inherit model #1's
# footprint.
# =====================================================================

import os
import gc
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, mixed_precision
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ---------------------------------------------------------------
# 0. GPU / MEMORY SETUP
# ---------------------------------------------------------------
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

# Mixed precision roughly halves activation memory on a T4 and speeds
# up training. Safe default for transfer-learning image models.
mixed_precision.set_global_policy('mixed_float16')

print(f"GPUs available: {len(gpus)}")
print(f"Mixed precision policy: {mixed_precision.global_policy()}")

# ---------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------
DATA_PATH = "/content/drive/MyDrive/DatasetRubaya/CyAUG-Dataset"

MODELS_TO_TRAIN = 'fast'   # 'all', 'fast', or a custom list

CONFIG = {
    'img_size': (224, 224),
    'batch_size': 32,          # lower to 16 if you still see OOM
    'epochs': 50,
    'patience': 8,
    'learning_rate': 1e-3,
    'validation_split': 0.2,   # taken out of the "train" portion below
    'test_split': 0.2,         # taken out of the full dataset
    'seed': 42,
    # If your dataset is huge and you just want to validate the
    # pipeline runs end-to-end before a full run, cap images per class:
    'max_images_per_class': None,   # e.g. 300 for a quick smoke test
}

data_path = Path(DATA_PATH)
if not data_path.exists():
    raise FileNotFoundError(f"Data path not found: {DATA_PATH}")

output_dir = Path('soil_classification_results')
output_dir.mkdir(exist_ok=True)

# ---------------------------------------------------------------
# 2. (OPTIONAL) BUILD A CAPPED WORKING COPY OF THE DATASET
# ---------------------------------------------------------------
# Loading straight from Google Drive with tf.data works but Drive I/O
# can be slow and flaky in Colab. If you hit "Input/Output error" or
# very slow first epochs, copy the dataset to local Colab disk first:
#
#   !rm -rf /content/soil_data
#   !cp -r "{DATA_PATH}" /content/soil_data
#   data_path = Path("/content/soil_data")
#
# This does NOT load images into RAM — it's a disk-to-disk copy, so it
# won't cause an OOM, only takes a few minutes depending on dataset size.

if CONFIG['max_images_per_class']:
    import shutil
    capped_path = Path('/content/soil_data_capped')
    if capped_path.exists():
        shutil.rmtree(capped_path)
    capped_path.mkdir(parents=True)
    for class_dir in data_path.iterdir():
        if not class_dir.is_dir():
            continue
        dest = capped_path / class_dir.name
        dest.mkdir()
        files = sorted(list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')))
        for f in files[:CONFIG['max_images_per_class']]:
            shutil.copy(f, dest / f.name)
    data_path = capped_path
    print(f"✓ Using capped dataset at {data_path} "
          f"({CONFIG['max_images_per_class']} images/class max)")

# ---------------------------------------------------------------
# 3. STREAMED DATASETS (no full-array load into RAM)
# ---------------------------------------------------------------
# image_dataset_from_directory reads images off disk batch-by-batch
# during training, not all at once. This is the single biggest fix.

full_train_ds = tf.keras.utils.image_dataset_from_directory(
    data_path,
    validation_split=CONFIG['test_split'],
    subset='training',
    seed=CONFIG['seed'],
    image_size=CONFIG['img_size'],
    batch_size=CONFIG['batch_size'],
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    data_path,
    validation_split=CONFIG['test_split'],
    subset='validation',
    seed=CONFIG['seed'],
    image_size=CONFIG['img_size'],
    batch_size=CONFIG['batch_size'],
)

class_names = full_train_ds.class_names
num_classes = len(class_names)
print(f"\n✓ Found {num_classes} soil types: {class_names}")

# Split the "training" portion further into train/val
train_batches = tf.data.experimental.cardinality(full_train_ds).numpy()
val_batches = int(train_batches * CONFIG['validation_split'])
val_ds = full_train_ds.take(val_batches)
train_ds = full_train_ds.skip(val_batches)

print(f"✓ Train batches: {tf.data.experimental.cardinality(train_ds).numpy()}")
print(f"✓ Val batches:   {tf.data.experimental.cardinality(val_ds).numpy()}")
print(f"✓ Test batches:  {tf.data.experimental.cardinality(test_ds).numpy()}")

# ---------------------------------------------------------------
# 4. CLASS WEIGHTS (computed by counting files, not loading images)
# ---------------------------------------------------------------
class_counts = {}
for cname in class_names:
    n = len(list((data_path / cname).glob('*.jpg'))) + \
        len(list((data_path / cname).glob('*.png')))
    class_counts[cname] = n

total = sum(class_counts.values())
class_weight = {
    i: total / (num_classes * class_counts[cname])
    for i, cname in enumerate(class_names)
}
print("\n✓ Class weights (for imbalance):")
for i, cname in enumerate(class_names):
    print(f"   {cname}: {class_weight[i]:.3f}  ({class_counts[cname]} images)")

# ---------------------------------------------------------------
# 5. PERFORMANCE PIPELINE (cache/prefetch, augmentation, normalization)
# ---------------------------------------------------------------
AUTOTUNE = tf.data.AUTOTUNE

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

# NOTE: no manual normalization step here — each model's own
# `preprocess_input` (imagenet-style) is applied inside the model
# builder below, so we never keep a separately-normalized full copy
# of the dataset in memory.

def prep(ds, training, preprocess_fn):
    if training:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda x, y: (preprocess_fn(x), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)
    # Deliberately NOT using .cache() here: caching the decoded/augmented
    # dataset in RAM reintroduces the same OOM risk for large datasets.
    # If your dataset is small (<2-3k images) you can safely add
    # `.cache()` right before `.prefetch(AUTOTUNE)` for a speed boost.

# ---------------------------------------------------------------
# 6. MODEL FACTORY (transfer learning, memory-conscious)
# ---------------------------------------------------------------
MODEL_BUILDERS = {
    'mobilenetv2': (tf.keras.applications.MobileNetV2,
                     tf.keras.applications.mobilenet_v2.preprocess_input),
    'efficientnetb0': (tf.keras.applications.EfficientNetB0,
                         tf.keras.applications.efficientnet.preprocess_input),
    'resnet50': (tf.keras.applications.ResNet50,
                  tf.keras.applications.resnet50.preprocess_input),
    'densenet121': (tf.keras.applications.DenseNet121,
                      tf.keras.applications.densenet.preprocess_input),
}

def build_model(name, input_shape, num_classes):
    base_cls, preprocess_fn = MODEL_BUILDERS[name]
    base = base_cls(include_top=False, weights='imagenet', input_shape=input_shape)
    base.trainable = False  # start frozen; unfreeze later for fine-tuning if desired

    inputs = tf.keras.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    model = tf.keras.Model(inputs, outputs)
    return model, preprocess_fn

if MODELS_TO_TRAIN == 'all':
    model_list = ['mobilenetv2', 'efficientnetb0', 'resnet50', 'densenet121']
elif MODELS_TO_TRAIN == 'fast':
    model_list = ['mobilenetv2', 'efficientnetb0', 'resnet50']
else:
    model_list = MODELS_TO_TRAIN

# ---------------------------------------------------------------
# 7. TRAIN LOOP — with explicit cleanup between models
# ---------------------------------------------------------------
results_summary = {}

for idx, model_name in enumerate(model_list, 1):
    print(f"\n{'='*80}\n[{idx}/{len(model_list)}] TRAINING: {model_name.upper()}\n{'='*80}")

    try:
        model, preprocess_fn = build_model(
            model_name, CONFIG['img_size'] + (3,), num_classes)
        print(f"✓ Model created: {model.count_params():,} params")

        train_ready = prep(train_ds, training=True, preprocess_fn=preprocess_fn)
        val_ready = prep(val_ds, training=False, preprocess_fn=preprocess_fn)
        test_ready = prep(test_ds, training=False, preprocess_fn=preprocess_fn)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(CONFIG['learning_rate']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )

        ckpt_path = output_dir / f'{model_name}_best.keras'
        cb = [
            callbacks.EarlyStopping(monitor='val_loss',
                                     patience=CONFIG['patience'],
                                     restore_best_weights=True),
            callbacks.ModelCheckpoint(str(ckpt_path), monitor='val_loss',
                                       save_best_only=True),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
        ]

        history = model.fit(
            train_ready,
            validation_data=val_ready,
            epochs=CONFIG['epochs'],
            class_weight=class_weight,
            callbacks=cb,
            verbose=1,
        )

        test_loss, test_acc = model.evaluate(test_ready, verbose=0)
        print(f"✓ {model_name} test accuracy: {test_acc:.4f}")

        # --- Save prediction probabilities to disk instead of keeping the
        # model object around for ensembling later. These arrays are tiny
        # (n_samples x n_classes), so this can never cause an OOM no matter
        # how many models you train. ---
        def _collect_probs(ds):
            probs_list, true_list = [], []
            for xb, yb in ds:
                probs_list.append(model.predict(xb, verbose=0))
                true_list.append(yb.numpy())
            return np.concatenate(probs_list), np.concatenate(true_list)

        val_probs, val_true = _collect_probs(val_ready)
        test_probs, test_true = _collect_probs(test_ready)

        np.save(output_dir / f'{model_name}_val_probs.npy', val_probs)
        np.save(output_dir / f'{model_name}_test_probs.npy', test_probs)
        # True labels are identical across models (same fixed dataset order),
        # so only write them once.
        if not (output_dir / 'y_val_true.npy').exists():
            np.save(output_dir / 'y_val_true.npy', val_true)
        if not (output_dir / 'y_test_true.npy').exists():
            np.save(output_dir / 'y_test_true.npy', test_true)

        y_true, y_pred = test_true, np.argmax(test_probs, axis=1)
        report = classification_report(y_true, y_pred, target_names=class_names,
                                        output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix — {model_name}')
        plt.ylabel('True'); plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(output_dir / f'{model_name}_confusion_matrix.png')
        plt.close()  # IMPORTANT: close figures, don't let them pile up in RAM

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='train')
        plt.plot(history.history['val_accuracy'], label='val')
        plt.title('Accuracy'); plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='val')
        plt.title('Loss'); plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f'{model_name}_history.png')
        plt.close()

        results_summary[model_name] = {
            'val_accuracy': float((np.argmax(val_probs, axis=1) == val_true).mean()),
            'test_accuracy': float(test_acc),
            'test_loss': float(test_loss),
            'macro_f1': report['macro avg']['f1-score'],
            'checkpoint': str(ckpt_path),
        }

        # Save just the metrics + best weights to disk — do NOT keep the
        # full model object alive in a `trained_models[name] = model`
        # dict, that's what silently accumulates RAM across the loop.
        with open(output_dir / f'{model_name}_report.json', 'w') as f:
            json.dump(report, f, indent=2)

        print(f"✓ {model_name} complete. Best weights saved to {ckpt_path}")

    except Exception as e:
        print(f"\n❌ Error training {model_name}: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # --- THE KEY MEMORY CLEANUP STEP ---
        # Without this, TF/Keras layers, optimizer state, and the
        # augmentation/prefetch graphs from model N stay resident while
        # model N+1 starts building, and RAM climbs every iteration
        # until Colab kills the session.
        for _name in ('model', 'train_ready', 'val_ready', 'test_ready'):
            if _name in dir():
                exec(f'del {_name}')
        tf.keras.backend.clear_session()
        gc.collect()

print("\n" + "="*80)
print("STEP 4 SUMMARY")
print("="*80)
for name, res in results_summary.items():
    print(f"  {name:20s} val_acc={res['val_accuracy']:.4f} "
          f"test_acc={res['test_accuracy']:.4f} macro_f1={res['macro_f1']:.4f}")


# =====================================================================
# STEP 5: ENSEMBLE — built purely from saved probability arrays.
# No model is ever reloaded here, so this step cannot cause an OOM
# regardless of how many models were trained above.
# =====================================================================
print("\n" + "="*80)
print("STEP 5: ENSEMBLE METHODS")
print("="*80)

y_val_true = np.load(output_dir / 'y_val_true.npy')
y_test_true = np.load(output_dir / 'y_test_true.npy')

available_models = [m for m in model_list if (output_dir / f'{m}_val_probs.npy').exists()]

if len(available_models) >= 2:
    val_probs_list = [np.load(output_dir / f'{m}_val_probs.npy') for m in available_models]
    test_probs_list = [np.load(output_dir / f'{m}_test_probs.npy') for m in available_models]

    ensemble_val_probs = np.mean(val_probs_list, axis=0)   # soft voting
    ensemble_test_probs = np.mean(test_probs_list, axis=0)

    y_val_pred_ens = np.argmax(ensemble_val_probs, axis=1)
    y_test_pred_ens = np.argmax(ensemble_test_probs, axis=1)

    ens_val_acc = float((y_val_pred_ens == y_val_true).mean())
    ens_test_acc = float((y_test_pred_ens == y_test_true).mean())
    ens_report = classification_report(y_test_true, y_test_pred_ens,
                                        target_names=class_names,
                                        output_dict=True, zero_division=0)

    print(f"✓ Ensemble ({' + '.join(available_models)})")
    print(f"   Val accuracy:  {ens_val_acc:.4f}")
    print(f"   Test accuracy: {ens_test_acc:.4f}")

    cm_ens = confusion_matrix(y_test_true, y_test_pred_ens)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_ens, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix — Ensemble (soft voting)')
    plt.ylabel('True'); plt.xlabel('Predicted')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'ensemble_confusion_matrix.png', dpi=200, bbox_inches='tight')
    plt.close()

    results_summary['ensemble'] = {
        'val_accuracy': ens_val_acc,
        'test_accuracy': ens_test_acc,
        'macro_f1': ens_report['macro avg']['f1-score'],
        'models_used': available_models,
    }
    print("✓ Ensemble evaluation complete!")
else:
    print("⚠ Fewer than 2 successfully trained models — skipping ensemble.")


# =====================================================================
# STEP 6: FINAL TEST SET EVALUATION
# =====================================================================
print("\n" + "="*80)
print("STEP 6: FINAL TEST SET EVALUATION")
print("="*80)

candidates = {k: v for k, v in results_summary.items() if 'val_accuracy' in v}

if candidates:
    best_name = max(candidates.items(), key=lambda kv: kv[1]['val_accuracy'])[0]
    best_info = candidates[best_name]
    print(f"\n🏆 Best model (by validation accuracy): {best_name}")
    print(f"📊 Test accuracy: {best_info['test_accuracy']:.4f}")

    if best_name == 'ensemble':
        y_test_pred_best = y_test_pred_ens
    else:
        best_test_probs = np.load(output_dir / f'{best_name}_test_probs.npy')
        y_test_pred_best = np.argmax(best_test_probs, axis=1)

    print(f"\n{classification_report(y_test_true, y_test_pred_best, target_names=class_names, zero_division=0)}")

    cm_test = confusion_matrix(y_test_true, y_test_pred_best)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Test Confusion Matrix — {best_name}')
    plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'test_confusion_matrix_best.png', dpi=300, bbox_inches='tight')
    plt.close()
else:
    print("\n⚠ No models were successfully trained.")
    best_name = "None"
    best_info = {'test_accuracy': 0.0}


# =====================================================================
# STEP 7: MODEL COMPARISON
# =====================================================================
print("\n" + "="*80)
print("STEP 7: MODEL COMPARISON")
print("="*80)

import pandas as pd

comparison_df = pd.DataFrame(results_summary).T
keep_cols = [c for c in ['val_accuracy', 'test_accuracy', 'macro_f1'] if c in comparison_df.columns]
comparison_df = comparison_df[keep_cols].sort_values('test_accuracy', ascending=False)
print(comparison_df)
comparison_df.to_csv(output_dir / 'model_comparison.csv')
print(f"\n✓ Saved: {output_dir / 'model_comparison.csv'}")


# =====================================================================
# STEP 8: SAVING RESULTS
# =====================================================================
print("\n" + "="*80)
print("STEP 8: SAVING RESULTS")
print("="*80)

import shutil
from datetime import datetime

if candidates and best_name != 'ensemble':
    src_ckpt = output_dir / f'{best_name}_best.keras'
    if src_ckpt.exists():
        dst = output_dir / f'final_best_model_{best_name}.keras'
        shutil.copy(src_ckpt, dst)
        print(f"✓ Saved: {dst.name}")
elif best_name == 'ensemble':
    print("✓ Best result is the ensemble — its component checkpoints are already "
          f"saved individually ({', '.join(available_models)}); combine their "
          "predictions (soft-vote average) at inference time.")

final_summary = {
    'best_model': best_name,
    'test_accuracy': float(best_info['test_accuracy']) if candidates else 0.0,
    'configuration': CONFIG,
    'class_names': class_names,
    'num_classes': num_classes,
    'all_results': results_summary,
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
}
with open(output_dir / 'results_summary.json', 'w') as f:
    json.dump(final_summary, f, indent=4)
print("✓ Saved: results_summary.json")

print("\n" + "="*80)
print("✅ PIPELINE COMPLETE!")
print("="*80)
if candidates:
    print(f"\n🏆 Best Model: {best_name}")
    print(f"📊 Test Accuracy: {best_info['test_accuracy']:.4f}")
    print(f"\n📁 All results saved in: {output_dir.absolute()}")
else:
    print("\n⚠ No models were successfully trained. Check errors above.")
