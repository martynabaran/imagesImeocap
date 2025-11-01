# -----------------------------
# 1. Imports
# -----------------------------
import os
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchaudio
from torchaudio import transforms as T
from torch.hub import load as torch_hub_load
#from datasets import Dataset
from sklearn.preprocessing import label_binarize

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score
)

from sklearn.model_selection import train_test_split

from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from torchvision.models import resnet18

# =========================================================
# 2. PLGRID SCRATCH PATHS
# =========================================================
# Jeśli jest MEMFS (RAM disk), użyj go; inaczej standardowy SCRATCH
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SCRATCH = os.environ.get("MEMFS", os.environ.get("SCRATCH", "/tmp"))
HF_CACHE = os.path.join(SCRATCH, "huggingface_cache")
DATASET_DIR2 = os.path.join("/net/tscratch/people/plgmarbar/", "iemocap")
# Ścieżka do folderu z danymi (rozpakowany ZIP)
IEMOCAP_DIR = os.path.join("/net/tscratch/people/plgmarbar/iemocap", "imeocap_data/imepocap_simplified")
CSV_PATH = os.path.join("/net/tscratch/people/plgmarbar/iemocap/imeocap_data/", "metadata.csv")
OUTPUT_DIR = os.path.join("/net/tscratch/people/plgmarbar/iemocap", "image_approach_checkpoints")
OUTPUT_DIR_PERSISTENT=OUTPUT_DIR
os.makedirs(OUTPUT_DIR_PERSISTENT, exist_ok=True)

os.makedirs(HF_CACHE, exist_ok=True)
# os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set Hugging Face cache
os.environ["HF_HOME"] = HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE

print("[INFO] Scratch directories configured:")
print(f"  SCRATCH:       {SCRATCH}")
# print(f"  DATASET_DIR:   {DATASET_DIR}")
print(f"  HF_CACHE:      {HF_CACHE}")
print(f"  OUTPUT_DIR:    {OUTPUT_DIR}")

# -----------------------------
# 3. Helper Functions
# -----------------------------

def prepare_dataset_paths(df, base_dir):

    df = df.copy()

    def make_new_path(x):
        # zamień backslash na slash, żeby łatwiej dzielić
        x = x.replace("\\", "/")
        parts = x.split("/")  # podziel ścieżkę po slashach
        base_name = parts[-2] # ostatni folder
        file_name = parts[-1] # nazwa pliku
        new_path = os.path.join(base_dir, base_name, file_name)
        return new_path.replace("\\", "/")


    df["full_path"] = df["audio"].apply(make_new_path)
    df = df.rename(columns={"full_path": "path","label_id": "label"})
    return df

def set_reproducibility(seed: int = 42):
    """Ensure reproducibility for torch and numpy."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def compute_ser_metrics(y_true, y_pred, y_score, labels=None):
    """
    Compute required SER metrics.

    Args:
        y_true: 1D array-like of true integer/string labels
        y_pred: 1D array-like of predicted labels (same type as y_true)
        y_score: 2D array-like of shape (n_samples, n_classes) of predicted probabilities / scores.
                 Column order must match `labels`.
        labels: list of label names in the column order used in y_score. If None, inferred from y_true+y_pred sorted.

    Returns:
        metrics: dict with entries:
          - accuracy, balanced_accuracy
          - precision/recall/f1 (macro, micro, weighted)
          - per_class: dict per label with precision/recall/f1/support, specificity, AP, AUC (if computable)
          - mAP (macro-average AP), AUC_macro (macro-avg OVR), mcc
          - confusion_matrix (as pd.DataFrame)
    """


    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    n_classes = len(labels)
    # --- Normalize labels ---
    def normalize_label(l):
        # If numeric label, convert to name if possible
        if isinstance(l, (int, np.integer)):
            if l < len(labels):
                return labels[int(l)]
            else:
                return str(l)
        # Else assume string
        return str(l)

    y_true = [normalize_label(l) for l in y_true]
    y_pred = [normalize_label(l) for l in y_pred]

    # Convert normalized names back to indices for numeric metrics
    y_true_idx = np.array([label_to_idx[l] for l in y_true])
    y_pred_idx = np.array([label_to_idx[l] for l in y_pred])

    # Scores: ensure shape (N, n_classes)
    if y_score is None:
        # fallback: create one-hot from predictions (not ideal for AUC/AP but still produce other metrics)
        y_score = np.zeros((len(y_pred_idx), n_classes), dtype=float)
        y_score[np.arange(len(y_pred_idx)), y_pred_idx] = 1.0
    else:
        y_score = np.asarray(y_score)
        if y_score.ndim != 2 or y_score.shape[1] != n_classes:
            raise ValueError(f"y_score must be shape (N, {n_classes}); got {y_score.shape}")

    # Primary metrics
    acc = float(accuracy_score(y_true_idx, y_pred_idx))
    bal_acc = float(balanced_accuracy_score(y_true_idx, y_pred_idx))

    # Precision/Recall/F1 (per-class & averages)
    p_r_f_support = precision_recall_fscore_support(y_true_idx, y_pred_idx, labels=range(n_classes), zero_division=0)
    precision_per = p_r_f_support[0]
    recall_per = p_r_f_support[1]
    f1_per = p_r_f_support[2]
    support_per = p_r_f_support[3]

    # Averages
    precision_macro = float(np.mean(precision_per))
    recall_macro = float(np.mean(recall_per))
    f1_macro = float(np.mean(f1_per))

    p_r_f_micro = precision_recall_fscore_support(y_true_idx, y_pred_idx, average='micro', zero_division=0)
    precision_micro, recall_micro, f1_micro = map(float, p_r_f_micro[:3])
    p_r_f_weighted = precision_recall_fscore_support(y_true_idx, y_pred_idx, average='weighted', zero_division=0)
    precision_weighted, recall_weighted, f1_weighted = map(float, p_r_f_weighted[:3])

    # Confusion matrix and specificity per class
    cm = confusion_matrix(y_true_idx, y_pred_idx, labels=range(n_classes))
    specificity_per = []
    per_class = {}
    for i, lab in enumerate(labels):
        TP = int(cm[i, i])
        FN = int(cm[i, :].sum() - TP)
        FP = int(cm[:, i].sum() - TP)
        TN = int(cm.sum() - (TP + FP + FN))
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        specificity_per.append(float(specificity))

        per_class[lab] = {
            'precision': float(precision_per[i]),
            'recall': float(recall_per[i]),
            'f1': float(f1_per[i]),
            'support': int(support_per[i]),
            'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN,
            'specificity': float(specificity),
            'AP': None,
            'AUC': None
        }

    # MCC
    try:
        mcc = float(matthews_corrcoef(y_true_idx, y_pred_idx))
    except Exception:
        mcc = None

    # AUC (macro OVR) and per-class AUC/AP where possible
    # Binarize true labels
    y_true_bin = label_binarize(y_true_idx, classes=list(range(n_classes)))
    # sklearn's roc_auc_score requires at least one positive label for each class to compute AUC
    aucs = []
    aps = []
    for i in range(n_classes):
        y_true_i = y_true_bin[:, i]
        y_score_i = y_score[:, i]
        # AUC
        try:
            auc_i = roc_auc_score(y_true_i, y_score_i)
        except Exception:
            auc_i = None
        # Average Precision
        try:
            ap_i = average_precision_score(y_true_i, y_score_i)
        except Exception:
            ap_i = None
        per_class[labels[i]]['AUC'] = float(auc_i) if auc_i is not None else None
        per_class[labels[i]]['AP'] = float(ap_i) if ap_i is not None else None
        if auc_i is not None:
            aucs.append(auc_i)
        if ap_i is not None:
            aps.append(ap_i)

    auc_macro = float(np.mean(aucs)) if len(aucs) > 0 else None
    mAP_macro = float(np.mean(aps)) if len(aps) > 0 else None

    # Build metric dict
    metrics = {
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'mcc': mcc,
        'auc_macro': auc_macro,
        'mAP_macro': mAP_macro,
        'per_class': per_class,
        'specificity_per_class': dict(zip(labels, specificity_per)),
        'confusion_matrix': {
            'labels': labels,
            'matrix': cm.tolist()
        }
    }
    return metrics

# Checkpointing helper (pseudo code-style saver to integrate into training loop)
def save_checkpoint_if_best(model, optimizer, epoch, metrics, checkpoint_dir, primary_metric_name='balanced_accuracy', maximize=True):
    """
    Save model checkpoint when primary metric improves.

    - metrics: dict returned from compute_ser_metrics on validation set.
    - checkpoint_dir: directory to save checkpoints
    - primary_metric_name: string key in metrics to use for selecting the best checkpoint (default 'balanced_accuracy')
    - maximize: True if higher is better (True for accuracy/balanced_accuracy), False if lower is better.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_meta_path = os.path.join(checkpoint_dir, 'best_checkpoint_meta.json')
    # load existing best
    if os.path.exists(best_meta_path):
        with open(best_meta_path, 'r') as f:
            best_meta = json.load(f)
    else:
        best_meta = {'best_value': None, 'best_epoch': None, 'best_file': None}

    cur_value = metrics.get(primary_metric_name, None)
    is_better = False
    if cur_value is None:
        is_better = False
    else:
        if best_meta['best_value'] is None:
            is_better = True
        else:
            if maximize:
                is_better = cur_value > best_meta['best_value']
            else:
                is_better = cur_value < best_meta['best_value']

    if is_better:
        # save model - adapt this to your framework (torch.save, tf.keras.Model.save, etc.)
        filename = f"checkpoint_epoch{epoch}_{primary_metric_name}_{cur_value:.6f}.pt"
        filepath = os.path.join(checkpoint_dir, filename)
        # Example for PyTorch:
        try:
            import torch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
                'metrics': metrics
            }, filepath)
        except Exception:
            # fallback: if not PyTorch, try model.save() for TF or other saving logic
            try:
                model.save(filepath)
            except Exception as e:
                # last resort: save only metrics
                with open(filepath + '.metrics.json', 'w') as f:
                    json.dump(metrics, f, indent=2)

        # update best meta
        best_meta['best_value'] = float(cur_value)
        best_meta['best_epoch'] = int(epoch)
        best_meta['best_file'] = filename
        with open(best_meta_path, 'w') as f:
            json.dump(best_meta, f, indent=2)

        # also save human-readable metrics snapshot
        metrics_snapshot_path = os.path.join(checkpoint_dir, f"metrics_epoch{epoch}.json")
        with open(metrics_snapshot_path, 'w') as f:
            json.dump(metrics, f, indent=2)

    return is_better

# Quick helper to persist final metrics into CSV for easy spreadsheet import
def metrics_to_dataframe(metrics):
    """
    Flattens metrics dict into a pandas DataFrame (one row per class + global row).
    """
    rows = []
    # global row
    global_row = {
        'label': 'GLOBAL',
        'accuracy': metrics.get('accuracy'),
        'balanced_accuracy': metrics.get('balanced_accuracy'),
        'precision_macro': metrics.get('precision_macro'),
        'recall_macro': metrics.get('recall_macro'),
        'f1_macro': metrics.get('f1_macro'),
        'precision_micro': metrics.get('precision_micro'),
        'recall_micro': metrics.get('recall_micro'),
        'f1_micro': metrics.get('f1_micro'),
        'mcc': metrics.get('mcc'),
        'auc_macro': metrics.get('auc_macro'),
        'mAP_macro': metrics.get('mAP_macro'),
    }
    rows.append(global_row)
    for lab, per in metrics['per_class'].items():
        row = {'label': lab}
        row.update({
            'precision': per['precision'],
            'recall': per['recall'],
            'f1': per['f1'],
            'support': per['support'],
            'specificity': per['specificity'],
            'AP': per['AP'],
            'AUC': per['AUC'],
            'TP': per['TP'], 'FP': per['FP'], 'TN': per['TN'], 'FN': per['FN']
        })
        rows.append(row)
    df = pd.DataFrame(rows)
    return df

print(f"[INFO] Using local dataset from: {IEMOCAP_DIR}")
print(f"[INFO] Loading metadata from: {CSV_PATH}")
IEMOCAP_DIR2 = "/net/tscratch/people/plgmarbar/iemocap/imeocap_data/imepocap_simplified"
# Wczytaj DataFrame
df = pd.read_csv(CSV_PATH)
df = prepare_dataset_paths(df, IEMOCAP_DIR2)
print(f"[INFO] Loaded {len(df)} samples")



# -----------------------
# 1. Dataset and Transforms
# -----------------------

class SERDataset(Dataset):
    def __init__(self, df, label2id, augment=False, sample_rate=16000, n_mels=128,
                 max_duration=5.0, mode="image"):  # NEW: mode argument
        self.df = df.reset_index(drop=True)
        self.label2id = label2id
        self.augment = augment
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.max_samples = int(sample_rate * max_duration)
        self.mode = mode  # "image" or "audio"

        # Audio transforms
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=n_mels, n_fft=1024, hop_length=512
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)

        # Audio augmentations
        self.noise_augment = lambda x: x + 0.005 * torch.randn_like(x)
        self.pitch_shift = torchaudio.transforms.PitchShift(sample_rate, n_steps=random.choice([-2, -1, 1, 2]))

        # Image augmentations
        self.img_transforms = T.Compose([
            T.RandomApply([T.RandomAffine(degrees=5, translate=(0.05, 0.05))], p=0.5),
            T.RandomHorizontalFlip(p=0.5)
        ])

        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=20)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=10)
        
    def __len__(self):
        return len(self.df)

    def _load_audio(self, path):
        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        if waveform.size(1) > self.max_samples:
            waveform = waveform[:, :self.max_samples]
        else:
            pad_len = self.max_samples - waveform.size(1)
            waveform = F.pad(waveform, (0, pad_len))
        return waveform

    def _augment_audio(self, waveform):
      with torch.no_grad():  # ✅ Disable gradient tracking inside augmentations
          if random.random() < 0.3:
              waveform = self.noise_augment(waveform)
          if random.random() < 0.3:
            waveform = self.pitch_shift(waveform)
      return waveform.detach()  # ✅ Ensure tensor doesn’t require grad


    def _augment_spectrogram(self, mel_db):
        # Apply SpecAugment (time/frequency masking)
        if random.random() < 0.5:
            mel_db = self.time_mask(mel_db)
        if random.random() < 0.5:
            mel_db = self.freq_mask(mel_db)
        return mel_db
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        waveform = self._load_audio(row['path'])
        if self.augment:
            waveform = self._augment_audio(waveform)
        label = self.label2id[row['label']]

        if self.mode == "audio":
            # For AST: return waveform directly
            return waveform, label

        # Otherwise (image mode): convert to Mel spectrogram
        mel = self.mel_transform(waveform)
        mel_db = self.db_transform(mel)
        #mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-9)
        mel_db = (mel_db - mel_db.mean(dim=-1, keepdim=True)) / (mel_db.std(dim=-1, keepdim=True) + 1e-9)

        img = mel_db.repeat(3, 1, 1)
        if self.augment:
             img = self.img_transforms(img)
        return img, label
        #if self.augment:
        #    mel_db = self._augment_spectrogram(mel_db)

        # PANNs expect 3-channel input sometimes (for pretrained CNNs)
        #mel_db = mel_db.expand(3, -1, -1)  # [3, n_mels, time]

        #return mel_db, torch.tensor(label, dtype=torch.long)

# -----------------------
# 5. Models
# -----------------------

# -----------------------
# ResNet-based model
# -----------------------
class ResNetSpectrogramClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        #self.backbone = resnet18(pretrained=True)
        self.backbone = resnet18(weights="IMAGENET1K_V1")

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

# -----------------------
# 3. Training & Evaluation
# -----------------------

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    all_y_true, all_y_pred, all_y_score = [], [], []
    for imgs, labels in tqdm(loader, desc="Train"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        preds = np.argmax(probs, axis=1)
        all_y_true.extend(labels.cpu().numpy())
        all_y_pred.extend(preds)
        all_y_score.extend(probs)
    return total_loss / len(loader.dataset), all_y_true, all_y_pred, np.array(all_y_score)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    all_y_true, all_y_pred, all_y_score = [], [], []
    for imgs, labels in tqdm(loader, desc="Eval"):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = F.cross_entropy(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        all_y_true.extend(labels.cpu().numpy())
        all_y_pred.extend(preds)
        all_y_score.extend(probs)
    return total_loss / len(loader.dataset), all_y_true, all_y_pred, np.array(all_y_score)


def train_and_evaluate(model, train_loader, val_loader, optimizer, device, labels, checkpoint_dir, n_epochs=20):
    best_val = None
    for epoch in range(1, n_epochs + 1):
        print(f"\nEpoch {epoch}/{n_epochs}")
        train_loss, y_true_train, y_pred_train, y_score_train = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, y_true_val, y_pred_val, y_score_val = evaluate(model, val_loader, device)

        val_metrics = compute_ser_metrics(y_true_val, y_pred_val, y_score_val, labels)
        print(json.dumps({k: val_metrics[k] for k in ['accuracy', 'balanced_accuracy', 'f1_macro']}, indent=2))
        save_checkpoint_if_best(model, optimizer, epoch, val_metrics, checkpoint_dir)
    return model

# -----------------------
# 4. Main Experiment Setup
# -----------------------
def get_model(model_name, num_classes):
    model_name = model_name.lower()
    if model_name == "resnet18":
        print("[INFO] Initializing ResNet18...")
        return ResNetSpectrogramClassifier(num_classes)
    elif model_name == "panns_cnn14":
        print("[INFO] Initializing PANNs (Cnn14)...")
        return PANNsSpectrogramClassifier(num_classes,freeze_backbone=False )
    elif model_name == "ast":
        print("[INFO] Initializing Audio Spectrogram Transformer (AST)...")
        return ASTModelWrapper(num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def run_experiment(df, model_name, output_dir="checkpoints", batch_size=16, n_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #labels = sorted(df['label'].unique())
    #label2id = {l: i for i, l in enumerate(labels)}
    #id2label = {v: k for k, v in label2id.items()}
    df["label"] = df["label"].replace({9: 8})
    print("[INFO] Class distribution after merging:")
    print(df["label"].value_counts())

# =========================================================
# 3️⃣ Define emotion list (fixed label order)
# =========================================================
    emotions = ['neutral', 'frustrated', 'angry', 'sad', 'happy', 'excited', 'surprise', 'fear', 'other']

# Create mapping dictionaries
    label2id = {emotion: idx for idx, emotion in enumerate(emotions)}
    id2label = {idx: emotion for idx, emotion in enumerate(emotions)}
    labels = list(label2id.keys())

    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['label'], random_state=42)

    # Set dataset mode
    if model_name.lower() in ["resnet18", "panns_cnn14"]:
        dataset_mode = "image"
    elif model_name.lower() == "ast":
        dataset_mode = "audio"
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    print(f"[INFO] Preparing datasets for mode: {dataset_mode}")

    # Create datasets and loaders
    train_ds = SERDataset(train_df, label2id, augment=True, mode=dataset_mode)
    val_ds = SERDataset(val_df, label2id, augment=False, mode=dataset_mode)
    test_ds = SERDataset(test_df, label2id, augment=False, mode=dataset_mode)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Create model
    model = get_model(model_name, num_classes=len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Train
    print(f"\n===== Training {model_name.upper()} model =====")
    checkpoint_dir = os.path.join(output_dir, model_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    model = train_and_evaluate(model, train_loader, val_loader, optimizer, device, labels, checkpoint_dir, n_epochs=n_epochs)

    # Evaluate on test
    print(f"\n===== Evaluating {model_name.upper()} on Test Set =====")
    test_loss, y_true, y_pred, y_score = evaluate(model, test_loader, device)
    y_true_names = [id2label[int(i)] for i in y_true]
    y_pred_names = [id2label[int(i)] for i in y_pred]

    metrics = compute_ser_metrics(
        y_true_names, y_pred_names, y_score, labels=list(label2id.keys())
    )
    df_metrics = metrics_to_dataframe(metrics)
    df_metrics.to_csv(os.path.join(output_dir, f"{model_name}_test_metrics.csv"), index=False)
    with open(os.path.join(output_dir, f"{model_name}_test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps({k: metrics[k] for k in ['accuracy', 'balanced_accuracy', 'f1_macro']}, indent=2))

run_experiment(df,model_name="resnet18", output_dir=OUTPUT_DIR, batch_size=2, n_epochs=25)
