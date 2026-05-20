import os
import json
import pickle
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA

import sys
HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
if os.path.join(BASE_DIR, "src", "learning") not in sys.path:
    sys.path.append(os.path.join(BASE_DIR, "src", "learning"))
from loss import FocalLoss

BG, PANEL, TEXT, GRID = "#0D1117", "#161B22", "#E6EDF3", "#30363D"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PyTorchMLPClassifier(nn.Module):
    def _make_act(self, activation):
        if activation.lower() == "tanh":
            return nn.Tanh()
        elif activation.lower() == "logistic":
            return nn.Sigmoid()
        return nn.ReLU()

    def __init__(self, input_size, hidden_layers, num_classes=2, activation="relu", dropout=0.3):
        super().__init__()
        self.activation_name = activation
        layers = []
        in_features = input_size

        for h in hidden_layers:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(self._make_act(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = h

        layers.append(nn.Linear(in_features, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def _plot_confusion(cm, classes, ts, name_prefix, result_dir):
    fig, ax = plt.subplots(figsize=(5, 4), facecolor=BG)
    ax.set_facecolor(PANEL); ax.spines[:].set_color(GRID)
    ax.tick_params(colors=TEXT)

    im = ax.imshow(cm, cmap="Blues", vmin=0)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, color=TEXT)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, color=TEXT, rotation=90, va="center")
    ax.set_xlabel("Predicted", color=TEXT); ax.set_ylabel("True", color=TEXT)
    ax.set_title(f"{name_prefix.upper()}-MLP Confusion Matrix", color=TEXT, fontsize=12, pad=8)

    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)
    for i in range(len(classes)):
        for j in range(len(classes)):
            tc = "white" if cm_norm[i, j] > 0.5 else TEXT
            ax.text(j, i - 0.15, f"{cm_norm[i,j]*100:.1f}%", ha="center", va="center", fontsize=12, color=tc, fontweight="bold")
            ax.text(j, i + 0.2,  f"({cm[i,j]})", ha="center", va="center", fontsize=9,  color=tc)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.ax.tick_params(colors=TEXT, labelsize=8)
    plt.tight_layout()

    out = os.path.join(result_dir, f"{name_prefix}_confusion_{ts}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)

def train_and_evaluate_pytorch(X, y, subjects, args, name_prefix, feature_desc, meta_extras=None, use_pca=False):
    os.makedirs(args.model_dir,  exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = le.classes_

    print(f"\n[Train] Classes: {classes}  |  Total: {len(X)}  |  Feature dim: {X.shape[1]}")
    for cls in classes:
        print(f"  {cls}: {(y == cls).sum()} samples")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
    
    n_pca_actual = min(getattr(args, "n_pca", 32), X_tr.shape[0], X_tr.shape[1]) if use_pca else None

    # Pipeline components
    scaler = StandardScaler()
    pca = PCA(n_components=n_pca_actual, random_state=42) if use_pca else None

    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_te)
    if use_pca:
        X_tr_scaled = pca.fit_transform(X_tr_scaled)
        X_te_scaled = pca.transform(X_te_scaled)

    input_size = X_tr_scaled.shape[1]
    
    model = PyTorchMLPClassifier(
        input_size=input_size, 
        hidden_layers=args.hidden, 
        num_classes=len(classes),
        activation=args.activation,
        dropout=0.3
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    
    use_focal = getattr(args, "use_focal_loss", False)
    if use_focal:
        gamma = getattr(args, "focal_gamma", 2.0)
        alpha_val = getattr(args, "focal_alpha", [2.5, 1.0])
        alpha_tensor = torch.tensor(alpha_val, dtype=torch.float32).to(device) if alpha_val else None
        criterion = FocalLoss(gamma=gamma, alpha=alpha_tensor)
    else:
        criterion = nn.CrossEntropyLoss()

    batch_size = 64
    epochs = args.epochs
    best_loss = float("inf")
    patience = 20
    no_improve_count = 0

    print(f"\n[Train] PyTorch MLP Training ({name_prefix})")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        # Shuffle
        idx = np.random.permutation(len(X_tr_scaled))
        X_tr_s, y_tr_s = X_tr_scaled[idx], y_tr[idx]
        
        for i in range(0, len(X_tr_s), batch_size):
            X_b = torch.tensor(X_tr_s[i:i+batch_size], dtype=torch.float32).to(device)
            y_b = torch.tensor(y_tr_s[i:i+batch_size], dtype=torch.long).to(device)

            optimizer.zero_grad()
            logits = model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y_b)

        train_loss /= len(X_tr_s)

        model.eval()
        with torch.no_grad():
            val_logits = model(torch.tensor(X_tr_scaled, dtype=torch.float32).to(device))
            val_loss = criterion(val_logits, torch.tensor(y_tr, dtype=torch.long).to(device)).item()

        scheduler.step(val_loss)
        
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Eval
    model.eval()
    with torch.no_grad():
        te_logits = model(torch.tensor(X_te_scaled, dtype=torch.float32).to(device))
        y_pred = te_logits.argmax(dim=1).cpu().numpy()

    acc = accuracy_score(y_te, y_pred)
    report = classification_report(y_te, y_pred, target_names=classes)
    cm = confusion_matrix(y_te, y_pred)

    print(f"\n[Result] Test Accuracy: {acc:.4f}")
    print(report)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    overwrite = getattr(args, "overwrite", True)
    if overwrite:
        ts_tag = ""
    else:
        ts_tag = f"_{ts}"

    report_path = os.path.join(args.result_dir, f"{name_prefix}_mlp_report{ts_tag}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"{name_prefix.upper()}-MLP Classification Report (PyTorch)\n")
        f.write(f"Timestamp : {ts}\n")
        f.write(feature_desc + "\n")
        f.write(f"Hidden    : {args.hidden}  Activation={args.activation}\n")
        f.write(f"Epochs    : {args.epochs}  LR={args.lr}\n")
        if use_focal:
            f.write(f"Loss      : FocalLoss (gamma={getattr(args, 'focal_gamma', 2.0)}, alpha={getattr(args, 'focal_alpha', [2.5, 1.0])})\n")
        else:
            f.write(f"Loss      : CrossEntropyLoss\n")
        f.write(f"Train/Test: {len(X_tr)} / {len(X_te)}\n\n")
        f.write(f"Test Accuracy: {acc:.4f}\n\n")
        f.write(report)
        f.write(f"\nConfusion Matrix:\n{cm}\n")

    _plot_confusion(cm, classes, ts if not overwrite else "", name_prefix, args.result_dir)

    print("\n[CV] 5-Fold Cross Validation...")
    cv_accs = []
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y_enc)):
        X_tr_f, X_val_f = X[tr_idx], X[val_idx]
        y_tr_f, y_val_f = y_enc[tr_idx], y_enc[val_idx]

        sc_f = StandardScaler()
        pc_f = PCA(n_components=n_pca_actual, random_state=42) if use_pca else None

        X_tr_s = sc_f.fit_transform(X_tr_f)
        X_val_s = sc_f.transform(X_val_f)
        if use_pca:
            X_tr_s = pc_f.fit_transform(X_tr_s)
            X_val_s = pc_f.transform(X_val_s)

        m_f = PyTorchMLPClassifier(
            input_size=X_tr_s.shape[1], hidden_layers=args.hidden, 
            num_classes=len(classes), activation=args.activation
        ).to(device)
        
        opt_f = optim.Adam(m_f.parameters(), lr=args.lr, weight_decay=1e-4)

        for epoch in range(args.epochs):
            m_f.train()
            idx = np.random.permutation(len(X_tr_s))
            X_t, y_t = X_tr_s[idx], y_tr_f[idx]
            for i in range(0, len(X_t), batch_size):
                X_b = torch.tensor(X_t[i:i+batch_size], dtype=torch.float32).to(device)
                y_b = torch.tensor(y_t[i:i+batch_size], dtype=torch.long).to(device)
                opt_f.zero_grad()
                loss = criterion(m_f(X_b), y_b)
                loss.backward()
                opt_f.step()

        m_f.eval()
        with torch.no_grad():
            v_logits = m_f(torch.tensor(X_val_s, dtype=torch.float32).to(device))
            v_pred = v_logits.argmax(dim=1).cpu().numpy()

        fold_acc = accuracy_score(y_val_f, v_pred)
        cv_accs.append(fold_acc)
        print(f"  Fold {fold+1}: {fold_acc:.4f}")
        
    print(f"  CV Mean±Std: {np.mean(cv_accs):.4f} ± {np.std(cv_accs):.4f}")

    with open(report_path, "a", encoding="utf-8") as f:
        f.write(f"\n5-Fold CV: {[round(a,4) for a in cv_accs]}\n")
        f.write(f"CV Mean: {np.mean(cv_accs):.4f}  CV Std: {np.std(cv_accs):.4f}\n")

    # Save
    model_path = os.path.join(args.model_dir, f"{name_prefix}_mlp.pth")
    le_path    = os.path.join(args.model_dir, f"{name_prefix}_label_encoder.pkl")
    scaler_path= os.path.join(args.model_dir, f"{name_prefix}_scaler.pkl")
    pca_path   = os.path.join(args.model_dir, f"{name_prefix}_pca.pkl")
    meta_path  = os.path.join(args.model_dir, f"{name_prefix}_mlp_meta.json")

    torch.save(model.state_dict(), model_path)
    with open(le_path, "wb") as f: pickle.dump(le, f)
    with open(scaler_path, "wb") as f: pickle.dump(scaler, f)
    if use_pca:
        with open(pca_path, "wb") as f: pickle.dump(pca, f)

    meta = {
        "hidden": args.hidden,
        "activation": args.activation,
        "use_focal_loss": getattr(args, "use_focal_loss", False),
        "focal_gamma": getattr(args, "focal_gamma", 2.0),
        "focal_alpha": getattr(args, "focal_alpha", [2.5, 1.0]),
        "classes": list(classes),
        "input_size": input_size,
        "test_accuracy": float(acc),
        "cv_mean": float(np.mean(cv_accs)),
        "cv_std":  float(np.std(cv_accs)),
        "timestamp": ts,
    }
    if meta_extras:
        meta.update(meta_extras)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)

    print(f"[Model] Saved → {model_path}")
    print(f"[Model] Meta  → {meta_path}")
    
    return model, le, scaler, pca

def run_inference_pytorch(args, name_prefix, extract_func, targets, use_pca=False):
    model_path = args.model_path or os.path.join(args.model_dir, f"{name_prefix}_mlp.pth")
    le_path    = os.path.join(args.model_dir, f"{name_prefix}_label_encoder.pkl")
    scaler_path= os.path.join(args.model_dir, f"{name_prefix}_scaler.pkl")
    pca_path   = os.path.join(args.model_dir, f"{name_prefix}_pca.pkl")
    meta_path  = os.path.join(args.model_dir, f"{name_prefix}_mlp_meta.json")

    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}"); sys.exit(1)

    with open(le_path, "rb") as f: le = pickle.load(f)
    with open(scaler_path, "rb") as f: scaler = pickle.load(f)
    pca = None
    if use_pca and os.path.exists(pca_path):
        with open(pca_path, "rb") as f: pca = pickle.load(f)
    with open(meta_path, "r") as f: meta = json.load(f)

    model = PyTorchMLPClassifier(
        input_size=meta["input_size"],
        hidden_layers=meta["hidden"],
        num_classes=len(meta["classes"]),
        activation=meta["activation"]
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"\n{'─'*68}")
    print(f"{'File':<40} {'True':>6} {'Pred':>6} {'Conf':>8}")
    print(f"{'─'*68}")

    correct = 0
    valid = 0
    for fp in targets:
        # Extract true label from fp using _parse_label logic
        stem = os.path.splitext(os.path.basename(fp))[0].lower()
        parts = stem.split("_")
        lraw = parts[-1]
        true_label = "big" if "big" in lraw else ("small" if "smal" in lraw else None)
        
        try:
            feat = extract_func(fp, args, meta)
            if feat is None:
                continue
            
            # Predict
            X_s = scaler.transform(feat.reshape(1, -1))
            if pca:
                X_s = pca.transform(X_s)
                
            with torch.no_grad():
                logits = model(torch.tensor(X_s, dtype=torch.float32).to(device))
                prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
                
            pred = le.inverse_transform([prob.argmax()])[0]
            conf = prob.max()
            
            if true_label:
                valid += 1
                match = "✓" if pred == true_label else "✗"
                if pred == true_label: correct += 1
            else:
                match = "?"
                
            print(f"  {os.path.basename(fp)[:36]:<40} {true_label or '?':>6} {pred:>6} {conf:>8.3f} {match}")
        except Exception as e:
            print(f"  {os.path.basename(fp)[:36]:<40} ERR: {e}")

    if valid > 0:
        print(f"{'─'*68}")
        print(f"Accuracy: {correct}/{valid} = {correct/valid:.4f}")

