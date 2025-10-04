"""
EEG Weak Signal Detector - With Statistical Testing
Removes lab contamination + proper significance testing
"""

import os
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import threading
import queue
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import stats

try:
    from datasets import load_dataset
except ImportError as e:
    print(f"Missing dependency: {e}")
    exit()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EEG_SAMPLE_RATE = 512

# Exclude lab objects by default
TARGET_CATEGORIES = {
    # Animals - definitely not in lab
    'elephant': 22, 'giraffe': 25, 'bear': 23, 'zebra': 24,
    'cow': 21, 'sheep': 20, 'horse': 19, 'dog': 18, 'cat': 17,
    'bird': 16,
    
    # Vehicles
    'airplane': 5, 'train': 7, 'boat': 9, 'bus': 6, 'truck': 8,
    'motorcycle': 4, 'bicycle': 2, 'car': 3,
    
    # Outdoor objects
    'traffic light': 10, 'fire hydrant': 11, 'stop sign': 13,
    'parking meter': 14, 'bench': 15,
    
    # Sports equipment
    'frisbee': 34, 'skis': 35, 'snowboard': 36, 'sports ball': 37,
    'kite': 38, 'baseball bat': 39, 'skateboard': 41, 'surfboard': 42,
    
    # Food
    'banana': 52, 'apple': 53, 'orange': 55, 'broccoli': 56,
    'pizza': 59, 'donut': 60, 'cake': 61,
    
    # OPTIONAL - uncomment to include lab objects for comparison
    # 'person': 1, 'chair': 62, 'laptop': 73, 'book': 84,
}

CATEGORY_NAMES = {v: k for k, v in TARGET_CATEGORIES.items()}

def cohens_d(x, y):
    """Calculate Cohen's d effect size"""
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)

class FixedCNNClassifier(nn.Module):
    """CNN with configurable time window"""
    def __init__(self, n_channels=64, n_timepoints=154, num_classes=len(TARGET_CATEGORIES)):
        super().__init__()
        
        self.conv1 = nn.Conv1d(n_channels, 128, kernel_size=25, stride=1, padding=12)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(128, 256, kernel_size=15, stride=1, padding=7)
        self.bn2 = nn.BatchNorm1d(256)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(256, 512, kernel_size=7, stride=1, padding=3)
        self.bn3 = nn.BatchNorm1d(512)
        self.pool3 = nn.MaxPool1d(2)
        
        # Calculate feature size dynamically
        temp_size = n_timepoints
        for _ in range(3):
            temp_size = temp_size // 2
        self.feature_size = 512 * temp_size
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_size, 512),
            nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.pool1(F.elu(self.bn1(self.conv1(x))))
        x = self.pool2(F.elu(self.bn2(self.conv2(x))))
        x = self.pool3(F.elu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class MultiLabelEEGDataset(Dataset):
    def __init__(self, coco_path, annotations_path, split='train', max_samples=None,
                 start_ms=50, end_ms=350):
        self.coco_path = Path(coco_path)
        self.start_ms = start_ms
        self.end_ms = end_ms
        
        print(f"Loading Alljoined ({split}) with window {start_ms}-{end_ms}ms...")
        self.dataset = load_dataset("Alljoined/05_125", split=split, streaming=False)
        
        if max_samples:
            self.dataset = self.dataset.select(range(min(int(max_samples), len(self.dataset))))
        
        print(f"Loading COCO annotations...")
        with open(annotations_path, 'r') as f:
            coco_data = json.load(f)
        
        self.image_categories = defaultdict(set)
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            cat_id = ann['category_id']
            if cat_id in CATEGORY_NAMES:
                self.image_categories[img_id].add(cat_id)
        
        self.labels = []
        self.valid_indices = []
        
        for idx, sample in enumerate(self.dataset):
            coco_id = sample['coco_id']
            if coco_id in self.image_categories and len(self.image_categories[coco_id]) > 0:
                label = torch.zeros(len(TARGET_CATEGORIES))
                for cat_id in self.image_categories[coco_id]:
                    if cat_id in CATEGORY_NAMES:
                        cat_idx = list(TARGET_CATEGORIES.values()).index(cat_id)
                        label[cat_idx] = 1.0
                
                if label.sum() > 0:
                    self.labels.append(label)
                    self.valid_indices.append(idx)
        
        print(f"Valid samples: {len(self.valid_indices)}")
        
        # Statistics
        category_counts = defaultdict(int)
        for label in self.labels:
            for idx, present in enumerate(label):
                if present:
                    cat_id = list(TARGET_CATEGORIES.values())[idx]
                    category_counts[CATEGORY_NAMES[cat_id]] += 1
        
        print(f"\nCategory counts (total {len(category_counts)}):")
        for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {cat}: {count}")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        sample = self.dataset[real_idx]
        
        eeg_data = np.array(sample['EEG'], dtype=np.float32)
        
        start_idx = int((self.start_ms / 1000) * EEG_SAMPLE_RATE)
        end_idx = int((self.end_ms / 1000) * EEG_SAMPLE_RATE)
        
        if eeg_data.shape[1] >= end_idx:
            eeg_window = eeg_data[:, start_idx:end_idx]
        else:
            eeg_window = eeg_data[:, start_idx:]
        
        eeg_window = (eeg_window - eeg_window.mean(axis=1, keepdims=True)) / \
                     (eeg_window.std(axis=1, keepdims=True) + 1e-8)
        
        return torch.from_numpy(eeg_window).float(), self.labels[idx]

class StatisticalAnalyzerGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("EEG Statistical Signal Analyzer")
        self.geometry("1200x950")
        
        self.coco_path = ""
        self.annotations_path = ""
        self.model = None
        self.train_thread = None
        self.stop_flag = threading.Event()
        self.log_queue = queue.Queue()
        
        self.setup_gui()
        self.process_logs()
    
    def setup_gui(self):
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        train_tab = ttk.Frame(notebook)
        notebook.add(train_tab, text="Training")
        self.setup_train_tab(train_tab)
        
        analysis_tab = ttk.Frame(notebook)
        notebook.add(analysis_tab, text="Statistical Analysis")
        self.setup_analysis_tab(analysis_tab)
    
    def setup_train_tab(self, parent):
        title = tk.Label(parent, text="Clean Multi-Label Detection (No Lab Objects)", 
                        font=("Arial", 14, "bold"))
        title.pack(pady=5)
        
        info = tk.Label(parent, 
                       text="Excludes: person, chair, laptop, book (lab contamination)\n"
                            "Focuses: Animals, vehicles, outdoor objects only\n"
                            "Statistical testing with t-tests and effect sizes",
                       fg="blue", font=("Arial", 9))
        info.pack(pady=5)
        
        # Paths
        path_frame = ttk.LabelFrame(parent, text="Dataset")
        path_frame.pack(pady=5, padx=10, fill=tk.X)
        
        tk.Label(path_frame, text="COCO:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=3)
        self.coco_var = tk.StringVar()
        ttk.Entry(path_frame, textvariable=self.coco_var, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(path_frame, text="Browse", command=self.browse_coco).grid(row=0, column=2)
        
        tk.Label(path_frame, text="Annotations:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=3)
        self.ann_var = tk.StringVar()
        ttk.Entry(path_frame, textvariable=self.ann_var, width=50).grid(row=1, column=1, padx=5)
        ttk.Button(path_frame, text="Browse", command=self.browse_ann).grid(row=1, column=2)
        
        # Settings
        settings_frame = ttk.LabelFrame(parent, text="Configuration")
        settings_frame.pack(pady=5, padx=10, fill=tk.X)
        
        tk.Label(settings_frame, text="Max Samples:").grid(row=0, column=0, padx=5)
        self.max_var = tk.IntVar(value=3000)
        tk.Spinbox(settings_frame, from_=1000, to=10000, increment=1000,
                  textvariable=self.max_var, width=10).grid(row=0, column=1)
        
        tk.Label(settings_frame, text="Epochs:").grid(row=0, column=2, padx=5)
        self.epochs_var = tk.IntVar(value=30)
        tk.Spinbox(settings_frame, from_=10, to=50, increment=5,
                  textvariable=self.epochs_var, width=10).grid(row=0, column=3)
        
        tk.Label(settings_frame, text="Window Start (ms):").grid(row=1, column=0, padx=5)
        self.start_ms_var = tk.IntVar(value=50)
        tk.Spinbox(settings_frame, from_=0, to=500, increment=50,
                  textvariable=self.start_ms_var, width=10).grid(row=1, column=1)
        
        tk.Label(settings_frame, text="Window End (ms):").grid(row=1, column=2, padx=5)
        self.end_ms_var = tk.IntVar(value=350)
        tk.Spinbox(settings_frame, from_=100, to=600, increment=50,
                  textvariable=self.end_ms_var, width=10).grid(row=1, column=3)
        
        # Buttons
        btn_frame = tk.Frame(parent)
        btn_frame.pack(pady=5)
        
        self.train_btn = tk.Button(btn_frame, text="Train Model", 
                                   command=self.start_train,
                                   bg="#4CAF50", fg="white")
        self.train_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = tk.Button(btn_frame, text="Stop", 
                                  command=self.stop_train,
                                  bg="#f44336", fg="white",
                                  state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.analyze_btn = tk.Button(btn_frame, text="Statistical Analysis",
                                     command=self.analyze_signals,
                                     bg="#2196F3", fg="white")
        self.analyze_btn.pack(side=tk.LEFT, padx=5)
        
        # Progress
        self.progress = ttk.Progressbar(parent, mode='determinate')
        self.progress.pack(fill=tk.X, padx=10, pady=5)
        
        # Log
        log_frame = ttk.LabelFrame(parent, text="Training Log")
        log_frame.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)
        
        self.log_text = tk.Text(log_frame, height=12, bg='black', fg='lightgreen',
                               font=('Courier', 8))
        self.log_text.pack(fill=tk.BOTH, expand=True)
    
    def setup_analysis_tab(self, parent):
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.tight_layout(pad=3.0)
        
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def browse_coco(self):
        path = filedialog.askdirectory()
        if path:
            self.coco_var.set(path)
            self.coco_path = path
    
    def browse_ann(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if path:
            self.ann_var.set(path)
            self.annotations_path = path
    
    def log(self, msg):
        self.log_queue.put(msg)
    
    def process_logs(self):
        try:
            while not self.log_queue.empty():
                msg = self.log_queue.get_nowait()
                self.log_text.insert(tk.END, msg + "\n")
                self.log_text.see(tk.END)
        except queue.Empty:
            pass
        self.after(100, self.process_logs)
    
    def start_train(self):
        if not self.coco_path or not self.annotations_path:
            messagebox.showerror("Error", "Select paths first")
            return
        
        self.stop_flag.clear()
        self.train_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        self.train_thread = threading.Thread(target=self._train_loop, daemon=True)
        self.train_thread.start()
    
    def stop_train(self):
        self.stop_flag.set()
    
    def _train_loop(self):
        try:
            start_ms = self.start_ms_var.get()
            end_ms = self.end_ms_var.get()
            
            self.log("="*70)
            self.log("CLEAN SIGNAL DETECTION (NO LAB OBJECTS)")
            self.log(f"Device: {DEVICE}")
            self.log(f"Categories: {len(TARGET_CATEGORIES)}")
            self.log(f"Time window: {start_ms}-{end_ms}ms")
            self.log("="*70)
            
            # Calculate timepoints
            n_timepoints = int(((end_ms - start_ms) / 1000) * EEG_SAMPLE_RATE)
            
            self.model = FixedCNNClassifier(n_timepoints=n_timepoints, 
                                           num_classes=len(TARGET_CATEGORIES)).to(DEVICE)
            params = sum(p.numel() for p in self.model.parameters())
            self.log(f"Parameters: {params:,}")
            
            dataset = MultiLabelEEGDataset(
                self.coco_path,
                self.annotations_path,
                'train',
                int(self.max_var.get() * 1.25),
                start_ms=start_ms,
                end_ms=end_ms
            )
            
            total = len(dataset)
            train_size = int(0.8 * total)
            val_size = total - train_size
            
            train_set, val_set = torch.utils.data.random_split(
                dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
            
            self.log(f"Train: {train_size}, Val: {val_size}")
            
            train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=0)
            
            optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=0.01)
            criterion = nn.BCEWithLogitsLoss()
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs_var.get())
            
            best_val_loss = float('inf')
            
            for epoch in range(self.epochs_var.get()):
                if self.stop_flag.is_set():
                    break
                
                self.model.train()
                train_loss = 0
                
                for eeg, labels in train_loader:
                    if self.stop_flag.is_set():
                        break
                    
                    eeg, labels = eeg.to(DEVICE), labels.to(DEVICE)
                    
                    optimizer.zero_grad()
                    logits = self.model(eeg)
                    loss = criterion(logits, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                self.model.eval()
                val_loss = 0
                
                with torch.no_grad():
                    for eeg, labels in val_loader:
                        eeg, labels = eeg.to(DEVICE), labels.to(DEVICE)
                        logits = self.model(eeg)
                        loss = criterion(logits, labels)
                        val_loss += loss.item()
                
                train_loss /= len(train_loader)
                val_loss /= len(val_loader)
                
                scheduler.step()
                
                self.progress['value'] = ((epoch + 1) / self.epochs_var.get()) * 100
                
                self.log(f"Epoch {epoch+1}/{self.epochs_var.get()}: "
                        f"TrLoss={train_loss:.4f} ValLoss={val_loss:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'val_loss': val_loss,
                        'start_ms': start_ms,
                        'end_ms': end_ms,
                    }, 'clean_signal_detector.pth')
                    self.log(f"  -> Saved (ValLoss: {val_loss:.4f})")
            
            self.log("\n" + "="*70)
            self.log(f"COMPLETE - Best: {best_val_loss:.4f}")
            self.log("="*70)
            
        except Exception as e:
            self.log(f"ERROR: {e}")
            import traceback
            self.log(traceback.format_exc())
        finally:
            self.train_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
    
    def analyze_signals(self):
        if self.model is None:
            messagebox.showerror("Error", "Train model first")
            return
        
        try:
            self.log("\n" + "="*70)
            self.log("STATISTICAL ANALYSIS")
            self.log("="*70)
            
            start_ms = self.start_ms_var.get()
            end_ms = self.end_ms_var.get()
            
            test_dataset = MultiLabelEEGDataset(
                self.coco_path,
                self.annotations_path,
                'test',
                1000,
                start_ms=start_ms,
                end_ms=end_ms
            )
            
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
            
            all_probs = []
            all_labels = []
            
            self.model.eval()
            with torch.no_grad():
                for eeg, labels in test_loader:
                    eeg = eeg.to(DEVICE)
                    logits = self.model(eeg)
                    probs = torch.sigmoid(logits)
                    all_probs.append(probs.cpu())
                    all_labels.append(labels)
            
            all_probs = torch.cat(all_probs, dim=0).numpy()
            all_labels = torch.cat(all_labels, dim=0).numpy()
            
            self.log(f"\nAnalyzing {len(all_probs)} test samples...")
            self.log("\nStatistical Results:")
            self.log("-" * 90)
            self.log(f"{'Category':<15} {'Mean±SD (Present)':<20} {'Mean±SD (Absent)':<20} "
                    f"{'Δ':<8} {'d':<6} {'p-val':<8} {'N'}")
            self.log("-" * 90)
            
            results = []
            for idx, (cat_name, cat_id) in enumerate(TARGET_CATEGORIES.items()):
                present_mask = all_labels[:, idx] == 1
                absent_mask = all_labels[:, idx] == 0
                
                if present_mask.sum() >= 10 and absent_mask.sum() >= 10:
                    probs_present = all_probs[present_mask, idx]
                    probs_absent = all_probs[absent_mask, idx]
                    
                    mean_p = probs_present.mean()
                    mean_a = probs_absent.mean()
                    std_p = probs_present.std()
                    std_a = probs_absent.std()
                    diff = mean_p - mean_a
                    
                    # Statistical tests
                    t_stat, p_val = stats.ttest_ind(probs_present, probs_absent)
                    effect_size = cohens_d(probs_present, probs_absent)
                    
                    results.append({
                        'category': cat_name,
                        'mean_present': mean_p,
                        'mean_absent': mean_a,
                        'std_present': std_p,
                        'std_absent': std_a,
                        'diff': diff,
                        'effect_size': effect_size,
                        'p_value': p_val,
                        'probs_present': probs_present,
                        'probs_absent': probs_absent,
                        'n_present': present_mask.sum(),
                        'n_absent': absent_mask.sum()
                    })
            
            # Sort by effect size
            results.sort(key=lambda x: abs(x['effect_size']), reverse=True)
            
            # Print results
            significant_count = 0
            for r in results:
                sig_marker = "***" if r['p_value'] < 0.001 else "**" if r['p_value'] < 0.01 else "*" if r['p_value'] < 0.05 else ""
                
                if r['p_value'] < 0.05:
                    significant_count += 1
                
                self.log(f"{r['category']:<15} "
                        f"{r['mean_present']:.4f}±{r['std_present']:.3f}      "
                        f"{r['mean_absent']:.4f}±{r['std_absent']:.3f}      "
                        f"{r['diff']:+.4f}  "
                        f"{r['effect_size']:+.3f}  "
                        f"{r['p_value']:.4f}{sig_marker:<3} "
                        f"{r['n_present']}/{r['n_absent']}")
            
            self.log("-" * 90)
            self.log(f"Significant results (p<0.05): {significant_count}/{len(results)}")
            self.log("Effect size interpretation: d>0.2=small, d>0.5=medium, d>0.8=large")
            self.log("Significance: * p<0.05, ** p<0.01, *** p<0.001")
            
            # Plot top 4 by effect size
            for i, r in enumerate(results[:4]):
                ax = self.axes[i // 2, i % 2]
                ax.clear()
                
                ax.hist(r['probs_absent'], bins=25, alpha=0.6, label='Absent', 
                       color='blue', density=True)
                ax.hist(r['probs_present'], bins=25, alpha=0.6, label='Present',
                       color='red', density=True)
                
                ax.axvline(r['mean_absent'], color='blue', linestyle='--', linewidth=2)
                ax.axvline(r['mean_present'], color='red', linestyle='--', linewidth=2)
                
                sig = "***" if r['p_value'] < 0.001 else "**" if r['p_value'] < 0.01 else "*" if r['p_value'] < 0.05 else "ns"
                ax.set_title(f"{r['category']}\nd={r['effect_size']:+.3f}, p={r['p_value']:.4f} {sig}")
                ax.set_xlabel('Probability')
                ax.set_ylabel('Density')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            self.log(f"Analysis error: {e}")
            import traceback
            self.log(traceback.format_exc())

if __name__ == "__main__":
    print("\nEEG Statistical Signal Analyzer")
    print("="*60)
    print("Excludes lab contamination (person, chair, laptop, book)")
    print("Focuses on animals, vehicles, outdoor objects")
    print("Proper statistical testing with t-tests and effect sizes")
    print("="*60)
    
    app = StatisticalAnalyzerGUI()
    app.mainloop()