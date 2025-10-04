"""
EEG Category Brain Explorer - Fixed Channel Mapping
"""

import os
import sys
import warnings
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import mne
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy import stats
from scipy.signal import hilbert
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
from pathlib import Path
from collections import defaultdict
import threading

try:
    from datasets import load_dataset
except ImportError:
    print("Missing datasets library")
    sys.exit(1)

mne.set_log_level('WARNING')
warnings.filterwarnings('ignore')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EEG_SAMPLE_RATE = 512

TARGET_CATEGORIES = {
    'elephant': 22, 'giraffe': 25, 'bear': 23, 'zebra': 24,
    'cow': 21, 'sheep': 20, 'horse': 19, 'dog': 18, 'cat': 17, 'bird': 16,
    'airplane': 5, 'train': 7, 'boat': 9, 'bus': 6, 'truck': 8,
    'motorcycle': 4, 'bicycle': 2, 'car': 3,
    'traffic light': 10, 'fire hydrant': 11, 'stop sign': 13,
    'parking meter': 14, 'bench': 15,
    'frisbee': 34, 'skis': 35, 'snowboard': 36, 'sports ball': 37,
    'kite': 38, 'baseball bat': 39, 'skateboard': 41, 'surfboard': 42,
    'banana': 52, 'apple': 53, 'orange': 55, 'broccoli': 56,
    'pizza': 59, 'donut': 60, 'cake': 61,
}

CATEGORY_NAMES = {v: k for k, v in TARGET_CATEGORIES.items()}

# BioSemi 64 channel layout - exact names from standard montage
BIOSEMI_64_CHANNELS = [
    'Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7',
    'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7',
    'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9',
    'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz',
    'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4',
    'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz',
    'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2',
    'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2'
]

class CNNClassifier(nn.Module):
    def __init__(self, n_channels=64, n_timepoints=154, num_classes=len(TARGET_CATEGORIES)):
        super().__init__()
        
        self.conv1 = nn.Conv1d(n_channels, 128, kernel_size=25, padding=12)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(128, 256, kernel_size=15, padding=7)
        self.bn2 = nn.BatchNorm1d(256)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(256, 512, kernel_size=7, padding=3)
        self.bn3 = nn.BatchNorm1d(512)
        self.pool3 = nn.MaxPool1d(2)
        
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
        x = self.pool1(torch.nn.functional.elu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.nn.functional.elu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.nn.functional.elu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def create_mne_raw_from_alljoined(eeg_data):
    """
    Create MNE Raw object from Alljoined EEG data with proper channel locations
    
    Parameters:
    eeg_data : numpy array (64, time_points)
    
    Returns:
    raw : mne.io.RawArray with proper channel locations
    """
    # Create info structure
    info = mne.create_info(
        ch_names=BIOSEMI_64_CHANNELS,
        sfreq=EEG_SAMPLE_RATE,
        ch_types='eeg'
    )
    
    # Create raw object
    raw = mne.io.RawArray(eeg_data, info)
    
    # Set standard montage - this assigns 3D coordinates
    montage = mne.channels.make_standard_montage('biosemi64')
    raw.set_montage(montage, match_case=False, on_missing='warn')
    
    # Verify that channels have locations
    n_located = sum(1 for ch in raw.info['chs'] if not np.isnan(ch['loc'][:3]).any())
    
    if n_located < 60:  # Need at least 60/64 channels with locations
        raise RuntimeError(f"Only {n_located}/64 channels have locations. Check montage mapping.")
    
    return raw

class CategoryBrainExplorer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("EEG Category Brain Explorer")
        self.geometry("1400x900")
        
        self.model = None
        self.test_data = None
        self.category_trials = {}
        self.brain_figures = []
        self.stop_flag = threading.Event()
        
        self.setup_gui()
    
    def setup_gui(self):
        main_container = tk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        left_panel = ttk.Frame(main_container, width=400)
        main_container.add(left_panel)
        
        right_panel = ttk.Frame(main_container)
        main_container.add(right_panel)
        
        self.setup_left_panel(left_panel)
        self.setup_right_panel(right_panel)
    
    def setup_left_panel(self, parent):
        tk.Label(parent, text="Category Brain Explorer", 
                font=("Arial", 14, "bold")).pack(pady=10)
        
        # Model loading
        model_frame = ttk.LabelFrame(parent, text="Step 1: Load Trained Model")
        model_frame.pack(pady=10, padx=10, fill=tk.X)
        
        tk.Button(model_frame, text="Load Model (.pth)", 
                 command=self.load_model, bg="#2196F3", fg="white").pack(pady=5)
        self.model_status = tk.Label(model_frame, text="No model loaded", fg="gray")
        self.model_status.pack()
        
        # Data loading
        data_frame = ttk.LabelFrame(parent, text="Step 2: Load Test Data")
        data_frame.pack(pady=10, padx=10, fill=tk.X)
        
        tk.Label(data_frame, text="COCO Images:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.coco_var = tk.StringVar()
        ttk.Entry(data_frame, textvariable=self.coco_var, width=25).grid(row=0, column=1, padx=5)
        tk.Button(data_frame, text="...", command=self.browse_coco, width=3).grid(row=0, column=2)
        
        tk.Label(data_frame, text="Annotations:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.ann_var = tk.StringVar()
        ttk.Entry(data_frame, textvariable=self.ann_var, width=25).grid(row=1, column=1, padx=5)
        tk.Button(data_frame, text="...", command=self.browse_ann, width=3).grid(row=1, column=2)
        
        tk.Button(data_frame, text="Load & Classify Test Data",
                 command=self.load_and_classify, bg="#4CAF50", fg="white").grid(row=2, column=0, columnspan=3, pady=5)
        self.data_status = tk.Label(data_frame, text="No data loaded", fg="gray")
        self.data_status.grid(row=3, column=0, columnspan=3)
        
        # Category selection
        cat_frame = ttk.LabelFrame(parent, text="Step 3: Select Category")
        cat_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        tk.Label(cat_frame, text="Search:").pack()
        self.search_var = tk.StringVar()
        self.search_var.trace('w', self.filter_categories)
        tk.Entry(cat_frame, textvariable=self.search_var).pack(fill=tk.X, padx=5)
        
        list_frame = tk.Frame(cat_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.category_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set)
        self.category_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.category_listbox.yview)
        
        self.category_listbox.bind('<<ListboxSelect>>', self.on_category_select)
        
        self.cat_info = tk.Label(cat_frame, text="", fg="blue", wraplength=350)
        self.cat_info.pack()
        
        # Source analysis
        source_frame = ttk.LabelFrame(parent, text="Step 4: Brain Analysis")
        source_frame.pack(pady=10, padx=10, fill=tk.X)
        
        tk.Label(source_frame, text="Threshold (percentile):").grid(row=0, column=0, padx=5)
        self.threshold_var = tk.DoubleVar(value=0.75)
        tk.Scale(source_frame, from_=0.5, to=0.95, resolution=0.05,
                variable=self.threshold_var, orient=tk.HORIZONTAL, length=200).grid(row=0, column=1)
        
        tk.Label(source_frame, text="Time Window (s):").grid(row=1, column=0, padx=5)
        time_frame = tk.Frame(source_frame)
        time_frame.grid(row=1, column=1)
        self.time_start = tk.DoubleVar(value=0.05)
        self.time_end = tk.DoubleVar(value=0.35)
        tk.Spinbox(time_frame, from_=0, to=1, increment=0.05, textvariable=self.time_start, width=6).pack(side=tk.LEFT)
        tk.Label(time_frame, text=" to ").pack(side=tk.LEFT)
        tk.Spinbox(time_frame, from_=0, to=1, increment=0.05, textvariable=self.time_end, width=6).pack(side=tk.LEFT)
        
        tk.Button(source_frame, text="Localize Brain Sources",
                 command=self.localize_category, bg="#9C27B0", fg="white",
                 font=("Arial", 10, "bold")).grid(row=2, column=0, columnspan=2, pady=10)
        
        self.progress = ttk.Progressbar(parent, mode='determinate')
        self.progress.pack(fill=tk.X, padx=10, pady=10)
        
        self.status_label = tk.Label(parent, text="Ready", wraplength=350, fg="green")
        self.status_label.pack(pady=5)
    
    def setup_right_panel(self, parent):
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Statistics
        stats_tab = ttk.Frame(notebook)
        notebook.add(stats_tab, text="Category Statistics")
        
        self.fig_stats, self.ax_stats = plt.subplots(1, 1, figsize=(8, 6))
        canvas_stats = FigureCanvasTkAgg(self.fig_stats, stats_tab)
        canvas_stats.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Distribution
        dist_tab = ttk.Frame(notebook)
        notebook.add(dist_tab, text="Probability Distribution")
        
        self.fig_dist, self.ax_dist = plt.subplots(1, 1, figsize=(8, 6))
        canvas_dist = FigureCanvasTkAgg(self.fig_dist, dist_tab)
        canvas_dist.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Log
        log_tab = ttk.Frame(notebook)
        notebook.add(log_tab, text="Analysis Log")
        
        self.log_text = tk.Text(log_tab, bg='black', fg='lightgreen', font=('Courier', 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        log_scroll = ttk.Scrollbar(self.log_text)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=log_scroll.set)
        log_scroll.config(command=self.log_text.yview)
    
    def log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.update()
    
    def browse_coco(self):
        path = filedialog.askdirectory()
        if path:
            self.coco_var.set(path)
    
    def browse_ann(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if path:
            self.ann_var.set(path)
    
    def load_model(self):
        path = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pth")])
        if not path:
            return
        
        try:
            self.model = CNNClassifier().to(DEVICE)
            checkpoint = torch.load(path, map_location=DEVICE)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            self.model_status.config(text=f"Model loaded: {Path(path).name}", fg="green")
            self.log(f"Loaded model from {Path(path).name}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{e}")
            self.log(f"ERROR: {e}")
    
    def load_and_classify(self):
        if self.model is None:
            messagebox.showerror("Error", "Load model first")
            return
        
        if not self.coco_var.get() or not self.ann_var.get():
            messagebox.showerror("Error", "Select data paths")
            return
        
        thread = threading.Thread(target=self._classify_data, daemon=True)
        thread.start()
    
    def _classify_data(self):
        try:
            self.log("="*60)
            self.log("LOADING AND CLASSIFYING TEST DATA")
            self.progress['value'] = 10
            
            from torch.utils.data import Dataset
            
            class QuickDataset(Dataset):
                def __init__(self, coco_path, ann_path, max_samples=1000):
                    self.dataset = load_dataset("Alljoined/05_125", split='test', streaming=False)
                    self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
                    
                    with open(ann_path, 'r') as f:
                        coco_data = json.load(f)
                    
                    self.image_categories = defaultdict(set)
                    for ann in coco_data['annotations']:
                        img_id = ann['image_id']
                        if ann['category_id'] in CATEGORY_NAMES:
                            self.image_categories[img_id].add(ann['category_id'])
                    
                    self.labels = []
                    self.valid_indices = []
                    
                    for idx, sample in enumerate(self.dataset):
                        coco_id = sample['coco_id']
                        if coco_id in self.image_categories:
                            label = torch.zeros(len(TARGET_CATEGORIES))
                            for cat_id in self.image_categories[coco_id]:
                                if cat_id in CATEGORY_NAMES:
                                    cat_idx = list(TARGET_CATEGORIES.values()).index(cat_id)
                                    label[cat_idx] = 1.0
                            
                            if label.sum() > 0:
                                self.labels.append(label)
                                self.valid_indices.append(idx)
                
                def __len__(self):
                    return len(self.valid_indices)
                
                def __getitem__(self, idx):
                    real_idx = self.valid_indices[idx]
                    sample = self.dataset[real_idx]
                    
                    eeg_data = np.array(sample['EEG'], dtype=np.float32)
                    start_idx = int(0.05 * EEG_SAMPLE_RATE)
                    end_idx = int(0.35 * EEG_SAMPLE_RATE)
                    eeg_window = eeg_data[:, start_idx:end_idx]
                    
                    eeg_window = (eeg_window - eeg_window.mean(axis=1, keepdims=True)) / \
                                 (eeg_window.std(axis=1, keepdims=True) + 1e-8)
                    
                    return torch.from_numpy(eeg_window).float(), self.labels[idx], eeg_data
            
            dataset = QuickDataset(self.coco_var.get(), self.ann_var.get())
            loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
            
            self.log(f"Loaded {len(dataset)} test samples")
            self.progress['value'] = 30
            
            all_probs = []
            all_labels = []
            all_eeg = []
            
            with torch.no_grad():
                for eeg, labels, eeg_full in loader:
                    eeg = eeg.to(DEVICE)
                    logits = self.model(eeg)
                    probs = torch.sigmoid(logits)
                    all_probs.append(probs.cpu())
                    all_labels.append(labels)
                    all_eeg.append(eeg_full)
            
            all_probs = torch.cat(all_probs, dim=0).numpy()
            all_labels = torch.cat(all_labels, dim=0).numpy()
            all_eeg = torch.cat(all_eeg, dim=0).numpy()
            
            self.progress['value'] = 70
            
            self.category_trials = {}
            
            for cat_idx, (cat_name, cat_id) in enumerate(TARGET_CATEGORIES.items()):
                present_mask = all_labels[:, cat_idx] == 1
                
                if present_mask.sum() >= 5:
                    probs_present = all_probs[present_mask, cat_idx]
                    eeg_present = all_eeg[present_mask]
                    
                    self.category_trials[cat_name] = {
                        'probabilities': probs_present,
                        'eeg_data': eeg_present,
                        'n_trials': len(probs_present),
                        'mean_prob': probs_present.mean(),
                        'std_prob': probs_present.std()
                    }
            
            self.progress['value'] = 90
            
            self.update_category_list()
            self.plot_category_stats()
            
            self.data_status.config(text=f"{len(self.category_trials)} categories ready", fg="green")
            self.log(f"Classification complete: {len(self.category_trials)} categories with ≥5 trials")
            
            self.progress['value'] = 100
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.log(f"ERROR: {e}")
            import traceback
            self.log(traceback.format_exc())
        finally:
            self.progress['value'] = 0
    
    def update_category_list(self):
        self.category_listbox.delete(0, tk.END)
        self.all_categories = []
        
        for cat_name in sorted(self.category_trials.keys()):
            info = self.category_trials[cat_name]
            display = f"{cat_name:<15} (N={info['n_trials']:3d}, p̄={info['mean_prob']:.3f})"
            self.category_listbox.insert(tk.END, display)
            self.all_categories.append(cat_name)
    
    def filter_categories(self, *args):
        search = self.search_var.get().lower()
        self.category_listbox.delete(0, tk.END)
        
        for cat_name in sorted(self.category_trials.keys()):
            if search in cat_name.lower():
                info = self.category_trials[cat_name]
                display = f"{cat_name:<15} (N={info['n_trials']:3d}, p̄={info['mean_prob']:.3f})"
                self.category_listbox.insert(tk.END, display)
    
    def on_category_select(self, event):
        selection = self.category_listbox.curselection()
        if not selection:
            return
        
        cat_display = self.category_listbox.get(selection[0])
        cat_name = cat_display.split()[0]
        
        if cat_name in self.category_trials:
            info = self.category_trials[cat_name]
            self.cat_info.config(
                text=f"Selected: {cat_name}\n"
                     f"Trials: {info['n_trials']} | "
                     f"Mean prob: {info['mean_prob']:.3f} ± {info['std_prob']:.3f}"
            )
            
            self.plot_probability_distribution(cat_name)
    
    def plot_category_stats(self):
        self.ax_stats.clear()
        
        categories = sorted(self.category_trials.keys(), 
                          key=lambda x: self.category_trials[x]['mean_prob'],
                          reverse=True)[:15]
        
        means = [self.category_trials[c]['mean_prob'] for c in categories]
        stds = [self.category_trials[c]['std_prob'] for c in categories]
        
        self.ax_stats.barh(categories, means, xerr=stds, capsize=3, color='steelblue')
        self.ax_stats.set_xlabel('Mean Probability')
        self.ax_stats.set_title('Top 15 Categories by Mean Probability')
        self.ax_stats.grid(axis='x', alpha=0.3)
        
        self.fig_stats.tight_layout()
        self.fig_stats.canvas.draw()
    
    def plot_probability_distribution(self, cat_name):
        self.ax_dist.clear()
        
        info = self.category_trials[cat_name]
        probs = info['probabilities']
        
        self.ax_dist.hist(probs, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        self.ax_dist.axvline(info['mean_prob'], color='red', linestyle='--', linewidth=2, label='Mean')
        self.ax_dist.set_xlabel('Probability')
        self.ax_dist.set_ylabel('Count')
        self.ax_dist.set_title(f"Probability Distribution: {cat_name}")
        self.ax_dist.legend()
        self.ax_dist.grid(alpha=0.3)
        
        self.fig_dist.tight_layout()
        self.fig_dist.canvas.draw()
    
    def localize_category(self):
        selection = self.category_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Select a category first")
            return
        
        cat_display = self.category_listbox.get(selection[0])
        cat_name = cat_display.split()[0]
        
        thread = threading.Thread(target=self._run_localization, args=(cat_name,), daemon=True)
        thread.start()
    
    def _run_localization(self, cat_name):
        try:
            self.log(f"\n{'='*60}")
            self.log(f"SOURCE LOCALIZATION: {cat_name.upper()}")
            self.log(f"{'='*60}")
            
            info = self.category_trials[cat_name]
            threshold = self.threshold_var.get()
            
            percentile_thresh = np.percentile(info['probabilities'], threshold * 100)
            high_mask = info['probabilities'] >= percentile_thresh
            
            high_eeg = info['eeg_data'][high_mask]
            self.log(f"Selected {len(high_eeg)} high-probability trials (>{percentile_thresh:.3f})")
            
            if len(high_eeg) < 3:
                messagebox.showwarning("Warning", "Need at least 3 high-probability trials")
                return
            
            self.progress['value'] = 20
            
            # Average trials
            avg_eeg = high_eeg.mean(axis=0)
            
            # Create MNE Raw with proper channel locations
            self.log("Creating MNE Raw object with BioSemi 64 montage...")
            raw = create_mne_raw_from_alljoined(avg_eeg)
            
            self.log(f"Channels with locations: {sum(1 for ch in raw.info['chs'] if not np.isnan(ch['loc'][:3]).any())}/64")
            
            # Preprocess
            raw.filter(0.5, 50, fir_design='firwin', verbose=False)
            raw.set_eeg_reference('average', projection=True, verbose=False)
            
            t_start = self.time_start.get()
            t_end = min(self.time_end.get(), raw.times[-1])
            raw.crop(tmin=t_start, tmax=t_end)
            
            self.log(f"Analyzing time window: {t_start:.2f}-{t_end:.2f}s")
            self.progress['value'] = 40
            
            # Setup forward model
            subjects_dir = os.path.join(os.path.expanduser('~'), 'mne_data')
            
            if not os.path.isdir(os.path.join(subjects_dir, 'fsaverage')):
                self.log("Downloading fsaverage template...")
                mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir, verbose=False)
            
            src = mne.setup_source_space('fsaverage', spacing='ico4',
                                        subjects_dir=subjects_dir, verbose=False)
            
            model = mne.make_bem_model('fsaverage', ico=4,
                                      conductivity=(0.3, 0.006, 0.3),
                                      subjects_dir=subjects_dir, verbose=False)
            bem_sol = mne.make_bem_solution(model, verbose=False)
            
            self.progress['value'] = 60
            
            fwd = mne.make_forward_solution(raw.info, trans='fsaverage',
                                           src=src, bem=bem_sol,
                                           eeg=True, meg=False, verbose=False)
            
            noise_cov = mne.compute_raw_covariance(raw, verbose=False)
            inv = mne.minimum_norm.make_inverse_operator(raw.info, fwd, noise_cov,
                                                         loose=0.2, depth=0.8, verbose=False)
            
            self.progress['value'] = 80
            
            stc = mne.minimum_norm.apply_inverse_raw(raw, inv, lambda2=1/9,
                                                     method='sLORETA', verbose=False)
            
            self.log("Source reconstruction complete")
            self.progress['value'] = 90
            
            self.after(0, lambda: self._visualize_sources(stc, cat_name, subjects_dir))
            
        except Exception as e:
            self.log(f"ERROR: {e}")
            import traceback
            self.log(traceback.format_exc())
            messagebox.showerror("Error", str(e))
        finally:
            self.progress['value'] = 0
    
    def _visualize_sources(self, stc, cat_name, subjects_dir):
        try:
            power_data = stc.data ** 2
            stc_power = mne.SourceEstimate(
                power_data, vertices=stc.vertices,
                tmin=stc.tmin, tstep=stc.tstep, subject='fsaverage'
            )
            
            brain = stc_power.plot(
                subjects_dir=subjects_dir,
                subject='fsaverage',
                surface='pial',
                hemi='both',
                colormap='hot',
                clim=dict(kind='percent', lims=[95, 97, 99]),
                time_label=f"{cat_name} - Power (t=%0.2fs)",
                size=(1200, 800),
                background='white',
                verbose=False
            )
            
            self.brain_figures.append(brain)
            self.log(f"3D visualization created for {cat_name}")
            
        except Exception as e:
            self.log(f"Visualization error: {e}")
            messagebox.showerror("Error", f"Visualization failed:\n{e}")

if __name__ == "__main__":
    try:
        mne.viz.set_3d_backend("pyvistaqt")
    except:
        pass
    
    app = CategoryBrainExplorer()
    app.mainloop()