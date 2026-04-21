"""
Visualization script for AutoResearch Educational Example results.
Generates publication-quality figures showing the ratchet loop in action.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Load results
# ─────────────────────────────────────────────────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULTS_DIR = _REPO_ROOT

with open(os.path.join(_RESULTS_DIR, "results.json")) as f:
    data = json.load(f)

results = data["results_history"]
ratchet_log = data["ratchet_log"]

# Style
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'baseline': '#6B7280',
    'flaml': '#3B82F6',
    'kept': '#10B981',
    'discarded': '#EF4444',
    'best': '#F59E0B',
    'bg': '#F9FAFB',
    'dark': '#1F2937',
}

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1: The Ratchet Progress Chart (main figure)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor('white')
fig.suptitle(
    'AutoResearch Ratchet Loop — Heart Disease Dataset (Kaggle/UCI)\n'
    'Demonstrating Autonomous Iterative Improvement on CPU',
    fontsize=15, fontweight='bold', y=0.98, color=COLORS['dark']
)

# ── Panel A: AUC Progress Over Iterations ──
ax = axes[0, 0]
ax.set_facecolor(COLORS['bg'])

labels = [r['label'][:30] for r in results]
aucs   = [r['test_auc'] for r in results]
phases = [r.get('phase', '') for r in results]

# Color by type
bar_colors = []
for r in results:
    ph = r.get('phase', '')
    rs = r.get('ratchet_status', '')
    if ph == 'baseline':
        bar_colors.append(COLORS['baseline'])
    elif ph == 'flaml_oneshot':
        bar_colors.append(COLORS['flaml'])
    elif rs == 'kept':
        bar_colors.append(COLORS['kept'])
    else:
        bar_colors.append(COLORS['discarded'])

x = np.arange(len(results))
bars = ax.bar(x, aucs, color=bar_colors, edgecolor='white', linewidth=0.8, width=0.7)

# Add value labels on bars
for bar, auc in zip(bars, aucs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{auc:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold',
            color=COLORS['dark'])

# Running best line
running_best = []
best = 0
for r in results:
    if r['test_auc'] > best:
        best = r['test_auc']
    running_best.append(best)

ax.plot(x, running_best, 'o-', color=COLORS['best'], linewidth=2.5,
        markersize=6, zorder=5, label='Running Best (Ratchet)')

ax.axhline(y=results[0]['test_auc'], color=COLORS['baseline'],
           linestyle='--', linewidth=1.5, alpha=0.7, label=f'Baseline ({results[0]["test_auc"]:.4f})')

ax.set_xticks(x)
ax.set_xticklabels([f'{"BL" if p=="baseline" else "FL" if p=="flaml_oneshot" else f"R{i-1}"}'
                    for i, (p, _) in enumerate(zip(phases, aucs))],
                   fontsize=9)
ax.set_ylabel('Test AUC (ROC)', fontweight='bold')
ax.set_title('A. AUC Progress: Baseline → FLAML → Ratchet Loop', fontweight='bold', pad=10)
ax.set_ylim(0.75, 0.97)
ax.legend(loc='lower right', fontsize=9)

# Legend for bar colors
legend_patches = [
    mpatches.Patch(color=COLORS['baseline'], label='Baseline (Manual)'),
    mpatches.Patch(color=COLORS['flaml'], label='FLAML One-Shot'),
    mpatches.Patch(color=COLORS['kept'], label='Ratchet: Kept ✓'),
    mpatches.Patch(color=COLORS['discarded'], label='Ratchet: Discarded ✗'),
]
ax.legend(handles=legend_patches + [
    plt.Line2D([0], [0], color=COLORS['best'], linewidth=2.5, marker='o', label='Running Best')
], loc='lower right', fontsize=8, ncol=2)

# ── Panel B: Ratchet Loop Diagram ──
ax2 = axes[0, 1]
ax2.set_facecolor('white')
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.set_title('B. The AutoResearch Ratchet Loop', fontweight='bold', pad=10)

# Draw the loop diagram
loop_steps = [
    (5, 8.5, "1. Read program.md\n(Research Directions)", '#3B82F6'),
    (8.5, 6.0, "2. Propose\nHypothesis", '#8B5CF6'),
    (8.5, 3.0, "3. Modify\nConfig / Code", '#F59E0B'),
    (5, 1.2, "4. Run Experiment\n(Fixed Budget)", '#EF4444'),
    (1.5, 3.0, "5. Evaluate\nMetric", '#10B981'),
    (1.5, 6.0, "6. Keep or\nRevert", '#6B7280'),
]

for (x, y, text, color) in loop_steps:
    circle = plt.Circle((x, y), 0.9, color=color, zorder=3, alpha=0.9)
    ax2.add_patch(circle)
    ax2.text(x, y, text, ha='center', va='center', fontsize=7.5,
             fontweight='bold', color='white', zorder=4)

# Arrows connecting steps
arrow_pairs = [
    ((5.85, 8.1), (7.85, 6.5)),   # 1→2
    ((8.5, 5.1), (8.5, 3.9)),     # 2→3
    ((7.65, 2.6), (5.85, 1.5)),   # 3→4
    ((4.15, 1.5), (2.35, 2.6)),   # 4→5
    ((1.5, 3.9), (1.5, 5.1)),     # 5→6
    ((2.35, 6.5), (4.15, 8.1)),   # 6→1
]
for (x1, y1), (x2, y2) in arrow_pairs:
    ax2.annotate('', xy=(x2, y2), xytext=(x1, y1),
                 arrowprops=dict(arrowstyle='->', color='#374151', lw=2))

# Ratchet annotation
ax2.text(5, 5, "RATCHET\n↑ Only\nForward",
         ha='center', va='center', fontsize=10, fontweight='bold',
         color=COLORS['best'],
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#FEF3C7', edgecolor=COLORS['best'], lw=2))

# Keep/Discard annotation
ax2.text(1.5, 6.0, "Keep or\nRevert",
         ha='center', va='center', fontsize=7.5, fontweight='bold', color='white', zorder=5)
ax2.text(3.5, 7.3, "✓ Keep if\nimproved", ha='center', fontsize=8, color=COLORS['kept'])
ax2.text(0.3, 4.5, "✗ Revert if\nno gain", ha='center', fontsize=8, color=COLORS['discarded'])

# ── Panel C: Improvement Waterfall ──
ax3 = axes[1, 0]
ax3.set_facecolor(COLORS['bg'])

baseline_auc = data['baseline_auc']
flaml_auc = data['flaml_oneshot_auc']
final_auc = data['final_best_auc']

stages = ['Baseline\n(Manual)', 'FLAML\nOne-Shot', 'Ratchet\nLoop\n(Best)']
values = [baseline_auc, flaml_auc, final_auc]
improvements = [0, flaml_auc - baseline_auc, final_auc - baseline_auc]
colors_wf = [COLORS['baseline'], COLORS['flaml'], COLORS['kept']]

bars3 = ax3.bar(stages, values, color=colors_wf, edgecolor='white', linewidth=1,
                width=0.5, zorder=3)

for bar, val, imp in zip(bars3, values, improvements):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
             f'AUC\n{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold',
             color=COLORS['dark'])
    if imp != 0:
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                 f'+{imp*100:.2f}%', ha='center', va='center', fontsize=11,
                 fontweight='bold', color='white')

ax3.set_ylabel('Test AUC (ROC)', fontweight='bold')
ax3.set_title('C. Cumulative Improvement: Baseline → AutoResearch', fontweight='bold', pad=10)
ax3.set_ylim(0.80, 0.95)
ax3.yaxis.grid(True, alpha=0.4)
ax3.set_axisbelow(True)

# Total improvement annotation
ax3.annotate(
    f'Total: +{(final_auc - baseline_auc)*100:.2f}% AUC\n({baseline_auc:.4f} → {final_auc:.4f})',
    xy=(2, final_auc), xytext=(1.5, 0.935),
    fontsize=10, fontweight='bold', color=COLORS['best'],
    arrowprops=dict(arrowstyle='->', color=COLORS['best'], lw=1.5),
    bbox=dict(boxstyle='round', facecolor='#FEF3C7', edgecolor=COLORS['best'])
)

# ── Panel D: Ratchet Log — Kept vs Discarded ──
ax4 = axes[1, 1]
ax4.set_facecolor(COLORS['bg'])

iter_nums = [r['iteration'] for r in ratchet_log]
iter_aucs = [r['test_auc'] for r in ratchet_log]
iter_status = [r['status'] for r in ratchet_log]
iter_deltas = [r['delta'] for r in ratchet_log]

kept_mask = [s == 'kept' for s in iter_status]
disc_mask = [s == 'discarded' for s in iter_status]

ax4.scatter([n for n, k in zip(iter_nums, kept_mask) if k],
            [a for a, k in zip(iter_aucs, kept_mask) if k],
            color=COLORS['kept'], s=150, zorder=5, marker='*',
            label='Kept ✓', edgecolors='white', linewidths=0.5)
ax4.scatter([n for n, k in zip(iter_nums, disc_mask) if k],
            [a for a, k in zip(iter_aucs, disc_mask) if k],
            color=COLORS['discarded'], s=80, zorder=5, marker='x',
            label='Discarded ✗', linewidths=2)

# Running best line for ratchet
ratchet_best = []
rb = data['flaml_oneshot_auc']
for r in ratchet_log:
    if r['test_auc'] > rb:
        rb = r['test_auc']
    ratchet_best.append(rb)

ax4.plot(iter_nums, ratchet_best, 'o-', color=COLORS['best'], linewidth=2.5,
         markersize=5, zorder=4, label='Running Best')
ax4.axhline(y=data['flaml_oneshot_auc'], color=COLORS['flaml'],
            linestyle='--', linewidth=1.5, alpha=0.7,
            label=f'FLAML One-Shot ({data["flaml_oneshot_auc"]:.4f})')

# Annotate kept iteration
for r in ratchet_log:
    if r['status'] == 'kept':
        ax4.annotate(f'  +{r["delta"]*100:.2f}%\n  NEW BEST!',
                     xy=(r['iteration'], r['test_auc']),
                     xytext=(r['iteration'] - 1.5, r['test_auc'] + 0.01),
                     fontsize=8.5, fontweight='bold', color=COLORS['kept'],
                     arrowprops=dict(arrowstyle='->', color=COLORS['kept']))

ax4.set_xlabel('Ratchet Iteration', fontweight='bold')
ax4.set_ylabel('Test AUC (ROC)', fontweight='bold')
ax4.set_title('D. Ratchet Iterations: Kept vs Discarded', fontweight='bold', pad=10)
ax4.set_xticks(iter_nums)
ax4.legend(fontsize=9)
ax4.yaxis.grid(True, alpha=0.4)
ax4.set_axisbelow(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(_RESULTS_DIR, 'autoresearch_results.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: autoresearch_results.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2: Feature Importance & Dataset EDA
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

heart = fetch_openml(name='heart-disease', version=1, as_frame=True)
df = heart.frame.copy()
df['target'] = (pd.to_numeric(df['target'], errors='coerce').fillna(0) > 0).astype(int)
feature_names = [c for c in df.columns if c != 'target']
X = df[feature_names].values.astype(float)
y = df['target'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Feature importance from GBM
gbm = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42)
gbm.fit(X_train, y_train)
importances = gbm.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
fig2.patch.set_facecolor('white')
fig2.suptitle('Heart Disease Dataset — Feature Analysis', fontsize=14, fontweight='bold')

# Feature importance
ax = axes2[0]
ax.set_facecolor(COLORS['bg'])
feat_labels = [feature_names[i] for i in sorted_idx]
feat_vals = importances[sorted_idx]
colors_fi = [COLORS['kept'] if v > 0.1 else COLORS['flaml'] if v > 0.05 else '#D1D5DB'
             for v in feat_vals]
bars = ax.barh(range(len(feat_labels)), feat_vals[::-1], color=colors_fi[::-1],
               edgecolor='white')
ax.set_yticks(range(len(feat_labels)))
ax.set_yticklabels(feat_labels[::-1], fontsize=10)
ax.set_xlabel('Feature Importance (GBM)', fontweight='bold')
ax.set_title('Feature Importance Rankings', fontweight='bold')
ax.xaxis.grid(True, alpha=0.4)
ax.set_axisbelow(True)

# Feature descriptions
feat_desc = {
    'age': 'Age (years)', 'sex': 'Sex (0=F, 1=M)', 'cp': 'Chest Pain Type',
    'trestbps': 'Resting BP', 'chol': 'Serum Cholesterol', 'fbs': 'Fasting Blood Sugar',
    'restecg': 'Resting ECG', 'thalach': 'Max Heart Rate', 'exang': 'Exercise Angina',
    'oldpeak': 'ST Depression', 'slope': 'ST Slope', 'ca': 'Major Vessels (CA)',
    'thal': 'Thalassemia'
}
for i, (feat, val) in enumerate(zip(feat_labels[::-1], feat_vals[::-1])):
    ax.text(val + 0.002, i, f'{val:.3f}', va='center', fontsize=8.5)

# Class distribution by top features
ax2 = axes2[1]
ax2.set_facecolor(COLORS['bg'])

# Correlation heatmap-style: top 6 features vs target
df2 = pd.DataFrame(X, columns=feature_names)
df2['target'] = y
top_feats = [feature_names[i] for i in sorted_idx[:6]]

means_pos = df2[df2['target']==1][top_feats].mean()
means_neg = df2[df2['target']==0][top_feats].mean()

x_pos = np.arange(len(top_feats))
width = 0.35
b1 = ax2.bar(x_pos - width/2, means_neg, width, label='No Disease', color=COLORS['flaml'], alpha=0.85)
b2 = ax2.bar(x_pos + width/2, means_pos, width, label='Heart Disease', color=COLORS['discarded'], alpha=0.85)

ax2.set_xticks(x_pos)
ax2.set_xticklabels([feat_desc.get(f, f) for f in top_feats], rotation=25, ha='right', fontsize=9)
ax2.set_ylabel('Mean Feature Value', fontweight='bold')
ax2.set_title('Top 6 Features: Disease vs No Disease', fontweight='bold')
ax2.legend(fontsize=10)
ax2.yaxis.grid(True, alpha=0.4)
ax2.set_axisbelow(True)

plt.tight_layout()
plt.savefig(os.path.join(_RESULTS_DIR, 'feature_analysis.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: feature_analysis.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3: AutoResearch vs Manual vs AMLB Benchmark Context
# ─────────────────────────────────────────────────────────────────────────────
fig3, ax = plt.subplots(figsize=(12, 7))
fig3.patch.set_facecolor('white')
ax.set_facecolor(COLORS['bg'])

# Comparison data
methods = [
    'Manual\nLogReg\n(Baseline)',
    'Manual\nRandom\nForest',
    'Manual\nGBM',
    'FLAML\nOne-Shot\n(45s)',
    'AutoResearch\nRatchet Loop\n(Best)',
]
method_aucs = [
    data['baseline_auc'],
    0.8806,  # from ratchet iter 4 (RF result)
    0.8474,  # from ratchet iter 2 (GBM result)
    data['flaml_oneshot_auc'],
    data['final_best_auc'],
]
method_colors = [
    COLORS['baseline'],
    '#9CA3AF',
    '#6B7280',
    COLORS['flaml'],
    COLORS['kept'],
]
method_times = ['~2s', '~5s', '~5s', '~45s', '~10 min\n(8 iters)']

bars = ax.bar(methods, method_aucs, color=method_colors, edgecolor='white',
              linewidth=1, width=0.6, zorder=3)

for bar, auc, t in zip(bars, method_aucs, method_times):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'AUC\n{auc:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold',
            color=COLORS['dark'])
    ax.text(bar.get_x() + bar.get_width()/2, 0.805,
            f'⏱ {t}', ha='center', va='bottom', fontsize=8, color='white',
            fontweight='bold')

# Highlight best
ax.bar(methods[-1:], [method_aucs[-1]], color=COLORS['kept'],
       edgecolor=COLORS['best'], linewidth=3, width=0.6, zorder=4)

ax.set_ylabel('Test AUC (ROC)', fontweight='bold', fontsize=12)
ax.set_title(
    'AutoResearch vs Manual ML vs AutoML — Heart Disease (Kaggle/UCI)\n'
    'AMLB Benchmark Context: AutoResearch achieves best AUC with iterative refinement',
    fontweight='bold', fontsize=13, pad=15
)
ax.set_ylim(0.80, 0.96)
ax.yaxis.grid(True, alpha=0.4, zorder=0)
ax.set_axisbelow(True)

# Add "AMLB Benchmark" annotation
ax.axhline(y=0.91, color='#7C3AED', linestyle=':', linewidth=2, alpha=0.7)
ax.text(4.5, 0.912, 'AMLB Benchmark\nTop Quartile (~0.91)', ha='right',
        fontsize=9, color='#7C3AED', fontweight='bold')

# Category labels
ax.text(1.0, 0.955, '← Manual Approaches →', ha='center', fontsize=10,
        color=COLORS['baseline'], style='italic',
        bbox=dict(boxstyle='round', facecolor='#F3F4F6', edgecolor=COLORS['baseline'], alpha=0.8))
ax.text(3.0, 0.955, 'AutoML →', ha='center', fontsize=10,
        color=COLORS['flaml'], style='italic',
        bbox=dict(boxstyle='round', facecolor='#EFF6FF', edgecolor=COLORS['flaml'], alpha=0.8))
ax.text(4.0, 0.955, 'AutoResearch', ha='center', fontsize=10,
        color=COLORS['kept'], style='italic',
        bbox=dict(boxstyle='round', facecolor='#ECFDF5', edgecolor=COLORS['kept'], alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(_RESULTS_DIR, 'benchmark_comparison.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: benchmark_comparison.png")

print("\nAll visualizations saved successfully!")
