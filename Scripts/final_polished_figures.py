"""
===============================================================================
ADVERSARIAL TRANSFERABILITY TO HOG-BASED CLASSIFIERS
Publication-Quality Figures - FINAL POLISHED VERSION
===============================================================================

Professional Academic Color Palette (Colorblind-friendly, based on Okabe-Ito):
- Primary: Royal Blue (#0072B2) - Main classifier/clean data
- Secondary: Vermillion/Reddish-Orange (#D55E00) - Attack results  
- Tertiary: Sky Blue (#56B4E9) - Secondary comparisons
- Accent: Bluish Green (#009E73) - Positive/retention
- Warning: Orange (#E69F00) - FGSM attack
- Alert: Reddish Purple (#CC79A7) - ANN/neural

Statistical Notes:
- r = Pearson correlation coefficient (measures linear relationship strength)
- p = p-value (probability result occurred by chance; p<0.05 is significant)
- Correlations computed across configurations, NOT repeated trials
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
from scipy import stats
from math import pi
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PROFESSIONAL ACADEMIC COLOR PALETTE (Okabe-Ito based + Royal Blue theme)
# =============================================================================

# Primary palette - colorblind friendly
PALETTE = {
    'royal_blue': '#0072B2',      # Primary - clean/baseline
    'vermillion': '#D55E00',      # Secondary - attacks/drops
    'sky_blue': '#56B4E9',        # Tertiary - comparisons
    'bluish_green': '#009E73',    # Success/retention
    'orange': '#E69F00',          # FGSM
    'reddish_purple': '#CC79A7',  # ANN
    'dark_gray': '#404040',       # Text/lines
    'light_gray': '#E5E5E5',      # Background
}

# Classifier colors - consistent throughout
COLORS = {
    'KNN': '#0072B2',        # Royal Blue
    'DT': '#E69F00',         # Orange
    'LSVM': '#009E73',       # Bluish Green
    'KSVM': '#D55E00',       # Vermillion
    'ANN': '#CC79A7',        # Reddish Purple
    'VGG': '#332288',        # Dark Purple (Surrogate)
    'AlexNet': '#56B4E9',    # Sky Blue (Target)
}

# Attack colors
ATTACK_COLORS = {
    'Clean': '#009E73',      # Royal Blue
    'FGSM': '#0072B2' ,       # Orange
    'PGD': '#D55E00',        # Vermillion
}

MARKERS = {
    'KNN': 'o', 'DT': 's', 'LSVM': '^', 'KSVM': 'D', 
    'ANN': 'p', 'VGG': 'X', 'AlexNet': 'P'
}

# =============================================================================
# PUBLICATION STYLE CONFIGURATION
# =============================================================================

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'lines.linewidth': 2,
    'lines.markersize': 8,
    'axes.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': PALETTE['dark_gray'],
    'axes.labelcolor': PALETTE['dark_gray'],
    'xtick.color': PALETTE['dark_gray'],
    'ytick.color': PALETTE['dark_gray'],
    'grid.linewidth': 0.5,
    'grid.alpha': 0.4,
    'legend.frameon': True,
    'legend.framealpha': 0.95,
    'legend.edgecolor': PALETTE['dark_gray'],
    'legend.fancybox': False,
})

# =============================================================================
# DATA LOADING
# =============================================================================

def load_all_data():
    """Load experiment results and CNN baseline data."""
    
    with open('/home/achraf/Research/MLHACK_PAPER/JSONS/experiment_results.json', 'r') as f:
        results = json.load(f)
    with open('/home/achraf/Research/MLHACK_PAPER/JSONS/experiment_results_ANN.json', 'r') as f:
        results_ann = json.load(f)
    with open('/home/achraf/Research/MLHACK_PAPER/JSONS/configs.json', 'r') as f:
        configs = json.load(f)
    
    data = []
    classifiers = ['KNN', 'DT', 'LSVM', 'KSVM']
    
    for i, config in enumerate(configs):
        exp_key = f'EXPERIMENT{i}'
        
        for clf in classifiers:
            if exp_key in results and clf in results[exp_key]:
                data.append({
                    'experiment': i,
                    'classifier': clf,
                    'pipeline': 'HOG+Classical',
                    'cell_size': config['PXCELL'],
                    'orientations': config['ORIENTATIONS'],
                    'block_size': config['CELLBLOCK'],
                    'epsilon': config['EPS'],
                    'orig_acc': results[exp_key][clf]['orig_acc'],
                    'fsgm_acc': results[exp_key][clf]['fsgm_acc'],
                    'pgd_acc': results[exp_key][clf]['pgd_acc'],
                })
        
        if exp_key in results_ann and 'ANN' in results_ann[exp_key]:
            ann_orig = results_ann[exp_key]['ANN']['orig_acc']
            if ann_orig > 0.15:
                data.append({
                    'experiment': i,
                    'classifier': 'ANN',
                    'pipeline': 'HOG+Neural',
                    'cell_size': config['PXCELL'],
                    'orientations': config['ORIENTATIONS'],
                    'block_size': config['CELLBLOCK'],
                    'epsilon': config['EPS'],
                    'orig_acc': ann_orig,
                    'fsgm_acc': results_ann[exp_key]['ANN']['fsgm_acc'],
                    'pgd_acc': results_ann[exp_key]['ANN']['pgd_acc'],
                })
    
    df = pd.DataFrame(data)
    
    # Derived metrics
    df['fsgm_drop'] = df['orig_acc'] - df['fsgm_acc']
    df['pgd_drop'] = df['orig_acc'] - df['pgd_acc']
    df['fsgm_rel_drop'] = (df['orig_acc'] - df['fsgm_acc']) / df['orig_acc'] * 100
    df['pgd_rel_drop'] = (df['orig_acc'] - df['pgd_acc']) / df['orig_acc'] * 100
    df['fsgm_retention'] = df['fsgm_acc'] / df['orig_acc'] * 100
    df['pgd_retention'] = df['pgd_acc'] / df['orig_acc'] * 100
    
    # CNN baselines
    cnn_data = pd.DataFrame([
        {'classifier': 'VGG', 'pipeline': 'Surrogate', 
         'orig_acc': 0.98, 'fsgm_acc': 0.18, 'pgd_acc': 0.01},
        {'classifier': 'AlexNet', 'pipeline': 'Target', 
         'orig_acc': 0.90, 'fsgm_acc': 0.77, 'pgd_acc': 0.83},
    ])
    cnn_data['fsgm_rel_drop'] = (cnn_data['orig_acc'] - cnn_data['fsgm_acc']) / cnn_data['orig_acc'] * 100
    cnn_data['pgd_rel_drop'] = (cnn_data['orig_acc'] - cnn_data['pgd_acc']) / cnn_data['orig_acc'] * 100
    cnn_data['fsgm_retention'] = cnn_data['fsgm_acc'] / cnn_data['orig_acc'] * 100
    cnn_data['pgd_retention'] = cnn_data['pgd_acc'] / cnn_data['orig_acc'] * 100
    
    return df, cnn_data


def save_figure(fig, name, output_dir='/home/achraf/Research/MLHACK_PAPER/Figures'):
    """Save figure in both SVG and PDF formats."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    fig.savefig(f'{output_dir}/{name}.svg', format='svg', bbox_inches='tight', )
    fig.savefig(f'{output_dir}/{name}.pdf', format='pdf', bbox_inches='tight', dpi=300)
    # fig.savefig(f'{output_dir}/{name}.png', format='png', bbox_inches='tight', dpi=300)
    print(f"  ✓ Saved: {name}.svg, at {output_dir}")


# =============================================================================
# FIGURE 1: CROSS-PARADIGM COMPARISON (includes all models)
# =============================================================================

def figure1_cross_paradigm(df, cnn_data):
    """Cross-paradigm adversarial degradation - CENTRAL RESULT."""
    
    c5 = df[df['experiment'] == 4]
    
    # All models including ANN
    models = ['KSVM', 'KNN', 'LSVM', 'ANN', 'AlexNet', 'VGG']
    
    fig, ax = plt.subplots(figsize=(10, 5.5))
    
    x = np.arange(len(models))
    width = 0.25
    
    orig_vals, fsgm_vals, pgd_vals = [], [], []
    
    for model in models:
        if model in ['VGG', 'AlexNet']:
            row = cnn_data[cnn_data['classifier'] == model].iloc[0]
        else:
            model_data = c5[c5['classifier'] == model]
            if len(model_data) == 0:
                # Fallback to experiment 3 for ANN if not in C5
                model_data = df[(df['experiment'] == 3) & (df['classifier'] == model)]
            row = model_data.iloc[0]
        orig_vals.append(row['orig_acc'])
        fsgm_vals.append(row['fsgm_acc'])
        pgd_vals.append(row['pgd_acc'])
    
    bars1 = ax.bar(x - width, orig_vals, width, label='Clean', 
                   color=ATTACK_COLORS['Clean'], edgecolor='white', linewidth=1)
    bars2 = ax.bar(x, fsgm_vals, width, label='FGSM', 
                   color=ATTACK_COLORS['FGSM'], edgecolor='white', linewidth=1)
    bars3 = ax.bar(x + width, pgd_vals, width, label='PGD', 
                   color=ATTACK_COLORS['PGD'], edgecolor='white', linewidth=1)
    
    # Value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Labels
    pipeline_labels = ['RBF-SVM\n(HOG)', 'K-NN\n(HOG)', 'Lin-SVM\n(HOG)', 
                       'ANN\n(HOG)', 'AlexNet\n(Target)', 'VGG\n(Surrogate)']
    ax.set_xticks(x)
    ax.set_xticklabels(pipeline_labels, fontsize=10)
    
    # Separator
    ax.axvline(x=3.5, color=PALETTE['dark_gray'], linestyle='--', alpha=0.5, linewidth=1.5)
    ax.text(1.5, 1.02, 'HOG Feature Pipelines', ha='center', fontsize=11, 
            style='italic', color=PALETTE['dark_gray'])
    ax.text(4.5, 1.02, 'CNN Pipelines', ha='center', fontsize=11, 
            style='italic', color=PALETTE['dark_gray'])
    
    ax.set_ylabel('Classification Accuracy', fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper right', title='Condition', title_fontsize=10)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    return fig


# =============================================================================
# FIGURE 2: FGSM vs PGD SCATTER (includes all models + CNN)
# =============================================================================

def figure2_fgsm_vs_pgd(df, cnn_data):
    """FGSM vs PGD - SURPRISING REVERSAL finding."""
    
    fig, ax = plt.subplots(figsize=(7, 7))
    
    # Plot all HOG classifiers including ANN
    for clf in ['KNN', 'DT', 'LSVM', 'KSVM', 'ANN']:
        clf_data = df[df['classifier'] == clf]
        if len(clf_data) == 0:
            continue
        ax.scatter(clf_data['fsgm_acc'] * 100, clf_data['pgd_acc'] * 100,
                  c=COLORS[clf], marker=MARKERS[clf], s=100, label=f'{clf}',
                  edgecolors='white', linewidths=1, alpha=0.85, zorder=3)
    
    # Plot CNN baselines with larger markers
    for _, row in cnn_data.iterrows():
        clf = row['classifier']
        ax.scatter(row['fsgm_acc'] * 100, row['pgd_acc'] * 100,
                  c=COLORS[clf], marker=MARKERS[clf], s=200, 
                  label=f'{clf}', edgecolors='white', linewidths=2, zorder=4)
    
    # Diagonal
    ax.plot([0, 100], [0, 100], color=PALETTE['dark_gray'], linestyle='--', 
            alpha=0.6, linewidth=2, zorder=1)
    
    # Shaded regions
    ax.fill_between([0, 100], [0, 100], [100, 100], alpha=0.2, 
                   color=PALETTE['royal_blue'], zorder=0)
    ax.fill_between([0, 100], [0, 0], [0, 100], alpha=0.2, 
                   color=PALETTE['vermillion'], zorder=0)
    
    # Annotations
    ax.annotate('PGD less effective\nat transfer', 
               xy=(20, 60), fontsize=10, style='italic', 
               color=PALETTE['royal_blue'], fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        alpha=0.9, edgecolor=PALETTE['bluish_green']))
    ax.annotate('FGSM less effective\n(typical DL behavior)', 
               xy=(55, 20), fontsize=10, style='italic', 
               color=PALETTE['vermillion'], fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        alpha=0.9, edgecolor=PALETTE['vermillion']))
    
    ax.set_xlabel('Accuracy under FGSM Attack (%)', fontweight='bold')
    ax.set_ylabel('Accuracy under PGD Attack (%)', fontweight='bold')
    ax.set_xlim(0, 95)
    ax.set_ylim(0, 95)
    ax.legend(loc='upper left', fontsize=9, ncol=2, title='Classifier', title_fontsize=10)
    ax.set_aspect('equal')
    ax.xaxis.grid(True, linestyle='--', alpha=0.4)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    return fig


# =============================================================================
# FIGURE 3: BLOCK SIZE IMPACT (includes ALL classifiers including ANN)
# =============================================================================

def figure3_block_size(df):
    """Block size impact on ALL classifiers including ANN."""
    
    # Block experiments: 0 (B=1), 3 (B=2), 4 (B=3)
    block_data = df[df['experiment'].isin([0, 3, 4])].copy()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    classifiers = ['KNN', 'KSVM', 'LSVM', 'ANN']
    block_sizes = [1, 2, 3]
    
    # Left: FGSM accuracy trends
    ax = axes[0]
    for clf in classifiers:
        clf_data = block_data[block_data['classifier'] == clf].sort_values('block_size')
        if len(clf_data) < 3:
            continue
        
        ax.plot(clf_data['block_size'], clf_data['fsgm_acc'], 
                'o-', color=COLORS[clf], marker=MARKERS[clf], label=clf,
                linewidth=2.5, markersize=12, markeredgecolor='white', markeredgewidth=1.5)
    
    ax.set_xlabel('HOG Block Size (cells per block)', fontweight='bold')
    ax.set_ylabel('Accuracy under FGSM Attack', fontweight='bold')
    ax.set_xticks(block_sizes)
    ax.set_xlim(0.5, 3.5)
    ax.set_ylim(0.25, 0.70)
    ax.legend(loc='lower right', title='Classifier')
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_title('(a) FGSM Attack', fontsize=12, fontweight='bold', pad=10)
    
    # Right: PGD accuracy trends  
    ax = axes[1]
    for clf in classifiers:
        clf_data = block_data[block_data['classifier'] == clf].sort_values('block_size')
        if len(clf_data) < 3:
            continue
        
        ax.plot(clf_data['block_size'], clf_data['pgd_acc'], 
                'o-', color=COLORS[clf], marker=MARKERS[clf], label=clf,
                linewidth=2.5, markersize=12, markeredgecolor='white', markeredgewidth=1.5)
    
    ax.set_xlabel('HOG Block Size (cells per block)', fontweight='bold')
    ax.set_ylabel('Accuracy under PGD Attack', fontweight='bold')
    ax.set_xticks(block_sizes)
    ax.set_xlim(0.5, 3.5)
    ax.set_ylim(0.25, 0.80)
    ax.legend(loc='lower right', title='Classifier')
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_title('(b) PGD Attack', fontsize=12, fontweight='bold', pad=10)
    
    plt.tight_layout()
    return fig


# =============================================================================
# FIGURE 4: EPSILON SENSITIVITY (COMPLETE with both FGSM and PGD drops)
# =============================================================================

def figure4_epsilon_sensitivity(df):
    """Epsilon sensitivity - COMPLETE with both attack types."""
    
    c1 = df[df['experiment'] == 0]  # eps=4
    c8 = df[df['experiment'] == 7]  # eps=8
    
    classifiers = ['KNN', 'DT', 'LSVM', 'KSVM']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Get data
    eps4_fsgm = [c1[c1['classifier'] == c]['fsgm_acc'].values[0] for c in classifiers]
    eps8_fsgm = [c8[c8['classifier'] == c]['fsgm_acc'].values[0] for c in classifiers]
    eps4_pgd = [c1[c1['classifier'] == c]['pgd_acc'].values[0] for c in classifiers]
    eps8_pgd = [c8[c8['classifier'] == c]['pgd_acc'].values[0] for c in classifiers]
    
    x = np.arange(len(classifiers))
    width = 0.35
    
    # Left: FGSM comparison
    ax = axes[0]
    bars1 = ax.bar(x - width/2, eps4_fsgm, width, label='ε = 4/255', 
                   color=PALETTE['royal_blue'], edgecolor='white', linewidth=1)
    bars2 = ax.bar(x + width/2, eps8_fsgm, width, label='ε = 8/255', 
                   color=PALETTE['vermillion'], edgecolor='white', linewidth=1)
    
    # Drop annotations
    for i in range(len(classifiers)):
        drop = (eps4_fsgm[i] - eps8_fsgm[i]) * 100
        mid_x = x[i]
        mid_y = (eps4_fsgm[i] + eps8_fsgm[i]) / 2
        ax.annotate(f'−{drop:.0f}pp', xy=(mid_x + 0.2, mid_y + 0.02),
                   ha='center', fontsize=9, color=PALETTE['dark_gray'], fontweight='bold')
    
    ax.set_xlabel('Classifier', fontweight='bold')
    ax.set_ylabel('Accuracy under FGSM', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classifiers)
    ax.set_ylim(0, 0.45)
    ax.legend(loc='upper right', title='Perturbation Budget')
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_title('(a) FGSM Attack', fontsize=12, fontweight='bold', pad=10)
    
    # Right: PGD comparison
    ax = axes[1]
    bars1 = ax.bar(x - width/2, eps4_pgd, width, label='ε = 4/255', 
                   color=PALETTE['royal_blue'], edgecolor='white', linewidth=1)
    bars2 = ax.bar(x + width/2, eps8_pgd, width, label='ε = 8/255', 
                   color=PALETTE['vermillion'], edgecolor='white', linewidth=1)
    
    # Drop annotations
    for i in range(len(classifiers)):
        drop = (eps4_pgd[i] - eps8_pgd[i]) * 100
        mid_x = x[i]
        mid_y = (eps4_pgd[i] + eps8_pgd[i]) / 2
        ax.annotate(f'−{drop:.0f}pp', xy=(mid_x+0.2, mid_y + 0.02),
                   ha='center', fontsize=9, color=PALETTE['dark_gray'], fontweight='bold')
    
    ax.set_xlabel('Classifier', fontweight='bold')
    ax.set_ylabel('Accuracy under PGD', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classifiers)
    ax.set_ylim(0, 0.50)
    ax.legend(loc='upper right', title='Perturbation Budget')
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_title('(b) PGD Attack', fontsize=12, fontweight='bold', pad=10)
    
    plt.tight_layout()
    return fig


# =============================================================================
# FIGURE 5: RETENTION HEATMAP (polished)
# =============================================================================

def figure5_retention_heatmap(df):
    """FGSM Accuracy Retention Heatmap - comprehensive overview."""
    
    pivot = df.pivot_table(
        values='fsgm_retention',
        index='classifier',
        columns='experiment',
        aggfunc='mean'
    )
    
    clf_order = ['KNN', 'DT', 'LSVM', 'KSVM', 'ANN']
    clf_order = [c for c in clf_order if c in pivot.index]
    pivot = pivot.reindex(clf_order)
    
    config_labels = ['C1\nBaseline', 'C2\nCell=6', 'C3\nCell=10', 
                     'C4\nBlock=2', 'C5\nBlock=3', 'C6\nOrient=9', 
                     'C7\nOrient=3', 'C8\nε=8']
    
    fig, ax = plt.subplots(figsize=(10, 4.5))
    
    # Custom colormap: red (low) -> white (mid) -> blue (high)
    cmap = sns.diverging_palette(15, 220, s=80, l=55, as_cmap=True)
    
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap=cmap,
                center=50, vmin=15, vmax=85,
                cbar_kws={'label': 'Accuracy Retention (%)', 'shrink': 0.8},
                linewidths=1, linecolor='white', ax=ax,
                annot_kws={'fontsize': 11, 'fontweight': 'bold'})
    
    ax.set_xticklabels(config_labels, rotation=0, fontsize=10)
    ax.set_ylabel('Classifier', fontweight='bold')
    ax.set_xlabel('HOG Configuration', fontweight='bold')
    
    # Highlight best and worst columns
    ax.annotate('Best', xy=(4.5, -0.3), xycoords='data',
               ha='center', fontsize=10, color=PALETTE['bluish_green'], fontweight='bold')
    ax.annotate('Worst', xy=(7.5, -0.3), xycoords='data',
               ha='center', fontsize=10, color=PALETTE['vermillion'], fontweight='bold')
    
    plt.tight_layout()
    return fig


# =============================================================================
# FIGURE 6: PAIRED ATTACK COMPARISON (brought back from initial)
# =============================================================================

def figure6_paired_attack(df):
    """Paired FGSM vs PGD comparison per classifier across configurations."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    classifiers = ['KNN', 'KSVM', 'LSVM', 'ANN']
    
    for ax, clf in zip(axes, classifiers):
        clf_data = df[df['classifier'] == clf].sort_values('experiment')
        
        if len(clf_data) == 0:
            continue
        
        x = np.arange(len(clf_data))
        
        # Plot both attacks
        ax.plot(x, clf_data['fsgm_acc'], 'o-', color=ATTACK_COLORS['FGSM'], 
                label='FGSM', linewidth=2.5, markersize=10, markeredgecolor='white')
        ax.plot(x, clf_data['pgd_acc'], 's-', color=ATTACK_COLORS['PGD'], 
                label='PGD', linewidth=2.5, markersize=10, markeredgecolor='white')
        
        # Fill between
        ax.fill_between(x, clf_data['fsgm_acc'], clf_data['pgd_acc'], 
                       alpha=0.15, color=PALETTE['dark_gray'])
        
        # Clean reference
        ax.plot(x, clf_data['orig_acc'], '--', color=ATTACK_COLORS['Clean'], 
                alpha=0.6, label='Clean', linewidth=2)
        
        ax.set_title(f'{clf}', fontsize=14, fontweight='bold', color=COLORS[clf])
        ax.set_xlabel('Configuration', fontweight='bold')
        ax.set_ylabel('Accuracy', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'C{i+1}' for i in clf_data['experiment']], fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.legend(loc='upper right', fontsize=9)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)
    
    plt.tight_layout()
    return fig


# =============================================================================
# FIGURE 7: FEATURE CORRELATION HEATMAP (polished, novel insight)
# =============================================================================

def figure7_correlation_heatmap(df):
    """Feature correlation heatmap - shows relationships between parameters and robustness."""
    
    # Only use complete data
    valid_df = df[df['classifier'].isin(['KNN', 'DT', 'LSVM', 'KSVM'])].copy()
    
    # Select relevant columns
    corr_cols = ['cell_size', 'orientations', 'block_size', 'epsilon',
                 'orig_acc', 'fsgm_acc', 'pgd_acc', 'fsgm_retention', 'pgd_retention']
    
    corr_matrix = valid_df[corr_cols].corr()
    
    # Rename for display
    display_names = {
        'cell_size': 'Cell Size',
        'orientations': 'Orientations',
        'block_size': 'Block Size',
        'epsilon': 'Epsilon (ε)',
        'orig_acc': 'Clean Acc.',
        'fsgm_acc': 'FGSM Acc.',
        'pgd_acc': 'PGD Acc.',
        'fsgm_retention': 'FGSM Ret.',
        'pgd_retention': 'PGD Ret.'
    }
    
    corr_matrix.index = [display_names[c] for c in corr_matrix.index]
    corr_matrix.columns = [display_names[c] for c in corr_matrix.columns]
    
    fig, ax = plt.subplots(figsize=(9, 8))
    
    # Mask upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    # Custom colormap
    cmap = sns.diverging_palette(220, 15, s=80, l=55, as_cmap=True)
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                cmap=cmap, center=0, vmin=-1, vmax=1,
                square=True, linewidths=1, linecolor='white',
                cbar_kws={'label': 'Pearson Correlation (r)', 'shrink': 0.7},
                ax=ax, annot_kws={'fontsize': 10, 'fontweight': 'bold'})
    
    ax.set_title('Parameter-Robustness Correlation Matrix', fontsize=13, 
                fontweight='bold', pad=15)
    
    # Add explanatory note
    ax.text(0.5, -0.22, 
           'Note: r = Pearson correlation coefficient (−1 to +1). '
           'Computed across 32 classifier-configuration pairs.',
           transform=ax.transAxes, fontsize=9, style='italic',
           ha='center', color=PALETTE['dark_gray'])
    
    plt.tight_layout()
    return fig


# =============================================================================
# FIGURE 8: RADAR COMPARISON (includes ANN)
# =============================================================================

def figure8_radar(df, cnn_data):
    """Radar chart - includes KNN, KSVM, ANN, and AlexNet."""
    
    c5 = df[df['experiment'] == 4]
    
    metrics = ['Clean\nAccuracy', 'FGSM\nAccuracy', 'PGD\nAccuracy', 
               'FGSM\nRetention', 'PGD\nRetention']
    
    # Include ANN
    models_to_plot = ['KNN', 'KSVM', 'ANN', 'AlexNet']
    data = {}
    
    for model in models_to_plot:
        if model == 'AlexNet':
            row = cnn_data[cnn_data['classifier'] == model].iloc[0]
            fsgm_ret = row['fsgm_acc'] / row['orig_acc']
            pgd_ret = row['pgd_acc'] / row['orig_acc']
            print("Alex Retention:", fsgm_ret, pgd_ret)
        else:
            model_data = c5[c5['classifier'] == model]
            if len(model_data) == 0:
                model_data = df[(df['experiment'] == 3) & (df['classifier'] == model)]
            if len(model_data) == 0:
                continue
            row = model_data.iloc[0]
            fsgm_ret = row['fsgm_retention'] / 100
            pgd_ret = row['pgd_retention'] / 100
        
        data[model] = [
            row['orig_acc'],
            row['fsgm_acc'],
            row['pgd_acc'],
            fsgm_ret,
            pgd_ret
        ]
    
    N = len(metrics)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 7), subplot_kw=dict(polar=True))
    
    for model in models_to_plot:
        if model not in data:
            continue
        values = data[model]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2.5, label=model,
                color=COLORS[model], markersize=8, markeredgecolor='white')
        ax.fill(angles, values, alpha=0.15, color=COLORS[model])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], fontsize=9)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.05), 
             title='Model', title_fontsize=11, fontsize=10)
    
    ax.set_facecolor(PALETTE['light_gray'])
    ax.grid(True, linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    return fig


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Generate all 8 polished figures."""
    print("="*70)
    print("ADVERSARIAL TRANSFERABILITY - FINAL POLISHED FIGURES")
    print("="*70)
    print("\nColor Palette: Okabe-Ito based (colorblind-friendly)")
    print("Output Formats: SVG (vector), PDF, PNG")
    print()
    
    df, cnn_data = load_all_data()
    
    figures = [
        ("fig1", figure1_cross_paradigm, 
         "Cross-paradigm comparison (all models)"),
        ("fig2", figure2_fgsm_vs_pgd, 
         "FGSM vs PGD scatter (surprising reversal)"),
        ("fig3", figure3_block_size, 
         "Block size impact (all classifiers + ANN)"),
        ("fig4", figure4_epsilon_sensitivity, 
         "Epsilon sensitivity (FGSM + PGD drops)"),
        ("fig5", figure5_retention_heatmap, 
         "Retention heatmap (comprehensive)"),
        ("fig6", figure6_paired_attack, 
         "Paired attack comparison (per classifier)"),
        ("fig7", figure7_correlation_heatmap, 
         "Feature correlation matrix (novel insight)"),
        ("fig8", figure8_radar, 
         "Radar comparison (KNN, KSVM, ANN, AlexNet)"),
    ]
    
    for name, func, description in figures:
        print(f"[{figures.index((name, func, description))+1}/8] {description}")
        try:
            if 'cnn_data' in func.__code__.co_varnames:
                fig = func(df, cnn_data)
            else:
                fig = func(df)
            save_figure(fig, name)
            plt.close(fig)
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("STATISTICAL TERMINOLOGY EXPLANATION")
    print("="*70)
    print("""
    r (Pearson correlation coefficient):
      - Measures strength of LINEAR relationship between two variables
      - Range: -1 (perfect negative) to +1 (perfect positive)
      - r ≈ 0.4 = moderate positive correlation
      - r ≈ 0.6 = strong positive correlation
    
    p (p-value):
      - Probability that the observed correlation occurred by chance
      - p < 0.05 = statistically significant (95% confidence)
      - p < 0.01 = highly significant (99% confidence)
      - p < 0.001 = very highly significant
    
    IMPORTANT: In this study, correlations are computed across 32 
    classifier-configuration pairs, measuring SENSITIVITY TO CONFIGURATION
    CHANGES, not measurement uncertainty from repeated trials.
    """)
    
    print("="*70)
    print("All figures saved to /home/claude/final_figures/")
    print("="*70)


if __name__ == "__main__":
    main()
