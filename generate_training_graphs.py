#!/usr/bin/env python3
"""
Training Progress Graph Generator
Generates comprehensive training graphs from TensorBoard logs and saves them alongside feature.npy
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import glob

# Set style for better-looking graphs
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

def load_training_data(data_dir, output_dir):
    """Load training data from numpy files and copy them to output directory"""
    data_files = {
        'losses_P': 'losses_P.npy',
        'losses_D': 'losses_D.npy', 
        'acces_P': 'acces_P.npy',
        'acces_D': 'acces_D.npy'
    }
    
    data = {}
    for key, filename in data_files.items():
        # First try in the specified directory
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            data[key] = np.load(filepath)
            print(f"Loaded {key}: {data[key].shape}")
        else:
            # Try in the root fidip folder
            root_filepath = os.path.join(data_dir.parent.parent.parent, filename)
            if os.path.exists(root_filepath):
                data[key] = np.load(root_filepath)
                print(f"Loaded {key} from root directory: {data[key].shape}")
                
                # Copy the file to the output directory
                import shutil
                output_filepath = os.path.join(output_dir, filename)
                shutil.copy2(root_filepath, output_filepath)
                print(f"Copied {filename} to {output_dir}")
            else:
                print(f"Warning: {filename} not found in {data_dir} or root directory")
                data[key] = None
    
    return data

def create_training_curves(data, output_dir, experiment_name="Training"):
    """Create comprehensive training curves"""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{experiment_name} - Training Progress Analysis', fontsize=16, fontweight='bold')
    
    # 1. Loss Curves
    ax1 = axes[0, 0]
    if data['losses_P'] is not None:
        epochs = range(len(data['losses_P']))
        ax1.plot(epochs, data['losses_P'], 'b-', linewidth=2, label='Pose Network Loss', marker='o', markersize=4)
    if data['losses_D'] is not None:
        ax1.plot(epochs, data['losses_D'], 'r-', linewidth=2, label='Domain Classifier Loss', marker='s', markersize=4)
    
    ax1.set_title('Training Losses Over Time', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 2. Accuracy Curves
    ax2 = axes[0, 1]
    if data['acces_P'] is not None:
        ax2.plot(epochs, data['acces_P'], 'g-', linewidth=2, label='Pose Network Accuracy', marker='o', markersize=4)
    if data['acces_D'] is not None:
        ax2.plot(epochs, data['acces_D'], 'm-', linewidth=2, label='Domain Classifier Accuracy', marker='s', markersize=4)
    
    ax2.set_title('Training Accuracies Over Time', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # 3. Loss Comparison (if both available)
    ax3 = axes[1, 0]
    if data['losses_P'] is not None and data['losses_D'] is not None:
        # Normalize losses for comparison
        norm_losses_P = (data['losses_P'] - data['losses_P'].min()) / (data['losses_P'].max() - data['losses_P'].min())
        norm_losses_D = (data['losses_D'] - data['losses_D'].min()) / (data['losses_D'].max() - data['losses_D'].min())
        
        ax3.plot(epochs, norm_losses_P, 'b-', linewidth=2, label='Pose Network (Normalized)', alpha=0.8)
        ax3.plot(epochs, norm_losses_D, 'r-', linewidth=2, label='Domain Classifier (Normalized)', alpha=0.8)
        ax3.set_title('Normalized Loss Comparison', fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Normalized Loss')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Insufficient data for comparison', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Loss Comparison (Data Unavailable)', fontweight='bold')
    
    # 4. Training Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create summary text
    summary_text = f"""
    Training Summary:
    
    Total Epochs: {len(data['losses_P']) if data['losses_P'] is not None else 'N/A'}
    
    Final Metrics:
    • Pose Network Loss: {data['losses_P'][-1]:.6f} (Final)
    • Domain Classifier Loss: {data['losses_D'][-1]:.6f} (Final)
    • Pose Network Accuracy: {data['acces_P'][-1]:.3f} (Final)
    • Domain Classifier Accuracy: {data['acces_D'][-1]:.3f} (Final)
    
    Training Progress:
    • Loss Reduction: {((data['losses_P'][0] - data['losses_P'][-1]) / data['losses_P'][0] * 100):.1f}% (Pose)
    • Accuracy Improvement: {((data['acces_P'][-1] - data['acces_P'][0]) / data['acces_P'][0] * 100):.1f}% (Pose)
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10, 
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, 'training_progress_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Training progress graph saved to: {output_path}")
    
    return output_path

def create_domain_adaptation_analysis(data, output_dir, experiment_name="Domain Adaptation"):
    """Create domain adaptation specific analysis"""
    
    if data['acces_D'] is None:
        print("Warning: Domain classifier accuracy data not available for domain adaptation analysis")
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'{experiment_name} - Domain Classifier Analysis', fontsize=16, fontweight='bold')
    
    epochs = range(len(data['acces_D']))
    
    # 1. Domain Classifier Accuracy Trend
    ax1 = axes[0]
    ax1.plot(epochs, data['acces_D'], 'r-', linewidth=3, label='Domain Classifier Accuracy', marker='o', markersize=5)
    ax1.axhline(y=0.5, color='k', linestyle='--', alpha=0.7, label='Random (0.5)')
    ax1.set_title('Domain Classifier Performance', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Add trend analysis
    if len(data['acces_D']) > 1:
        trend = np.polyfit(epochs, data['acces_D'], 1)
        trend_line = np.poly1d(trend)
        ax1.plot(epochs, trend_line(epochs), 'k--', alpha=0.7, label=f'Trend (slope: {trend[0]:.4f})')
        ax1.legend()
    
    # 2. Domain vs Pose Network Comparison
    ax2 = axes[1]
    if data['acces_P'] is not None:
        ax2.plot(epochs, data['acces_D'], 'r-', linewidth=2, label='Domain Classifier', marker='o', markersize=4)
        ax2.plot(epochs, data['acces_P'], 'b-', linewidth=2, label='Pose Network', marker='s', markersize=4)
        ax2.set_title('Domain vs Pose Network Accuracy', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
    else:
        ax2.text(0.5, 0.5, 'Pose Network data unavailable', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Domain vs Pose Network (Data Unavailable)', fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, 'domain_adaptation_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Domain adaptation analysis saved to: {output_path}")
    
    return output_path

def create_loss_heatmap(data, output_dir):
    """Create a heatmap showing loss patterns"""
    
    if data['losses_P'] is None or data['losses_D'] is None:
        print("Warning: Insufficient data for heatmap")
        return None
    
    # Create a matrix of losses for heatmap
    epochs = range(len(data['losses_P']))
    
    # Normalize the data
    losses_matrix = np.array([data['losses_P'], data['losses_D']])
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Create heatmap
    im = ax.imshow(losses_matrix, cmap='YlOrRd', aspect='auto')
    
    # Set labels
    ax.set_xticks(range(0, len(epochs), max(1, len(epochs)//10)))
    ax.set_xticklabels([f'E{i}' for i in range(0, len(epochs), max(1, len(epochs)//10))])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Pose Network Loss', 'Domain Classifier Loss'])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Loss Value', rotation=270, labelpad=20)
    
    ax.set_title('Training Loss Heatmap', fontweight='bold')
    ax.set_xlabel('Epoch')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, 'loss_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Loss heatmap saved to: {output_path}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Generate training progress graphs from feature.npy directory')
    parser.add_argument('feature_path', help='Path to feature.npy file')
    parser.add_argument('--experiment-name', default='FiDIP Training', help='Name for the experiment')
    parser.add_argument('--output-dir', help='Output directory (defaults to same as feature.npy)')
    
    args = parser.parse_args()
    
    # Get the directory containing feature.npy
    feature_path = Path(args.feature_path)
    if not feature_path.exists():
        print(f"Error: feature.npy not found at {feature_path}")
        sys.exit(1)
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = feature_path.parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading training data from: {feature_path.parent}")
    print(f"Saving graphs to: {output_dir}")
    
    # Load training data and copy files to output directory
    data = load_training_data(feature_path.parent, output_dir)
    
    # Check if we have any data
    if all(v is None for v in data.values()):
        print("Error: No training data files found!")
        print("Looking for: losses_P.npy, losses_D.npy, acces_P.npy, acces_D.npy")
        sys.exit(1)
    
    # Generate graphs
    print("\nGenerating training progress graphs...")
    
    # 1. Main training curves
    training_graph = create_training_curves(data, output_dir, args.experiment_name)
    
    # 2. Domain adaptation analysis
    domain_graph = create_domain_adaptation_analysis(data, output_dir, args.experiment_name)
    
    # 3. Loss heatmap
    heatmap_graph = create_loss_heatmap(data, output_dir)
    
    print(f"\n✅ Graph generation complete!")
    print(f"Generated files:")
    if training_graph:
        print(f"  • {training_graph}")
    if domain_graph:
        print(f"  • {domain_graph}")
    if heatmap_graph:
        print(f"  • {heatmap_graph}")

if __name__ == "__main__":
    main()
