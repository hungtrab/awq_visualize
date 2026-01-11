"""
Visualization utilities for AWQ + Marlin demos
==============================================

Provides functions to visualize:
- Weight distributions
- Quantization errors
- Performance comparisons
- Memory bandwidth utilization
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
import pandas as pd

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class VisualizationHelper:
    """Helper class for creating visualizations"""
    
    def __init__(self, output_dir: str = "."):
        self.output_dir = output_dir
        
    def plot_weight_distribution(self, 
                                weights_dict: Dict[str, np.ndarray],
                                title: str = "Weight Distribution",
                                filename: Optional[str] = None):
        """
        Plot weight distributions for multiple layers/states
        
        Args:
            weights_dict: Dictionary of {name: weights_array}
            title: Plot title
            filename: Output filename (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(weights_dict)))
        
        # Plot each distribution
        for idx, ((name, weights), color) in enumerate(zip(weights_dict.items(), colors)):
            if idx >= 4:  # Max 4 subplots
                break
                
            ax = axes[idx]
            w_flat = weights.flatten()
            
            # Histogram
            ax.hist(w_flat, bins=100, alpha=0.7, color=color, edgecolor='black')
            ax.set_title(f'{name}\n(μ={np.mean(w_flat):.4f}, σ={np.std(w_flat):.4f})')
            ax.set_xlabel('Weight Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            
            # Add vertical lines for mean and ±σ
            mean = np.mean(w_flat)
            std = np.std(w_flat)
            ax.axvline(mean, color='red', linestyle='--', linewidth=2, label='Mean')
            ax.axvline(mean + std, color='orange', linestyle=':', linewidth=1.5, label='±1σ')
            ax.axvline(mean - std, color='orange', linestyle=':', linewidth=1.5)
            ax.legend()
        
        # Hide unused subplots
        for idx in range(len(weights_dict), 4):
            axes[idx].axis('off')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if filename:
            plt.savefig(f"{self.output_dir}/{filename}", dpi=150, bbox_inches='tight')
            print(f"Saved: {filename}")
        
        return fig
    
    def plot_quantization_comparison(self,
                                    original: np.ndarray,
                                    quantized: np.ndarray,
                                    filename: Optional[str] = None):
        """
        Create comprehensive quantization comparison plots
        
        Args:
            original: Original FP16 weights
            quantized: Quantized (dequantized) weights
            filename: Output filename
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        orig_flat = original.flatten()
        quant_flat = quantized.flatten()
        error = orig_flat - quant_flat
        
        # 1. Original distribution
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(orig_flat, bins=100, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_title('Original Weights (FP16)')
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # 2. Quantized distribution
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(quant_flat, bins=100, alpha=0.7, color='red', edgecolor='black')
        ax2.set_title('Quantized Weights (INT4)')
        ax2.set_xlabel('Value')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # 3. Overlay
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.hist(orig_flat, bins=100, alpha=0.5, color='blue', label='Original')
        ax3.hist(quant_flat, bins=100, alpha=0.5, color='red', label='Quantized')
        ax3.set_title('Overlay Comparison')
        ax3.set_xlabel('Value')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Scatter plot
        ax4 = fig.add_subplot(gs[1, :2])
        sample_size = min(10000, len(orig_flat))
        indices = np.random.choice(len(orig_flat), sample_size, replace=False)
        ax4.scatter(orig_flat[indices], quant_flat[indices], alpha=0.3, s=1, c='green')
        
        # Add perfect correlation line
        min_val = min(orig_flat.min(), quant_flat.min())
        max_val = max(orig_flat.max(), quant_flat.max())
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect match')
        ax4.set_title('Original vs Quantized (Scatter)')
        ax4.set_xlabel('Original Weight')
        ax4.set_ylabel('Quantized Weight')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Error distribution
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.hist(error, bins=100, alpha=0.7, color='purple', edgecolor='black')
        ax5.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
        ax5.set_title('Quantization Error')
        ax5.set_xlabel('Error (Original - Quantized)')
        ax5.set_ylabel('Frequency')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Error statistics box
        ax6 = fig.add_subplot(gs[2, 0])
        ax6.axis('off')
        
        mse = np.mean(error ** 2)
        mae = np.mean(np.abs(error))
        max_error = np.max(np.abs(error))
        
        stats_text = f"""
        Quantization Quality Metrics:
        
        MSE:           {mse:.6f}
        MAE:           {mae:.6f}
        Max Error:     {max_error:.6f}
        Error Std:     {np.std(error):.6f}
        
        Relative Error: {(mse / np.mean(orig_flat**2)) * 100:.2f}%
        SNR:           {10 * np.log10(np.mean(orig_flat**2) / mse):.2f} dB
        """
        
        ax6.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 7. Q-Q plot
        ax7 = fig.add_subplot(gs[2, 1])
        from scipy import stats
        stats.probplot(error, dist="norm", plot=ax7)
        ax7.set_title('Q-Q Plot (Error vs Normal)')
        ax7.grid(True, alpha=0.3)
        
        # 8. Heatmap of error by percentile
        ax8 = fig.add_subplot(gs[2, 2])
        
        # Bin errors by original value magnitude
        percentiles = np.percentile(np.abs(orig_flat), np.linspace(0, 100, 11))
        binned_errors = []
        for i in range(len(percentiles) - 1):
            mask = (np.abs(orig_flat) >= percentiles[i]) & (np.abs(orig_flat) < percentiles[i+1])
            if mask.any():
                binned_errors.append(np.mean(np.abs(error[mask])))
            else:
                binned_errors.append(0)
        
        ax8.bar(range(len(binned_errors)), binned_errors, color='coral')
        ax8.set_title('Mean Absolute Error by Magnitude')
        ax8.set_xlabel('Percentile Bin')
        ax8.set_ylabel('Mean |Error|')
        ax8.grid(True, alpha=0.3)
        
        plt.suptitle('Comprehensive Quantization Analysis', fontsize=16, fontweight='bold')
        
        if filename:
            plt.savefig(f"{self.output_dir}/{filename}", dpi=150, bbox_inches='tight')
            print(f"Saved: {filename}")
        
        return fig
    
    def plot_performance_comparison(self,
                                   results: Dict[str, Dict[str, float]],
                                   metric: str = "latency_ms",
                                   filename: Optional[str] = None):
        """
        Plot performance comparison across different methods
        
        Args:
            results: Dict of {method_name: {metric: value}}
            metric: Metric to plot
            filename: Output filename
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        methods = list(results.keys())
        values = [results[m].get(metric, 0) for m in methods]
        
        # Bar plot
        bars = ax1.bar(methods, values, color=plt.cm.viridis(np.linspace(0, 1, len(methods))))
        ax1.set_ylabel(metric.replace('_', ' ').title())
        ax1.set_title(f'Performance Comparison: {metric}')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # Speedup relative to first method (baseline)
        if len(methods) > 1:
            baseline = values[0]
            speedups = [baseline / v if v > 0 else 0 for v in values]
            
            bars2 = ax2.bar(methods, speedups, color=plt.cm.plasma(np.linspace(0, 1, len(methods))))
            ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Baseline')
            ax2.set_ylabel('Speedup (relative to baseline)')
            ax2.set_title(f'Speedup Analysis')
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}x', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(f"{self.output_dir}/{filename}", dpi=150, bbox_inches='tight')
            print(f"Saved: {filename}")
        
        return fig
    
    def plot_batch_size_scaling(self,
                               batch_sizes: List[int],
                               throughputs: Dict[str, List[float]],
                               filename: Optional[str] = None):
        """
        Plot throughput scaling with batch size
        
        Args:
            batch_sizes: List of batch sizes
            throughputs: Dict of {method_name: [throughputs]}
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        
        for method, values in throughputs.items():
            ax.plot(batch_sizes, values, marker='o', linewidth=2, 
                   markersize=8, label=method)
        
        ax.set_xlabel('Batch Size', fontsize=12)
        ax.set_ylabel('Throughput (tokens/s)', fontsize=12)
        ax.set_title('Throughput Scaling with Batch Size', fontsize=14, fontweight='bold')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=11)
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(f"{self.output_dir}/{filename}", dpi=150, bbox_inches='tight')
            print(f"Saved: {filename}")
        
        return fig
    
    def plot_memory_bandwidth(self,
                            memory_stats: Dict[str, float],
                            theoretical_bw: float = 1555,  # GB/s for A100
                            filename: Optional[str] = None):
        """
        Visualize memory bandwidth utilization
        
        Args:
            memory_stats: Dict of {operation: bandwidth_GB/s}
            theoretical_bw: Theoretical peak bandwidth
            filename: Output filename
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        operations = list(memory_stats.keys())
        bandwidths = list(memory_stats.values())
        utilizations = [bw / theoretical_bw * 100 for bw in bandwidths]
        
        # Bandwidth bar chart
        bars1 = ax1.bar(operations, bandwidths, color='steelblue')
        ax1.axhline(y=theoretical_bw, color='red', linestyle='--', 
                   linewidth=2, label=f'Peak BW ({theoretical_bw} GB/s)')
        ax1.set_ylabel('Bandwidth (GB/s)', fontsize=12)
        ax1.set_title('Memory Bandwidth', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom')
        
        # Utilization percentage
        bars2 = ax2.bar(operations, utilizations, color='coral')
        ax2.axhline(y=100, color='red', linestyle='--', linewidth=2, label='100% Utilization')
        ax2.set_ylabel('Utilization (%)', fontsize=12)
        ax2.set_title('Memory Bandwidth Utilization', fontsize=14)
        ax2.set_ylim([0, 110])
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(f"{self.output_dir}/{filename}", dpi=150, bbox_inches='tight')
            print(f"Saved: {filename}")
        
        return fig


def create_roofline_plot(compute_intensity: np.ndarray,
                         performance: np.ndarray,
                         peak_compute: float = 312,  # TFLOPS
                         peak_bandwidth: float = 1555,  # GB/s
                         filename: Optional[str] = None):
    """
    Create a roofline model plot
    
    Args:
        compute_intensity: Arithmetic intensity (FLOP/Byte)
        performance: Achieved performance (TFLOPS)
        peak_compute: Peak compute performance (TFLOPS)
        peak_bandwidth: Peak memory bandwidth (GB/s)
        filename: Output filename
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Roofline curve
    ci_range = np.logspace(-2, 3, 1000)
    
    # Memory-bound region
    memory_roof = peak_bandwidth * ci_range / 1000  # Convert to TFLOPS
    
    # Compute-bound region
    compute_roof = np.ones_like(ci_range) * peak_compute
    
    # Actual roofline (minimum of both)
    roofline = np.minimum(memory_roof, compute_roof)
    
    # Plot roofline
    ax.loglog(ci_range, roofline, 'k-', linewidth=3, label='Roofline')
    ax.loglog(ci_range, memory_roof, 'b--', linewidth=1, alpha=0.5, label='Memory Bound')
    ax.axhline(y=peak_compute, color='r', linestyle='--', linewidth=1, 
              alpha=0.5, label='Compute Bound')
    
    # Plot actual performance points
    ax.loglog(compute_intensity, performance, 'go', markersize=10, 
             label='Measured Performance', zorder=5)
    
    # Shaded regions
    ax.fill_between(ci_range, 0, memory_roof, where=(memory_roof < compute_roof),
                    color='blue', alpha=0.1, label='Memory Bound Region')
    ax.fill_between(ci_range, 0, compute_roof, where=(memory_roof >= compute_roof),
                    color='red', alpha=0.1, label='Compute Bound Region')
    
    ax.set_xlabel('Arithmetic Intensity (FLOP/Byte)', fontsize=12)
    ax.set_ylabel('Performance (TFLOPS)', fontsize=12)
    ax.set_title('Roofline Model Analysis', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
    
    return fig
