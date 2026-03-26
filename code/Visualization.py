import matplotlib.pyplot as plt
import numpy as np

def smooth(data, window):
    if window <= 1 or len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')

def plot_accuracy_comparison(*acc_lists, 
                            labels=None, 
                            title="Accuracy Comparison",
                            smoothing_window=5,
                            colors=None,
                            figsize=(12, 6)):
    plt.figure(figsize=figsize)
    
    default_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 
                      'pink', 'gray', 'olive', 'cyan', 'magenta', 'gold']
    
    if colors is None:
        colors = default_colors[:len(acc_lists)]
    
    if labels is None:
        labels = [f"Experiment {i+1}" for i in range(len(acc_lists))]
    
    for i, (acc, label, color) in enumerate(zip(acc_lists, labels, colors)):
        if smoothing_window > 1 and len(acc) >= smoothing_window:
            acc_smooth = smooth(acc, smoothing_window)
            x = range(smoothing_window-1, len(acc))
            plt.plot(x, acc_smooth, color=color, linewidth=2, 
                    label=f"{label} (smoothed)")
            
            plt.plot(acc, color=color, alpha=0.3, linewidth=1)
        else:
            plt.plot(acc, color=color, linewidth=2, label=label)
    
    plt.xlabel("Meta-iteration", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 110])
    
    plt.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

def plot_meta_loss_comparison(*loss_lists, 
                              labels=None, 
                              title="Meta-Loss Comparison",
                              smoothing_window=5,
                              colors=None,
                              figsize=(12, 6)):
    plt.figure(figsize=figsize)
    
    default_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 
                      'pink', 'gray', 'olive', 'cyan', 'magenta', 'gold']
    
    if colors is None:
        colors = default_colors[:len(loss_lists)]
    
    if labels is None:
        labels = [f"Experiment {i+1}" for i in range(len(loss_lists))]
    
    for i, (loss, label, color) in enumerate(zip(loss_lists, labels, colors)):
        if smoothing_window > 1 and len(loss) >= smoothing_window:
            loss_smooth = smooth(loss, smoothing_window)
            x = range(smoothing_window-1, len(loss))
            plt.plot(x, loss_smooth, color=color, linewidth=2, 
                    label=f"{label} (smoothed)")

            plt.plot(loss, color=color, alpha=0.3, linewidth=1)
        else:
            plt.plot(loss, color=color, linewidth=2, label=label)
    
    plt.xlabel("Meta-iteration", fontsize=12)
    plt.ylabel("Meta-Loss", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()