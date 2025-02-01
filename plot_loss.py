import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

def get_tensorboard_data(log_dir='out/logs'):
    train_data = {}  # Using dict to ensure unique iterations
    val_data = {}
    lr_data = {}
    
    for event_file in os.listdir(log_dir):
        if event_file.startswith('events.out.tfevents'):
            path = os.path.join(log_dir, event_file)
            for event in tf.compat.v1.train.summary_iterator(path):
                for value in event.summary.value:
                    if value.tag == 'train_loss':
                        train_data[event.step] = value.simple_value
                    elif value.tag == 'val_loss':
                        val_data[event.step] = value.simple_value
                    elif value.tag == 'lr':
                        lr_data[event.step] = value.simple_value
    
    # Sort by iteration number
    train_steps = sorted(train_data.keys())
    val_steps = sorted(val_data.keys())
    lr_steps = sorted(lr_data.keys())
    
    train_values = [train_data[step] for step in train_steps]
    val_values = [val_data[step] for step in val_steps]
    lr_values = [lr_data[step] for step in lr_steps]
    
    return (np.array(train_steps), np.array(train_values),
            np.array(val_steps), np.array(val_values),
            np.array(lr_steps), np.array(lr_values))

def plot_training_progress(output_file='loss_curve.png'):
    try:
        # Get data
        train_steps, train_values, val_steps, val_values, lr_steps, lr_values = get_tensorboard_data()
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])
        
        # Plot losses with smoothing
        window = 5  # Reduced smoothing window
        train_smooth = np.convolve(train_values, np.ones(window)/window, mode='valid')
        val_smooth = np.convolve(val_values, np.ones(window)/window, mode='valid')
        
        ax1.plot(train_steps[window-1:], train_smooth, label='Training Loss', alpha=0.7)
        ax1.plot(val_steps[window-1:], val_smooth, label='Validation Loss', alpha=0.7)
        ax1.set_title('Training Progress')
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot learning rate
        ax2.plot(lr_steps, lr_values, label='Learning Rate', color='green', alpha=0.7)
        ax2.set_xlabel('Iterations')
        ax2.set_ylabel('Learning Rate')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Loss curve saved as {output_file}")
        
    except Exception as e:
        print(f"Error reading TensorBoard logs: {e}")
        print("Make sure tensorflow is installed: pip install tensorflow")

def save_text_representation(output_file='training_data.txt'):
    try:
        train_steps, train_values, val_steps, val_values, lr_steps, lr_values = get_tensorboard_data()
        
        with open(output_file, 'w') as f:
            f.write("Training Progress Data\n")
            f.write("=====================\n\n")
            
            f.write("Format: Iteration | Training Loss | Validation Loss | Learning Rate\n")
            f.write("-" * 65 + "\n\n")
            
            # Create aligned data points
            all_steps = sorted(set(train_steps) | set(val_steps) | set(lr_steps))
            for step in all_steps:
                train_val = train_values[train_steps == step][0] if step in train_steps else "---"
                val_val = val_values[val_steps == step][0] if step in val_steps else "---"
                lr_val = lr_values[lr_steps == step][0] if step in lr_steps else "---"
                
                f.write(f"Iter {step:5d}: ")
                f.write(f"train={train_val:<10.4f} " if train_val != "---" else "train=---        ")
                f.write(f"val={val_val:<10.4f} " if val_val != "---" else "val=---        ")
                f.write(f"lr={lr_val:.2e}\n" if lr_val != "---" else "lr=---\n")
            
            # Add summary statistics
            f.write("\nSummary Statistics\n")
            f.write("=================\n")
            f.write(f"Total Iterations: {max(all_steps)}\n")
            f.write(f"Final Training Loss: {train_values[-1]:.4f}\n")
            f.write(f"Final Validation Loss: {val_values[-1]:.4f}\n")
            f.write(f"Final Learning Rate: {lr_values[-1]:.2e}\n")
            
            # Add loss trends
            f.write("\nLoss Trends\n")
            f.write("===========\n")
            f.write(f"Initial Training Loss: {train_values[0]:.4f}\n")
            f.write(f"Best Training Loss: {min(train_values):.4f}\n")
            f.write(f"Initial Validation Loss: {val_values[0]:.4f}\n")
            f.write(f"Best Validation Loss: {min(val_values):.4f}\n")
        
        print(f"Text representation saved as {output_file}")
        
    except Exception as e:
        print(f"Error creating text representation: {e}")

if __name__ == "__main__":
    plot_training_progress()  # Create the plot as before
    save_text_representation()  # Also save text representation 