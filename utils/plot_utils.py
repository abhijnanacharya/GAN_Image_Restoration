import matplotlib.pyplot as plt

def plot_training_progress(stats, output_dir, current_epoch):
    """
    Create and save comprehensive training progress plots.
    Args:
        stats: TrainingStats object containing loss and score data.
        output_dir: Directory to save the plot.
        current_epoch: Current epoch number.
    """
    plt.figure(figsize=(15, 10))

    # Plot Generator and Discriminator Losses
    plt.subplot(2, 1, 1)
    epochs = range(1, len(stats.epoch_g_losses) + 1)
    plt.plot(epochs, stats.epoch_g_losses, 'b-', label='Generator Loss')
    plt.plot(epochs, stats.epoch_d_losses, 'r-', label='Discriminator Loss')
    plt.title(f'Training Losses (Epoch {current_epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Real and Fake Scores
    plt.subplot(2, 1, 2)
    plt.plot(stats.real_scores[-100:], 'g-', label='Real Score (D(x))')
    plt.plot(stats.fake_scores[-100:], 'm-', label='Fake Score (D(G(z)))')
    plt.title('Discriminator Scores')
    plt.xlabel('Batch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)

    # Save plot
    plot_path = f"{output_dir}/training_progress_epoch_{current_epoch}.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()