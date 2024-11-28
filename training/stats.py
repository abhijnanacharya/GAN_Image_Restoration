import numpy as np

class TrainingStats:
    def __init__(self):
        self.epoch_g_losses = []
        self.epoch_d_losses = []
        self.batch_g_losses = []
        self.batch_d_losses = []
        self.real_scores = []
        self.fake_scores = []

    def update_batch_stats(self, g_loss, d_loss, real_score, fake_score):
        """Update statistics for a single batch."""
        self.batch_g_losses.append(g_loss)
        self.batch_d_losses.append(d_loss)
        self.real_scores.append(real_score)
        self.fake_scores.append(fake_score)

    def compute_epoch_stats(self):
        """Compute and store average statistics for an epoch."""
        self.epoch_g_losses.append(np.mean(self.batch_g_losses))
        self.epoch_d_losses.append(np.mean(self.batch_d_losses))
        # Clear batch stats for the next epoch
        self.batch_g_losses.clear()
        self.batch_d_losses.clear()
        self.real_scores.clear()
        self.fake_scores.clear()