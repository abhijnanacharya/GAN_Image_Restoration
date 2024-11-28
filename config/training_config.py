class TrainingConfig:
    def __init__(self):
        self.num_epochs = 5
        self.batch_size = 1
        self.lr = 0.0002
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.lambda_pixel = 100  # Weight for pixel-wise loss
        self.sample_interval = 10  # Save samples every 10 batches
        self.checkpoint_interval = 1  # Save checkpoints every epoch