import torch
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import random
from utils import Logger
import numpy as np

# Create a dummy model
model = torch.nn.Linear(10, 2)

# Initialize logger
logger = Logger(model, log_dir=Path("output/demo_logs"))
r = 5
    
for epoch in range(100):
    # Generate random metrics
    logger.write_dict(epoch, 100, epoch*np.sin(epoch/r),
                       epoch*np.cos(epoch/r), np.tan(epoch/r))

# Close logger
logger.close()
print("\nTest complete! Open TensorBoard to view results:")
print("tensorboard --logdir=output/demo_logs")