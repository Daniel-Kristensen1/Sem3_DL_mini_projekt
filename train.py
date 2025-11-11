
import os
from torch.utils.data import DataLoader
import config


# ... Kode til GPU / CPU osv

model = 
loss_function = 

Optimizer = # Hvad gør den?

train_dataset = 
test_dataset = 

train_loader = DataLoader(
    train_dataset,
    batch_size=config.BATCH_SIZE,
    num_workers=8,
    persistent_workers=True,
    drop_last=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=config.BATCH_SIZE,
    num_workers=8,
    persistent_workers=True,
    drop_last=True
)

