import h5py
import torch
import numpy as np
from tqdm import tqdm
from dataloader.BSD500 import BSD500B
from torch.utils.data import DataLoader

# idx = 0
# for i in range(238400):
#     print(i)
#     data = torch.tensor(np.array(bsd500[str(i)]))
#     data = data.unfold(1, 8, 1).unfold(2, 8, 1)
#     data = data.permute(0, 1, 2, 3, 4).reshape(-1, data.shape[3], data.shape[4])
#     bsd500b.create_dataset(str(i), shape=(1089, 8, 8), dtype='float32', data=data)

dataset = BSD500B('/home/ducotter/Lipschitz_DSNN/data/BSD500/train_8.h5')
new_dataset = h5py.File('/home/ducotter/Lipschitz_DSNN/data/BSD500/train_128.h5', 'w')
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=32, drop_last=True)

tbar = tqdm(dataloader, ncols=135, position=0, leave=True)
for idx, data in enumerate(tbar):
    new_dataset.create_dataset(str(idx), shape=(128, 1, 8, 8), dtype='float32', data=data)
new_dataset.close()
