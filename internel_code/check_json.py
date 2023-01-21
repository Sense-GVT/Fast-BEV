import mmcv
import numpy as np

data_path = '/mnt/share_data/czh/internal1/train_incremental.json'
data = mmcv.load(data_path)
print(len(data['infos']))