import mmcv
import ipdb
from tqdm import tqdm

filename_train = "data/nuscenes/nuscenes_infos_train_4d_interval3_max60.pkl"
filename_val = "data/nuscenes/nuscenes_infos_val_4d_interval3_max60.pkl"

data_train = mmcv.load(filename_train)
data_val = mmcv.load(filename_val)

filename_trainval = "data/nuscenes/nuscenes_infos_trainval_4d_interval3_max60.pkl"
data_train['infos'] += data_val['infos']

mmcv.dump(data_train, filename_trainval)
print(filename_trainval, len(data_train['infos']))
