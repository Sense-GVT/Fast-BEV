import random
import pickle

dataset = pickle.load(open('./data/nuscenes/nuscenes_infos_val.pkl', 'rb'))
print(dataset.keys())

sample = 500
seed = 42
random.seed(seed)
random.shuffle(dataset['infos'])
dataset['infos'] = dataset['infos'][:sample]

with open(f'./data/nuscenes/nuscenes_infos_val_shuf{sample}.pkl', 'wb') as fid:
    pickle.dump(dataset, fid)
