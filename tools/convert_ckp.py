import sys
import torch
import ipdb

input = sys.argv[1]
target = sys.argv[2]

model = torch.load(input)
model['state_dict'].update({'neck_fuse_0.weight': model['state_dict'].pop('neck_fuse.weight')})
model['state_dict'].update({'neck_fuse_0.bias': model['state_dict'].pop('neck_fuse.bias')})
torch.save(model, target)
