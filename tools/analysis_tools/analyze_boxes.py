import sys
import mmcv
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
from tqdm import tqdm

ann_file = sys.argv[1]
prefix = sys.argv[2]

data = mmcv.load(ann_file)
data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))

x_list = []
y_list = []
z_list = []
x_len_list = []
y_len_list = []
z_len_list = []
yaw_list = []
center_list = []
name_list = []

for info in tqdm(data_infos):
    gt_boxes = info['gt_boxes']
    gt_names = info['gt_names']
    for box, name in zip(gt_boxes, gt_names):
        x_min, y_min, z_min, x_max, y_max, z_max, yaw = box

        x_list.extend([x_min, x_max])
        y_list.extend([y_min, y_max])
        z_list.extend([z_min, z_max])

        x_len_list.extend([x_max-x_min])
        y_len_list.extend([y_max-y_min])
        z_len_list.extend([z_max-z_min])

        yaw_list.append(yaw)

        center_list.append([(x_min+x_max)//2, (y_min+y_max)//2, (z_min+z_max)//2])
        name_list.append(name)

writer = open(f"{prefix}analysis.txt", "w")
for name in set(name_list):
    txt = f"{name}: {sum([e == name for e in name_list])}"
    writer.write(f"{txt}\n")
    print(txt)

plt.figure(figsize=(12, 8))
plt.title("point_range")
plt.plot([e[0] for e in center_list], [e[1] for e in center_list])
plt.savefig(f"{prefix}point_range.png")

x_list = sorted(x_list)
y_list = sorted(y_list)
z_list = sorted(z_list)

x_len_list = sorted(x_len_list)
y_len_list = sorted(y_len_list)
z_len_list = sorted(z_len_list)

yaw_list = sorted(yaw_list)

plt.figure(figsize=(12, 8))
plt.title("x_range")
plt.plot(x_list)
plt.savefig(f"{prefix}x_range.png")
print(f"x_range: {x_list[0], x_list[-1]}")
writer.write(f"x_range: {x_list[0], x_list[-1]}\n")

plt.figure(figsize=(12, 8))
plt.title("x_len_range")
plt.plot(x_len_list)
plt.savefig(f"{prefix}x_len_range.png")
print(f"x_len_range: {x_len_list[0], x_len_list[-1]}")
writer.write(f"x_len_range: {x_len_list[0], x_len_list[-1]}\n")

plt.figure(figsize=(12, 8))
plt.title("y_range")
plt.plot(y_list)
plt.savefig(f"{prefix}y_range.png")
print(f"y_range: {y_list[0], y_list[-1]}")
writer.write(f"y_range: {y_list[0], y_list[-1]}\n")

plt.figure(figsize=(12, 8))
plt.title("y_len_range")
plt.plot(y_len_list)
plt.savefig(f"{prefix}y_len_range.png")
print(f"y_len_range: {y_len_list[0], y_len_list[-1]}")
writer.write(f"y_len_range: {y_len_list[0], y_len_list[-1]}\n")

plt.figure(figsize=(12, 8))
plt.title("z_range")
plt.plot(z_list)
plt.savefig(f"{prefix}z_range.png")
print(f"z_range: {z_list[0], z_list[-1]}")
writer.write(f"z_range: {z_list[0], z_list[-1]}\n")

plt.figure(figsize=(12, 8))
plt.title("z_len_range")
plt.plot(z_len_list)
plt.savefig(f"{prefix}z_len_range.png")
print(f"z_len_range: {z_len_list[0], z_len_list[-1]}")
writer.write(f"z_len_range: {z_len_list[0], z_len_list[-1]}\n")

plt.figure(figsize=(12, 8))
plt.title("yaw_range")
plt.plot(yaw_list)
plt.savefig(f"{prefix}yaw_range.png")
print(f"yaw_range: {yaw_list[0], yaw_list[-1]}")
writer.write(f"yaw_range: {yaw_list[0], yaw_list[-1]}\n")

writer.close()
