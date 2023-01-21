import sys
import mmcv
import time
import ipdb

ann_file = sys.argv[1]

data = mmcv.load(ann_file)
data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))


cam_types = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]
cam_kws = [
    "sensor2ego_translation",  # static
    "sensor2ego_rotation",  # static
    "ego2global_translation",  # dynamic
    "ego2global_rotation",  # dynamic
    "sensor2lidar_rotation",  # dynamic
    "sensor2lidar_translation",  # dynamic
    "cam_intrinsic",  # static
]

info_kws = [
    "lidar2ego_translation",  # static
    "lidar2ego_rotation",  # static
    "ego2global_translation",  # dynamic
    "ego2global_rotation",  # dynamic
]

extrinsic_kws = ["cam_intrinsic", "sensor2lidar_rotation", "sensor2lidar_translation"]

cam_id_list = [
    0,
]
kw_id_list = [
    0,
]

# ipdb.set_trace()
for info_id, info in enumerate(data_infos):
    # for kw_id, kw in enumerate(info_kws):
    #     if kw_id not in kw_id_list:
    #         continue
    #     print(kw, info[kw])
    #     time.sleep(0.5)
    for cam_id, cam in enumerate(cam_types):
        if cam_id not in cam_id_list:
            continue
        # for kw_id, kw in enumerate(extrinsic_kws):
        #     if kw_id not in kw_id_list:
        #         continue
        #     print(cam, kw, info["cams"][cam][kw])
        #     time.sleep(0.5)
        for kw_id, kw in enumerate(cam_kws):
            if kw_id not in kw_id_list:
                continue
            print(cam, kw, info["cams"][cam][kw])
            time.sleep(0.5)
