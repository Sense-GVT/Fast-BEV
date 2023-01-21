import sys
import mmcv
import time
import ipdb

ann_file = sys.argv[1]

data = mmcv.load(ann_file)
data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))


cam_types = [
    "center_camera_fov120",
    "left_front_camera",
    "left_rear_camera",
    "rear_camera",
    "right_rear_camera",
    "right_front_camera",
]
cam_kws = [
    "extrinsic",  # static
    "cam_intrinsic",  # static
]

info_kws = [
    "center2lidar",  # static
]

extrinsic_kws = [
    "cam_intrinsic",
    "extrinsic",
]

cam_id_list = [
    0,
]
kw_id_list = [
    1,
]

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
