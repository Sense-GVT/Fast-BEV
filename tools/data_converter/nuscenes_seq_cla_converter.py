import mmcv
import ipdb
from tqdm import tqdm

frm_cnt = {}
for set in ['val', 'train']:
    filename = f"data/cla/annotations/annotation_0616/sweeps/{set}_detr3d_with_sweeps_dataset.json"
    data = mmcv.load(filename)
    # ipdb.set_trace()
    for info in tqdm(data['infos']):
        for cam_type, cam_info in info['cams'].items():
            cam_info['data_path'] = cam_info['aws_path'].replace('s3://sh1984_datasets/', '')
            cam_info['timestamp'] = int(str(cam_info['timestamp'])[:16])
        info.update(dict(prev=info.pop('sweeps'), next=None))

        frm_len = len(info['prev'])
        if frm_len not in frm_cnt:
            frm_cnt[frm_len] = 0
        frm_cnt[frm_len] += 1

        for prev_id in range(len(info['prev'])):
            for cam_type, cam_info in info['prev'][prev_id].items():
                cam_info['data_path'] = cam_info['aws_path'].replace('s3://sh1984_datasets/', '')
                cam_info['timestamp'] = int(str(cam_info['timestamp'])[:16])
    dumpname = f"data/cla/annotations/annotation_0616/sweeps/{set}_detr3d_with_seq_dataset.json"
    mmcv.dump(data, dumpname)
    print(dumpname, frm_cnt)
