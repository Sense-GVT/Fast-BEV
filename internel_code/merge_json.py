import mmcv

def merge_json_utils(_type='train'):
    src_path = 'data/internal/%s.json' % _type
    incres_path = '/mnt/share_data/czh/internal1/%s_incremental.json' % _type
    src_data = mmcv.load(src_path)
    incres_data = mmcv.load(incres_path)

    # check if incres data in original data
    prev_path = '/nfs/chenzehui/code/detr3d-internal/internal_toolkit/%s_prev.txt' % _type
    new_path = '/nfs/chenzehui/code/detr3d-internal/internal_toolkit/%s.txt' % _type

    with open(prev_path, 'r') as f:
        prev_lines = f.readlines()
    prev_set = set()
    for prev_line in prev_lines:
        prev_set.add(prev_line.strip())
    with open(new_path, 'r') as f:
        new_lines = f.readlines()
    for new_line in new_lines:
        if new_line.strip() in prev_set:
            print(new_line)
            raise Exception
    new_data = dict(
        infos=src_data['infos'] + incres_data['infos']
    )
    mmcv.dump(new_data, src_path)

if __name__ == '__main__':
    merge_json_utils('train')
    # merge_json_utils('test')