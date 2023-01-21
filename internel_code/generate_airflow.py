
def gen_json(data_path, sequence_freq=10):
    template = \
    '''
    {
        "t1":{"data_path": ["%s"],
                "mode": "aws"
                },
        "t2":{"data_path": ["%s"],
                "video_sample_rate":%d,
                "run_camera_infer":true
                },
        "t3":{"data_path": ["%s"]},
        "t4":{"data_path": ["%s"]},
        "t5":{"data_path": ["%s"]}
    }
    ''' % (data_path, data_path, sequence_freq, data_path, data_path, data_path)
    print(template)

ceph_path_list = [
    'sz21_adas/2021_12/2021_12_17/CN015/9633/raw_data/2021_12_17_11_30_55_AutoCollect',
    'sz21_adas/2021_12/2021_12_17/CN015/9633/raw_data/2021_12_17_11_18_42_AutoCollect',
    'sz21_adas/2021_12/2021_12_16/CN015/9627/raw_data/2021_12_16_16_10_06_AutoCollect'
]
for ceph_path in ceph_path_list:
    gen_json(ceph_path)