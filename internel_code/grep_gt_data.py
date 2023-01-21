import os

grep_path = 'data/2021_12_17_9_30.txt'

with open(grep_path, 'r') as f:
    lines = f.readlines()
bash_file = open('grep_gt_data.sh', 'w')
grep_video_file = open('grep_video.sh', 'w')
for index, line in enumerate(lines):
    folder_name = line.strip()
    print("Processing %s..." % folder_name)
    # grep json
    cache_source_path = 'sz21_adas/perception_cla/dataset/%s' % folder_name
    json_grep_cmd = 'aws --endpoint-url=http://sz21.ceph.sensebee.xyz --profile ad_system_common '\
        's3 cp s3://%s/autolabel_cache/ /mnt/share_data/czh/indata/%s/autolabel_cache/ --recursive --exclude "*" --include "*.json"'\
         % (cache_source_path, folder_name)
    # grep output
    outputs_grep_cmd = 'aws --endpoint-url=http://sz21.ceph.sensebee.xyz --profile ad_system_common '\
        's3 cp s3://%s/autolabel_cache/outputs /mnt/share_data/czh/indata/%s/autolabel_cache/outputs/ --recursive'\
         % (cache_source_path, folder_name)
    videos_grep_cmd = 'scp -r phx:/mnt/share_data/czh/indata/%s/autolabel_cache/outputs/center_camera_fov120.mp4 ~/Downloads/%s.mp4' % (folder_name, folder_name)
    
    txt_grep_cmd = 'aws --endpoint-url=http://sz21.ceph.sensebee.xyz --profile ad_system_common '\
        's3 cp s3://%s/aws_data_path.txt /mnt/share_data/czh/indata/%s/'\
         % (cache_source_path, folder_name)

    gt_quality_cmd = 'aws --endpoint-url=http://sz21.ceph.sensebee.xyz --profile ad_system_common '\
        's3 cp s3://%s/autolabel_cache/gt_qualities.txt /mnt/share_data/czh/indata/%s/autolabel_cache/'\
         % (cache_source_path, folder_name)
    # mkdir_cmd = 'mkdir /mnt/share_data/czh/indata/%s/autolabel_cache' % folder_name
    # mv1_cmd = 'mv -f /mnt/share_data/czh/indata/%s/outputs /mnt/share_data/czh/indata/%s/autolabel_cache/' % (folder_name, folder_name)
    # mv2_cmd = 'mv -f /mnt/share_data/czh/indata/%s/*.json /mnt/share_data/czh/indata/%s/autolabel_cache/' % (folder_name, folder_name)
    bash_file.write(json_grep_cmd + '\n')
    bash_file.write(outputs_grep_cmd + '\n')
    bash_file.write(txt_grep_cmd + '\n')
    bash_file.write(gt_quality_cmd + '\n')

    grep_video_file.write(videos_grep_cmd + '\n')
    # bash_file.write(mkdir_cmd + '\n')
    # bash_file.write(mv1_cmd + '\n')
    # bash_file.write(mv2_cmd + '\n')

bash_file.close()
grep_video_file.close()
    