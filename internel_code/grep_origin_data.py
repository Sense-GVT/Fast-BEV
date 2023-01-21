import os

grep_path = 'data/2021_12_17_9_30.txt'
gt_data_path = '/mnt/share_data/czh/indata'
with open(grep_path, 'r') as f:
    lines = f.readlines()
bash_file = open('grep_origin_data.sh', 'w')
for index, line in enumerate(lines):
    folder_name = line.strip()
    print("Processing %s..." % folder_name)
    valid_folder_path = os.path.join(gt_data_path, folder_name)
    with open(os.path.join(valid_folder_path, 'aws_data_path.txt'), 'r') as f:
        data = f.readlines()
    source_path = data[0].strip()
    print(source_path)
    output_folder_name = source_path.split('/')[-1]

    aws_cmd1 = 'aws --endpoint-url=http://sz21.ceph.sensebee.xyz --profile ad_system_common '\
        's3 cp s3://%s/ /mnt/share_data/czh/indata/%s --recursive --exclude "*" --include "*.h264"' % (source_path, output_folder_name)
    aws_cmd2 = 'aws --endpoint-url=http://sz21.ceph.sensebee.xyz --profile ad_system_common '\
        's3 cp s3://%s/ /mnt/share_data/czh/indata/%s --recursive --exclude "*" --include "*.txt" --exclude "logs/*" ' % (source_path, output_folder_name)
    aws_cmd3 = 'aws --endpoint-url=http://sz21.ceph.sensebee.xyz --profile ad_system_common '\
        's3 cp s3://%s/config /mnt/share_data/czh/indata/%s/config --recursive' % (source_path, output_folder_name)
    bash_file.write(aws_cmd1 + '\n')
    bash_file.write(aws_cmd2 + '\n')
    bash_file.write(aws_cmd3 + '\n')
bash_file.close()


    