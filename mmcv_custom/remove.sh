set -e
set -x

cp  multi_scale_deform_attn.py /opt/conda/lib/python3.6/site-packages/mmcv/ops/multi_scale_deform_attn.py
cp cpp_extension.py /opt/conda/lib/python3.6/site-packages/torch/utils/
cp checkpoint.py /opt/conda/lib/python3.6/site-packages/fvcore/common/
