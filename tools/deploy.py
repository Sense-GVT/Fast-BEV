# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import mmcv
import os
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import init_dist, load_checkpoint


# from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor
import re
import sys
import json
import time
from easydict import EasyDict as edict
import ipdb


def os_popen(stmt, *parm):
    re = os.popen(stmt).readlines()
    result = []
    for i in range(0, len(re) - 1):
        res = re[i].strip('\n')
        result.append(res)
    if parm == ():
        return result
    else:
        line = int(parm[0]) - 1
        return result[line]


def submit_adela_task(save_path, release=False, download=False):

    IAG_AD_PROJECT_ID = 20
    # deployment_platform = ["cuda11.1-trt7.2.1-fp16-3080", "cuda11.1-trt7.2.1-fp32-P4", "cuda11.1-trt7.2.1-fp16-T4"]
    # deployment_platform = ["cuda11.1-trt7.2.1-fp16-3080"]
    # deployment_platform = ["cuda11.1-trt7.2.1-fp16-T4"]
    # deployment_platform = ["cuda11.1-trt7.2.1-fp16-3080", "cuda11.1-trt7.2.1-fp16-T4"]
    deployment_platform = []

    # generate release.json
    result = os.system("python -m adela.make_json {} -o {}".format(
        save_path, os.path.join(save_path, "release.json")))

    if result:
        print("generate release.json error")
        sys.exit(0)

    cmd = "python -m adela.cmd -p {} -ra {}".format(
        IAG_AD_PROJECT_ID, os.path.join(save_path, "release.json"))
    result = os_popen(cmd)
    result = [res.strip() for res in result]

    # deploy model
    release_id = 0
    for res in result:
        ret = re.findall(r"\'id\': \d+\.?\d*", res)
        if ret:
            ret = re.findall(r"\d+\.?\d*", res)
            release_id = int(ret[0])
    if not release_id:
        print("get release id error, please check")
        sys.exit(0)

    deploy_id_dict = {}
    for platform in deployment_platform:
        deployment = {"platform": platform, "max_batch_size": 6, "config_json": {
            "__image__": "registry.sensetime.com/nart/nart:1.2.13-dev-cuda11.1-03a66b76-1"}}
        json.dump(deployment, open(os.path.join(
            save_path, 'deployment-{}.json'.format(platform)), 'w'))

        cmd = "python -m adela.cmd -p {} -r {} -da {}".format(
            IAG_AD_PROJECT_ID, release_id,
            os.path.join(save_path, 'deployment-{}.json'.format(platform)))
        result = os_popen(cmd)
        result = [res.strip() for res in result]
        for res in result:
            ret = re.findall(r"\'id\': \d+\.?\d*", res)
            if ret:
                ret = re.findall(r"\d+\.?\d*", res)
                deploy_id_dict[platform] = int(ret[0])
                break

    # release deploy model
    if release:
        for platform, deploy_id in deploy_id_dict.items():
            status = "UNDEPLOYMENT"
            while "SUCCESS" not in status:
                print(
                    "project name: {}, deploy status: {}, wait platform: {} model convert success..."
                    .format("uniconv", status, platform),
                    flush=True)
                cmd = "python -m adela.cmd -p {} -d {}".format(
                    IAG_AD_PROJECT_ID, deploy_id)
                result = os_popen(cmd)
                result = [res.strip() for res in result]
                for res in result:
                    ret = re.match(r"\'status\': \'[a-zA-Z_]+[\w]*\'", res)
                    if ret:
                        status = ret.group()
                time.sleep(5)
            print(
                "project name: {}, platform: {} model convert done!!! good job!!!".
                format("uniconv", platform),
                flush=True)

    # # donwload deploy model
    if download:
        model_id_dict = {}
        for platform, deploy_id in deploy_id_dict.items():
            cmd = "python -m adela.cmd -p {} -d {} -ma".format(
                IAG_AD_PROJECT_ID, deploy_id)
            result = os_popen(cmd)
            result = [res.strip() for res in result]
            for res in result:
                ret = re.findall(r"\'id\': \d+\.?\d*", res)
                if ret:
                    ret = re.findall(r"\d+\.?\d*", res)
                    model_id_dict[platform] = int(ret[0])
                    break

        for platform, model_id in model_id_dict.items():
            cmd = "python -m adela.cmd -p {} -md {}".format(
                IAG_AD_PROJECT_ID, model_id)
            os_popen(cmd)

        current_path = os.path.abspath("./")
        os.system("mv {}/*.model {}".format(current_path, save_path))


class Wrapper(torch.nn.Module):
    def __init__(self, model, export_2d=False, export_3d=False, export_2dto3d=False):
        super().__init__()
        self.model = model
        self.export_2d = export_2d
        self.export_3d = export_3d

    def forward(self, img):
        img_metas = None
        out = self.model(
            return_loss=False, img=img, img_metas=img_metas,
            export_2d=self.export_2d,
            export_3d=self.export_3d,
        )
        return out


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--replace-bn',
        action='store_true',
        help='Whether to replace anybn to bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument('--release', action='store_true', help='Whether to release model')
    parser.add_argument('--download', action='store_true', help='Whether to download model')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--type', type=str, default="2d", help='deploy type')
    parser.add_argument('--name', type=str, default="KM_UniConvNV", help='deploy name')
    parser.add_argument('--size', type=str, default="6,3,544,960", help='deploy size')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    cfg = Config.fromfile(args.config)
    deploy_anno_file_cfg = {'data.test.ann_file': 'tools/utils/deploy/deploy_template.json'}
    cfg.merge_from_dict(deploy_anno_file_cfg)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = edict()
    dataset.CLASSES = cfg.class_names

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    if args.replace_bn:
        cfg.model.backbone.norm_cfg = {'type': 'BN', 'requires_grad': True}
        cfg.model.neck.norm_cfg = {'type': 'BN', 'requires_grad': True}
        cfg.model.neck_3d.norm_cfg = {'type': 'BN', 'requires_grad': True}
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    head_type = cfg.model.bbox_head.type

    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE

    # export to onnx
    save_path = args.checkpoint.split('.pth')[0]
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    if args.type == "2d":
        # 2d part export
        model = Wrapper(model, export_2d=True)
        img = torch.randn([int(e) for e in args.size.split(',')]).float()

        onnx_save_file = os.path.join(save_path, 'uniconv_tmp.onnx')
        simplified_onnx_save_file = os.path.join(save_path, 'uniconv.onnx')
        torch.onnx.export(model, img, onnx_save_file, opset_version=9, do_constant_folding=False,
                          verbose=True, input_names=["input"], output_names=["output"])
        cmd = f'python -m onnxsim {onnx_save_file} {simplified_onnx_save_file} 1'
        os.system(cmd)
        os.system(f'rm {onnx_save_file}')

        # tar
        parameters = mmcv.load('tools/utils/deploy/parameters_2d.json')
        parameters["input_h"] = cfg.data_config.input_size[0]
        parameters["input_w"] = cfg.data_config.input_size[1]
        parameters["image_means"] = cfg.img_norm_cfg.mean
        parameters["image_stds"] = cfg.img_norm_cfg.std
        parameters["is_rgb"] = cfg.img_norm_cfg.to_rgb
        parameters_save_file = os.path.join(save_path, 'parameters.json')
        mmcv.dump(parameters, parameters_save_file)

        meta = mmcv.load('tools/utils/deploy/meta.json')
        meta['model_name'] = args.name
        meta_save_file = os.path.join(save_path, 'meta.json')
        mmcv.dump(meta, meta_save_file)

        cmd = 'cd {} && tar cvf uniconv.tar uniconv.onnx parameters.json meta.json'.format(save_path)
        os.system(cmd)

        # submit adela task
        submit_adela_task(save_path, args.release, args.download)

    elif args.type == "3d":
        # 3d part export
        model = Wrapper(model, export_3d=True)
        input = torch.randn([int(e) for e in args.size.split(',')]).float()

        onnx_save_file = os.path.join(save_path, 'uniconv_tmp.onnx')
        simplified_onnx_save_file = os.path.join(save_path, 'uniconv.onnx')

        if bool(os.getenv("DEPLOY_DEBUG", False)):
            torch.onnx.export(model, input, onnx_save_file, opset_version=9, do_constant_folding=False,
                              verbose=True, input_names=["input"], output_names=["dummy"])
        else:
            if head_type == 'CenterHead':
                output_names = ["cls_score", "bbox_pred"]
            else:
                output_names = ["cls_score", "bbox_pred", "dir_cls_preds"]
            torch.onnx.export(model, input, onnx_save_file, opset_version=9, do_constant_folding=False,
                              verbose=True, input_names=["input"], output_names=output_names)

        cmd = f'python -m onnxsim {onnx_save_file} {simplified_onnx_save_file} 1'
        os.system(cmd)
        os.system(f'rm {onnx_save_file}')

        # tar
        parameters = mmcv.load('tools/utils/deploy/parameters_3d.json')
        if bool(os.getenv("DEPLOY_DEBUG", False)):
            parameters["model_files"]["net"]["output"] = {"dummy": "dummy"}
        if head_type == 'CenterHead':
            parameters["model_files"]["net"]["output"] = {"cls_score": "cls_score", "bbox_pred": "bbox_pred"}
            parameters["max_num"] = cfg.model.test_cfg.max_per_img
        else:
            parameters["max_num"] = cfg.model.test_cfg.max_num
        parameters["num_classes"] = len(cfg.class_names)
        parameters["n_voxels"] = cfg.model.n_voxels
        parameters["voxel_size"] = cfg.model.voxel_size
        parameters["pc_range"] = cfg.point_cloud_range

        parameters_save_file = os.path.join(save_path, 'parameters.json')
        mmcv.dump(parameters, parameters_save_file)

        meta = mmcv.load('tools/utils/deploy/meta.json')
        meta['model_name'] = args.name
        meta_save_file = os.path.join(save_path, 'meta.json')
        mmcv.dump(meta, meta_save_file)

        cmd = 'cd {} && tar cvf uniconv.tar uniconv.onnx parameters.json meta.json'.format(save_path)
        os.system(cmd)

        # submit adela task
        submit_adela_task(save_path, args.release, args.download)
    else:
        raise NotImplementedError

    return


if __name__ == '__main__':
    main()
