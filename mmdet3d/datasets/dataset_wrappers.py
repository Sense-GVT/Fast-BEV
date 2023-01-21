import os
import cv2
import skimage.io
import numpy as np
import matplotlib.pyplot as plt

from .builder import DATASETS


@DATASETS.register_module()
class CBGSDataset(object):
    """A wrapper of class sampled dataset with ann_file path. Implementation of
    paper `Class-balanced Grouping and Sampling for Point Cloud 3D Object
    Detection <https://arxiv.org/abs/1908.09492.>`_.

    Balance the number of scenes under different classes.

    Args:
        dataset (:obj:`CustomDataset`): The dataset to be class sampled.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.CLASSES = dataset.CLASSES
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}
        self.sample_indices = self._get_sample_indices()
        # self.dataset.data_infos = self.data_infos
        if hasattr(self.dataset, 'flag'):
            self.flag = np.array(
                [self.dataset.flag[ind] for ind in self.sample_indices],
                dtype=np.uint8)

    def _get_sample_indices(self):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations after class sampling.
        """
        class_sample_idxs = {cat_id: [] for cat_id in self.cat2id.values()}
        for idx in range(len(self.dataset)):
            sample_cat_ids = self.dataset.get_cat_ids(idx)
            for cat_id in sample_cat_ids:
                class_sample_idxs[cat_id].append(idx)
        duplicated_samples = sum(
            [len(v) for _, v in class_sample_idxs.items()])
        class_distribution = {
            k: len(v) / duplicated_samples
            for k, v in class_sample_idxs.items()
        }

        sample_indices = []

        frac = 1.0 / len(self.CLASSES)
        ratios = [frac / v for v in class_distribution.values()]
        for cls_inds, ratio in zip(list(class_sample_idxs.values()), ratios):
            sample_indices += np.random.choice(cls_inds,
                                               int(len(cls_inds) *
                                                   ratio)).tolist()
        return sample_indices

    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """
        ori_idx = self.sample_indices[idx]
        return self.dataset[ori_idx]

    def __len__(self):
        """Return the length of data infos.

        Returns:
            int: Length of data infos.
        """
        return len(self.sample_indices)


class MultiViewMixin:
    colors = np.multiply([
        plt.cm.get_cmap('gist_ncar', 37)((i * 7 + 5) % 37)[:3] for i in range(37)
    ], 255).astype(np.uint8).tolist()

    @staticmethod
    def draw_corners(img, corners, color, projection):
        corners_3d_4 = np.concatenate((corners, np.ones((8, 1))), axis=1)
        corners_2d_3 = corners_3d_4 @ projection.T
        z_mask = corners_2d_3[:, 2] > 0
        corners_2d = corners_2d_3[:, :2] / corners_2d_3[:, 2:]
        corners_2d = corners_2d.astype(np.int)
        for i, j in [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]:
            if z_mask[i] and z_mask[j]:
                img = cv2.line(
                    img=img,
                    pt1=tuple(corners_2d[i]),
                    pt2=tuple(corners_2d[j]),
                    color=color,
                    thickness=2,
                    lineType=cv2.LINE_AA
                )

    def show(self, results, out_dir):
        assert out_dir is not None, 'Expect out_dir, got none.'
        for i, result in enumerate(results):
            info = self.get_data_info(i)
            gt_bboxes = self.get_ann_info(i)
            for j in range(len(info['img_info'])):
                img1 = skimage.io.imread(info['img_info'][j]['filename'])
                img2 = skimage.io.imread(info['img_info'][j]['filename'])
                extrinsic = info['lidar2img']['extrinsic'][j]
                intrinsic = info['lidar2img']['intrinsic'][:3, :3]
                projection = intrinsic @ extrinsic[:3]
                if not len(result['scores_3d']):
                    continue
                corners = result['boxes_3d'].corners.numpy()
                scores = result['scores_3d'].numpy()
                labels = result['labels_3d'].numpy()
                for corner, score, label in zip(corners, scores, labels):
                    self.draw_corners(img1, corner, self.colors[label], projection)
                    
                corners = gt_bboxes['gt_bboxes_3d'].corners.numpy()
                labels = gt_bboxes['gt_labels_3d']
                for corner, label in zip(corners, labels):
                    self.draw_corners(img2, corner, colors[label], projection)
                    
                out_file_name = os.path.split(info['img_info'][j]['filename'])[-1][:-4]
                skimage.io.imsave(os.path.join(out_dir, '{}_pred.png'.format(out_file_name)), img1)
                skimage.io.imsave(os.path.join(out_dir, '{}_gt.png'.format(out_file_name)), img2)

                
@DATASETS.register_module()
class SmallDataset:
    """A wrapper of repeated dataset.
    The length of repeated dataset will be `times` larger than the original
    dataset. This is useful when the data loading time is long but the dataset
    is small. Using RepeatDataset can reduce the data loading time between
    epochs.
    Args:
        dataset (:obj:`Dataset`): The dataset to be repeated.
        times (int): Repeat times.
    """

    def __init__(self, dataset, num_images=10):
        self.dataset = dataset
        self.num_images = num_images
        self.CLASSES = dataset.CLASSES
        if hasattr(self.dataset, 'flag'):
            self.flag = self.dataset.flag[:self.num_images]
        
        self.step = (len(dataset) // num_images)
        self._ori_len = num_images
        
        # [0,5626,11252,16878,22504]
        if num_images == 1:
            #self.pool = [5626]
            #self.pool = [0]
            self.pool = [11252]
        elif num_images == 2:
            self.pool = [0, 5626]
        elif num_images == 5:
            self.pool = [0, 5626, 11252, 16878, 22504]
        else:
            self.pool = None

    def __getitem__(self, idx):
        if self.pool is not None:
            new_idx = self.pool[idx % self._ori_len]
        else:
            new_idx = (idx % self._ori_len) * self.step
            
        if self.num_images < 5:
            print("*** {} ***".format(new_idx))
        return self.dataset[new_idx]

    def __len__(self):
        """Length after repetition."""
        return self._ori_len
