import torch
from mmdet3d.registry import MODELS
from mmdet3d.models.segmentors.cylinder3d import Cylinder3D
from mmdet3d.structures import PointData
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig

from p3former.utils.freeze import eval_all_but, freeze_all_but


@MODELS.register_module()
class _P3Former(Cylinder3D):
    """P3Former."""

    # OCCUQ variant A (decode_head.occuq_cfg.freeze_base): only the OCCUQ
    # head trains (spec docs/superpowers/specs/2026-10-07-occuq-density-design.md)
    freeze_base = False

    def __init__(self,
                 voxel_encoder: ConfigType,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 loss_regularization: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(voxel_encoder=voxel_encoder,
                        backbone=backbone,
                        decode_head=decode_head,
                        neck=neck,
                        auxiliary_head=auxiliary_head,
                        loss_regularization=loss_regularization,
                        train_cfg=train_cfg,
                        test_cfg=test_cfg,
                        data_preprocessor=data_preprocessor,
                        init_cfg=init_cfg)
        occuq_cfg = getattr(self.decode_head, 'occuq_cfg', None)
        if occuq_cfg is not None and occuq_cfg.get('freeze_base', False):
            self.freeze_base = True
            freeze_all_but(self, self.decode_head.occuq_head)

    def train(self, mode: bool = True):
        """As nn.Module.train.

        With freeze_base, the voxel encoder, the backbone and the decode
        head outside the OCCUQ head stay in eval mode (fixed BatchNorm
        statistics). The data preprocessor keeps ``mode``, because it builds
        the voxel labels only in training mode."""
        super().train(mode)
        if self.freeze_base:
            for part in (self.voxel_encoder, self.backbone,
                         self.decode_head):
                eval_all_but(part, self.decode_head.occuq_head)
        return self

    def loss(self, batch_inputs_dict,batch_data_samples):
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs_dict (dict): Input sample dict which
                includes 'points' and 'imgs' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - imgs (Tensor, optional): Image tensor has shape (B, C, H, W).
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            Dict[str, Tensor]: A dictionary of loss components.
        """

        # extract features using backbone; frozen in OCCUQ variant A
        if self.freeze_base:
            with torch.no_grad():
                x = self.extract_feat(batch_inputs_dict)
        else:
            x = self.extract_feat(batch_inputs_dict)
        batch_inputs_dict['features'] = x.features
        losses = dict()
        loss_decode = self._decode_head_forward_train(batch_inputs_dict, batch_data_samples)
        losses.update(loss_decode)

        return losses

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        x = self.extract_feat(batch_inputs_dict)
        batch_inputs_dict['features'] = x.features
        pts_semantic_preds, pts_instance_preds, pts_ood_scores = \
            self.decode_head.predict(batch_inputs_dict, batch_data_samples)
        return self.postprocess_result(pts_semantic_preds,
                                       pts_instance_preds,
                                       batch_data_samples,
                                       pts_ood_scores)

    def postprocess_result(self, pts_semantic_preds, pts_instance_preds,
                           batch_data_samples, pts_ood_scores=None):
        for i in range(len(pts_semantic_preds)):
            seg_data = {'pts_semantic_mask': pts_semantic_preds[i],
                        'pts_instance_mask': pts_instance_preds[i]}
            if pts_ood_scores is not None:
                for key, value in pts_ood_scores[i].items():
                    seg_data[f'ood_{key}'] = value
            batch_data_samples[i].set_data(
                {'pred_pts_seg': PointData(**seg_data)})
        return batch_data_samples
