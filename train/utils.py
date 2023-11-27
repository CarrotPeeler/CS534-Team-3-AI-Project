import numpy as np
import torch
from ultralytics.utils.metrics import box_iou


def match_predictions(pred_classes, true_classes, iou, use_scipy=False):
    """
    Matches predictions to ground truth objects (pred_classes, true_classes) using IoU.

    Args:
        pred_classes (torch.Tensor): Predicted class indices of shape(N,).
        true_classes (torch.Tensor): Target class indices of shape(M,).
        iou (torch.Tensor): An NxM tensor containing the pairwise IoU values for predictions and ground of truth
        use_scipy (bool): Whether to use scipy for matching (more precise).

    Returns:
        (torch.Tensor): Correct tensor of shape(N,10) for 10 IoU thresholds.
    """
    iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
    # Dx10 matrix, where D - detections, 10 - IoU thresholds
    correct = np.zeros((pred_classes.shape[0], iouv.shape[0])).astype(bool)
    # LxD matrix where L - labels (rows), D - detections (columns)
    correct_class = true_classes[:, None] == pred_classes
    iou = iou * correct_class  # zero out the wrong classes
    iou = iou.cpu().numpy()
    for i, threshold in enumerate(iouv.cpu().tolist()):
        if use_scipy:
            # WARNING: known issue that reduces mAP in https://github.com/ultralytics/ultralytics/pull/4708
            import scipy  # scope import to avoid importing for all commands
            cost_matrix = iou * (iou >= threshold)
            if cost_matrix.any():
                labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(cost_matrix, maximize=True)
                valid = cost_matrix[labels_idx, detections_idx] > 0
                if valid.any():
                    correct[detections_idx[valid], i] = True
        else:
            matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
            matches = np.array(matches).T
            if matches.shape[0]:
                if matches.shape[0] > 1:
                    matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    # matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)


def process_batch(detections, labels):
    """
    Return correct prediction matrix.

    Args:
        detections (torch.Tensor): Tensor of shape [N, 6] representing detections.
            Each detection is of the format: x1, y1, x2, y2, conf, class.
        labels (torch.Tensor): Tensor of shape [M, 5] representing labels.
            Each label is of the format: class, x1, y1, x2, y2.

    Returns:
        (torch.Tensor): Correct prediction matrix of shape [N, 10] for 10 IoU levels.
    """
    iou = box_iou(labels[:, 1:], detections[:, :4])
    return match_predictions(detections[:, 5], labels[:, 0], iou)


def get_mean_results():
    pass