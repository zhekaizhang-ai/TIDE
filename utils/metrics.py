import torch


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """
    Compute change-detection metrics for the change class (label = 1).

    Args:
        pred   : binary tensor, any shape, values in {0, 1}
        target : binary tensor, same shape

    Returns:
        dict with keys: precision, recall, f1, iou, oa
    """
    pred   = pred.bool().view(-1)
    target = target.bool().view(-1)

    TP = ( pred &  target).sum().item()
    FP = ( pred & ~target).sum().item()
    FN = (~pred &  target).sum().item()
    TN = (~pred & ~target).sum().item()

    precision = TP / (TP + FP + 1e-8)
    recall    = TP / (TP + FN + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    iou       = TP / (TP + FP + FN + 1e-8)
    oa        = (TP + TN) / (TP + FP + FN + TN + 1e-8)

    return {
        'precision': precision,
        'recall':    recall,
        'f1':        f1,
        'iou':       iou,
        'oa':        oa,
    }
