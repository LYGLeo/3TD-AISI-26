import torch
import torch.nn.functional as F
from config import config

def ordinal_regression_loss(
    pred_q,
    event_time,
    mask,
    is_weekend=None,
    weekend_weight=1.5,
    event_weight=1.5,
    soft=None,       # Now defaults to config toggle
    sigma=1.5
):
    """
    Ordinal regression loss for uncensored time-to-event sequences with optional
    soft failure supervision (Gaussian smoothing before event time).

    Args:
        pred_q: (B, T) - predicted q(t|x), complement hazard
        event_time: (B,) - ground truth event time index (int)
        mask: (B,) - binary indicator (1 = uncensored)
        is_weekend: (B,) - optional, binary (1 = weekend sample)
        weekend_weight: float - down-weight for weekend samples
        event_weight: float - weight for failure part
        soft: bool - if True, apply Gaussian smoothing over failure loss
        sigma: float - standard deviation for Gaussian smoothing (if soft=True)

    Returns:
        Scalar loss (mean over valid samples)
    """
    # === NEW: use config toggle if soft is None ===
    if soft is None:
        soft = config.get("use_gaussian_smoothing", True)

    B, T = pred_q.shape
    device = pred_q.device

    # Clamp event_time to avoid out-of-bounds
    event_time = event_time.clamp(max=T - 1)

    # Compute log q(t) and log(1 - q(t)) safely
    log_q = torch.log(torch.clamp(pred_q, min=1e-6, max=1.0 - 1e-6))
    log_1_minus_q = torch.log(torch.clamp(1 - pred_q, min=1e-6, max=1.0 - 1e-6))

    # Time index for all steps
    time_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
    event_time_expanded = event_time.unsqueeze(1)

    # Survival component
    mask_before_event = (time_idx < event_time_expanded).float()
    log_survival_part = (log_q * mask_before_event).sum(dim=1)

    # Failure component
    if soft:
        gaussian_mask = torch.exp(-0.5 * ((time_idx - event_time_expanded) / sigma) ** 2)
        gaussian_mask = gaussian_mask * (time_idx <= event_time_expanded)
        gaussian_mask = gaussian_mask / (gaussian_mask.sum(dim=1, keepdim=True) + 1e-8)
        log_failure_part = (log_1_minus_q * gaussian_mask).sum(dim=1)
    else:
        log_failure_part = log_1_minus_q.gather(1, event_time.unsqueeze(1).to(torch.int64)).squeeze(1)

    # Total negative log-likelihood
    total_loss = - (log_survival_part + event_weight * log_failure_part)

    # Weekend down-weighting
    if is_weekend is not None:
        weekend_multiplier = torch.where(is_weekend.bool(), weekend_weight, 1.0).to(device)
        total_loss = (total_loss * weekend_multiplier).sum() / weekend_multiplier.sum()
    else:
        total_loss = total_loss.mean()

    return total_loss