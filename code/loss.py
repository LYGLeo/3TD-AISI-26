import torch
from config import config


def ordinal_regression_loss(
    pred_S,                  
    event_time,              
    is_weekend=None,      
    weekend_weight=1.5,
    event_weight=1.5,
    soft=None,
    sigma=2.0,
):
    """
    Discrete-time ordinal regression loss following the paper.

    Model directly outputs S(t), the survival probability at each interval.

    ℓ_{i,t} = -log S(t)                for t < t*   (survival)
             = -ω_e log(1 - S(t*))     for t = t*   (failure)
             = 0                        for t > t*   (masked)

    GSS applies normalized Gaussian weights w_i(t) centered at t*
    over t <= t* to smooth supervision near the event.
    """

    if soft is None:
        soft = config.get("use_gaussian_smoothing", False)

    event_weight   = config.get("event_weight", event_weight)
    weekend_weight = config.get("weekend_weight", weekend_weight)

    eps = 1e-6
    B, T = pred_S.shape
    device = pred_S.device

    event_time = event_time.clamp(max=T - 1)

    # log S(t) and log(1 - S(t))
    log_S         = torch.log(pred_S.clamp(eps, 1.0 - eps))          
    log_1_minus_S = torch.log((1.0 - pred_S).clamp(eps, 1.0 - eps))   

    t_idx     = torch.arange(T, device=device).unsqueeze(0).expand(B, -1) 
    tstar_exp = event_time.unsqueeze(1)                                    

    # Gaussian-Smoothed Supervision 
    if soft:
        w = torch.exp(-0.5 * ((t_idx - tstar_exp) / sigma) ** 2)
        w = w * (t_idx <= tstar_exp).float()
        w = w / (w.sum(dim=1, keepdim=True) + eps)                     
    else:
        w = torch.ones(B, T, device=device)

    # Survival term: sum_{t<t*} w(t) * log S(t)
    mask_before = (t_idx < tstar_exp).float()
    surv_ll = (log_S * w * mask_before).sum(dim=1)                   
    
    # Failure term: w(t*) * log(1 - S(t*))
    w_tstar = w.gather(1, event_time.unsqueeze(1)).squeeze(1)         
    fail_ll = w_tstar * log_1_minus_S.gather(1, event_time.unsqueeze(1)).squeeze(1)               

    # Event weight
    ll = surv_ll + event_weight * fail_ll                              

    # Weekend weight
    if is_weekend is not None:
        omega_w = torch.where(
            is_weekend.bool(),
            pred_S.new_full((B,), weekend_weight),
            pred_S.new_ones((B,)),
        )
        loss = -(ll * omega_w).sum() / omega_w.sum().clamp_min(eps)
    else:
        loss = -ll.mean()

    return loss
