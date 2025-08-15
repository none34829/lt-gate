# src/calibration.py
import torch
from collections import defaultdict

@torch.no_grad()
def calibrate_thresholds(
    model,
    loader,
    device,
    target_rate=0.02,     # 2% average spike fraction per layer per step
    batches=10,           # use a few small batches
    iters=3,              # a couple of refinement rounds
    tol=0.50,             # ±50% tolerance band
    min_th=0.05,          # clamp thresholds to sane range
    max_th=2.00,
    verbose=True,
):
    """
    Calibrate layer thresholds so each spiking layer reaches a target
    mean spike fraction per timestep, without touching inputs.
    """

    def get_thr(m):
        # Try common places thresholds might live
        if hasattr(m, "threshold"):
            return float(m.threshold)
        if hasattr(m, "dual_lif") and hasattr(m.dual_lif, "threshold"):
            return float(m.dual_lif.threshold)
        if hasattr(m, "cell") and hasattr(m.cell, "threshold"):
            return float(m.cell.threshold)
        return None

    def set_thr(m, v):
        v = float(max(min_th, min(max_th, v)))
        if hasattr(m, "threshold"):
            m.threshold = v
        elif hasattr(m, "dual_lif") and hasattr(m.dual_lif, "threshold"):
            m.dual_lif.threshold = v
        elif hasattr(m, "cell") and hasattr(m.cell, "threshold"):
            m.cell.threshold = v

    # Spiking layers to calibrate = any module that exposes a threshold in one of the above places.
    calibs = [(n, m) for n, m in model.named_modules() if get_thr(m) is not None]
    if verbose:
        print("Calibrating thresholds for:", [n for n, _ in calibs])

    # Forward hooks to measure spike fractions per layer
    def hook_factory(name, stats):
        def hook(_m, _inp, out):
            # Unify different return shapes
            if isinstance(out, (tuple, list)):
                if len(out) >= 3:     # DualLIFNeuron returns (fast, slow, merged)
                    spikes = out[2]
                else:                 # e.g., (spikes, recon_loss)
                    spikes = out[0]
            else:
                spikes = out
            stats[name]["sum"] += spikes.float().mean().item()
            stats[name]["n"] += 1
        return hook

    handles = []
    try:
        for it in range(iters):
            stats = defaultdict(lambda: {"sum": 0.0, "n": 0})
            # Register hooks
            for name, module in calibs:
                handles.append(module.register_forward_hook(hook_factory(name, stats)))

            model.train()  # we want real spiking behavior
            # Run a few batches to estimate rates
            for b, (seqs, _targets) in enumerate(loader):
                if b >= batches:
                    break
                seqs = seqs.to(device, non_blocking=True)
                if seqs.size(0) != 200:  # ensure [T,B,C,H,W]
                    seqs = seqs.transpose(0, 1)
                model.reset_state()
                T = seqs.shape[0]
                for t in range(T):
                    _ = model(seqs[t])

            # Remove hooks for this iteration
            for h in handles:
                h.remove()
            handles.clear()

            changes = 0
            lower = target_rate * (1 - tol)
            upper = target_rate * (1 + tol)

            # Adjust thresholds per layer
            for name, module in calibs:
                if stats[name]["n"] == 0:
                    continue
                rate = stats[name]["sum"] / stats[name]["n"]
                thr_now = get_thr(module)
                new_thr = thr_now

                if rate < lower:
                    # too quiet → reduce threshold
                    scale = max(0.5, rate / max(1e-6, target_rate))  # conservative
                    new_thr = thr_now * scale
                elif rate > upper:
                    # too active → increase threshold
                    scale = min(2.0, rate / max(1e-6, target_rate))  # conservative
                    new_thr = thr_now * scale

                new_thr = float(max(min_th, min(max_th, new_thr)))
                if abs(new_thr - thr_now) / max(1e-6, thr_now) > 0.05:
                    set_thr(module, new_thr)
                    changes += 1
                    if verbose:
                        print(f"[calib {it+1}] {name}: rate={rate:.4f} thr {thr_now:.4f} → {new_thr:.4f}")

            if changes == 0:
                if verbose:
                    print(f"Calibration converged at iter {it+1}.")
                break

    finally:
        for h in handles:
            h.remove()

    # Return final thresholds for logging/repro
    return {name: float(get_thr(m)) for name, m in calibs}

