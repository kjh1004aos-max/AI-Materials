from metrics.metrics_long_range import compute_all_metrics, setup_optimizer
import numpy as np
import torch

def evaluate_model_uncond(real_sig, gen_sig, args):
    if args.dataset in ['stock', 'sine', 'energy', 'time_series']:
        real = np.asarray(real_sig, dtype=float)
        gen = np.asarray(gen_sig, dtype=float)
        if real.shape != gen.shape:
            raise ValueError(f"Shape mismatch between real and generated signals: {real.shape} vs {gen.shape}")
        diff = real - gen
        mse = np.mean(diff ** 2)
        mae = np.mean(np.abs(diff))
        ss_res = np.sum(diff ** 2)
        ss_tot = np.sum((real - np.mean(real)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return {
            'mse': np.round(mse, 4),
            'mae': np.round(mae, 4),
            'r2': np.round(r2, 4)
        }
    else:
        real_sig, gen_sig = (torch.Tensor(real_sig).float(), torch.Tensor(gen_sig).float())
        scores = compute_all_metrics(real_sig, gen_sig, setup_optimizer if args.dataset == 'temperature_rain' else torch.nn.Identity(), args.device)
        return scores
