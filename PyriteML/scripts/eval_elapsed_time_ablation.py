"""Does the policy actually use elapsed_time_0? Real time vs wrong time on BC val data.

Evaluates the validation split (same episodes / seed as training) under paired
conditions that share every random draw:

  real                    unmodified batch
  time_shuffled_other     elapsed time of a sample from a *different* episode
  time_shuffled_same      elapsed time of a sample from the *same* episode, at least
                          --min-dt-s away (right trajectory, wrong phase)
  time_zero               elapsed time 0 ("just started")
  time_const:<T>          elapsed time fixed to T seconds (--const-times)
  shuffle:<key>           reference swaps (--also-shuffle, default rgb_0 and
                          robot0_eef_pos) to put the time effect on a scale

Metrics (see eval_scan_shuffle_ablation.py): paired denoising loss with shared noise and
timesteps, and DDIM-sampled actions from shared initial noise (normalised action MSE,
target-position error in mm, shift vs the real-time prediction). ``real_reseeded`` is the
sampling-noise floor. It also prints the elapsed-time range seen in validation so it can
be compared with what the robot feeds at inference.

Usage (from UMI-FT/):
  PYTHONPATH=PyriteML:.:PyriteUtility:/home/boeing/umi_ft_ws \
    python PyriteML/scripts/eval_elapsed_time_ablation.py \
    --ckpt /home/boeing/umi_ft_ws/data/training_outputs/stringer_v3.5/time/epoch=0150-train_loss=0.015.ckpt \
    --dataset /home/boeing/umi_ft_ws/data/datasets/stringer_v3.5/acp_replay_buffer.zarr
"""

import argparse
import json
import pathlib
import sys
import time
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from eval_scan_shuffle_ablation import (  # noqa: E402
    build_val_dataset,
    load_policy_and_cfg,
    paired_stats,
    per_sample_denoise_loss,
    pick_donors,
    sample_actions,
    swap_key,
)

TIME_KEY = "elapsed_time_0"


def pick_same_episode_donors(
    episodes: np.ndarray, times: np.ndarray, min_dt_s: float, rng: np.random.Generator
):
    """Donor from the same episode with |dt| >= min_dt_s. Returns (donors, found)."""
    n = len(episodes)
    donors = np.arange(n)
    found = np.zeros(n, dtype=bool)
    for i in range(n):
        cand = np.flatnonzero((episodes == episodes[i]) & (np.abs(times - times[i]) >= min_dt_s))
        if len(cand):
            donors[i] = rng.choice(cand)
            found[i] = True
    return donors, found


def set_time(obs: Dict[str, torch.Tensor], value: torch.Tensor):
    out = dict(obs)
    out[TIME_KEY] = value.reshape(obs[TIME_KEY].shape).to(obs[TIME_KEY].dtype)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--max-samples", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--loss-draws", type=int, default=4)
    ap.add_argument("--sample-batches", type=int, default=8)
    ap.add_argument("--const-times", nargs="*", type=float, default=[60.0, 150.0])
    ap.add_argument("--min-dt-s", type=float, default=10.0)
    ap.add_argument("--also-shuffle", nargs="*", default=["rgb_0", "robot0_eef_pos"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    t0 = time.time()
    policy, cfg = load_policy_and_cfg(args.ckpt, device)
    sparse_keys = list(cfg.task.shape_meta["sample"]["obs"]["sparse"].keys())
    assert TIME_KEY in sparse_keys, f"{TIME_KEY} is not a policy input of this checkpoint"
    also = [k for k in args.also_shuffle if k in sparse_keys]
    print(f"[eval] policy loaded in {time.time() - t0:.0f}s; start_pad_s="
          f"{cfg.task.shape_meta.obs[TIME_KEY].get('start_pad_s')}")
    norm = policy.sparse_normalizer[TIME_KEY]
    print(f"[eval] time normalizer: scale={norm.params_dict['scale'].flatten().tolist()} "
          f"offset={norm.params_dict['offset'].flatten().tolist()}")

    t0 = time.time()
    val = build_val_dataset(cfg, args.dataset)
    episodes_all = np.array([int(e) for e, _ in val.sampler.indices])
    print(f"[eval] val dataset: {len(val)} samples from {len(np.unique(episodes_all))} episodes "
          f"(built in {time.time() - t0:.0f}s)")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(val))[: min(args.max_samples, len(val))]
    loader = DataLoader(Subset(val, order.tolist()), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)

    loss_conds = ["real", "time_shuffled_other", "time_shuffled_same", "time_zero"]
    loss_conds += [f"time_const:{t:g}" for t in args.const_times]
    loss_conds += [f"shuffle:{k}" for k in also]
    sample_conds = loss_conds + ["real_reseeded"]
    losses: Dict[str, List[np.ndarray]] = {c: [] for c in loss_conds}
    nmse = {c: [] for c in sample_conds}
    pos_err_mm = {c: [] for c in sample_conds}
    pos_shift_mm = {c: [] for c in sample_conds if c != "real"}
    differs_other, same_found, all_times, all_eps = [], [], [], []
    normalizer = policy.sparse_normalizer

    start = 0
    for b, batch in enumerate(loader):
        tb = time.time()
        obs = {k: v.to(device) for k, v in batch["obs"]["sparse"].items()}
        act = batch["action"]["sparse"].to(device)
        B = act.shape[0]
        eps = episodes_all[order[start : start + B]]
        start += B
        times = obs[TIME_KEY].reshape(B).detach().cpu().numpy().astype(np.float64)
        all_times.append(times)
        all_eps.append(eps)

        d_other = torch.from_numpy(pick_donors(eps, rng)).to(device)
        d_same_np, found = pick_same_episode_donors(eps, times, args.min_dt_s, rng)
        d_same = torch.from_numpy(d_same_np).to(device)
        differs_other.append((obs[TIME_KEY][d_other] != obs[TIME_KEY]).reshape(B).cpu().numpy())
        same_found.append(found)

        cond_obs = {
            "real": obs,
            "time_shuffled_other": swap_key(obs, TIME_KEY, d_other),
            "time_shuffled_same": swap_key(obs, TIME_KEY, d_same),
            "time_zero": set_time(obs, torch.zeros(B, device=device)),
        }
        for t in args.const_times:
            cond_obs[f"time_const:{t:g}"] = set_time(obs, torch.full((B,), t, device=device))
        for k in also:
            cond_obs[f"shuffle:{k}"] = swap_key(obs, k, d_other)

        nact = normalizer["action"].normalize(act)
        acc = {c: torch.zeros(B, device=device) for c in loss_conds}
        for d in range(args.loss_draws):
            g = torch.Generator(device=device).manual_seed(args.seed * 100003 + b * 101 + d)
            noise = torch.randn(nact.shape, device=device, generator=g)
            noise2 = torch.randn(nact.shape, device=device, generator=g)
            t = torch.randint(0, policy.noise_scheduler.config.num_train_timesteps, (B,),
                              device=device, generator=g).long()
            for c in loss_conds:
                acc[c] += per_sample_denoise_loss(policy, cond_obs[c], nact, noise, noise2, t)
        for c in loss_conds:
            losses[c].append((acc[c] / args.loss_draws).cpu().numpy())

        if b < args.sample_batches:
            seed = 1_000_000 + args.seed * 1000 + b
            preds = {c: sample_actions(policy, cond_obs[c], seed) for c in loss_conds}
            preds["real_reseeded"] = sample_actions(policy, obs, seed + 7919)
            gt_n = normalizer["action"].normalize(act)
            for c, p in preds.items():
                p_n = normalizer["action"].normalize(p)
                nmse[c].append(((p_n - gt_n) ** 2).mean(dim=(1, 2)).cpu().numpy())
                pos_err_mm[c].append(
                    (1000.0 * (p[..., :3] - act[..., :3]).norm(dim=-1).mean(dim=1)).cpu().numpy())
                if c != "real":
                    pos_shift_mm[c].append(
                        (1000.0 * (p[..., :3] - preds["real"][..., :3]).norm(dim=-1).mean(dim=1))
                        .cpu().numpy())
        print(f"[eval] batch {b + 1}/{len(loader)} ({time.time() - tb:.1f}s)")

    L = {c: np.concatenate(v) for c, v in losses.items()}
    t_all = np.concatenate(all_times)
    ep_all = np.concatenate(all_eps)
    d_other = np.concatenate(differs_other)
    s_found = np.concatenate(same_found)
    ep_len = {int(e): float(t_all[ep_all == e].max()) for e in np.unique(ep_all)}
    report = {
        "ckpt": args.ckpt, "dataset": args.dataset,
        "n_samples": int(len(L["real"])),
        "n_val_episodes": int(len(np.unique(ep_all))),
        "elapsed_time_s": {
            "min": float(t_all.min()), "p50": float(np.percentile(t_all, 50)),
            "p95": float(np.percentile(t_all, 95)), "max": float(t_all.max()),
            "longest_val_episode_s": max(ep_len.values()),
        },
        "denoise_loss": {"real_mean": float(L["real"].mean())},
        "denoise_loss_donor_differs_only": {},
        "sampled": {},
    }
    masks = {"time_shuffled_other": d_other, "time_shuffled_same": s_found}
    for c in loss_conds[1:]:
        report["denoise_loss"][c] = paired_stats(L["real"], L[c])
        m = masks.get(c)
        if m is not None and m.any():
            report["denoise_loss_donor_differs_only"][c] = paired_stats(L["real"][m], L[c][m])
    if nmse["real"]:
        for c in sample_conds:
            e = {"naction_mse": float(np.concatenate(nmse[c]).mean()),
                 "pos_err_mm": float(np.concatenate(pos_err_mm[c]).mean())}
            if c != "real":
                e["pos_shift_vs_real_mm"] = float(np.concatenate(pos_shift_mm[c]).mean())
            report["sampled"][c] = e
        report["sampled"]["n_samples"] = int(len(np.concatenate(nmse["real"])))

    print("\n=== elapsed time ablation ===")
    print(f"ckpt: {args.ckpt}")
    r = report["elapsed_time_s"]
    print(f"{report['n_samples']} val samples, {report['n_val_episodes']} val episodes; "
          f"elapsed_time_0 min {r['min']:.1f} / median {r['p50']:.1f} / p95 {r['p95']:.1f} / "
          f"max {r['max']:.1f} s")
    print(f"same-episode donor (>= {args.min_dt_s:g}s apart) found for {100 * s_found.mean():.1f}% of samples")
    print("\nDenoising loss (shared noise/timesteps):")
    print(f"  {'real':<24s} {L['real'].mean():.5f}")
    for c in loss_conds[1:]:
        s = report["denoise_loss"][c]
        print(f"  {c:<24s} {s['mean']:.5f}   delta {s['delta']:+.5f} ± {s['delta_se']:.5f} (SE)"
              f"   {s['delta_rel_pct']:+.2f}%")
    if report["sampled"]:
        print(f"\nSampled actions ({report['sampled']['n_samples']} samples, DDIM, shared initial noise):")
        print(f"  {'condition':<24s} {'naction_mse':>12s} {'pos_err_mm':>11s} {'shift_vs_real_mm':>17s}")
        for c in sample_conds:
            e = report["sampled"][c]
            shift = f"{e['pos_shift_vs_real_mm']:.2f}" if "pos_shift_vs_real_mm" in e else "-"
            print(f"  {c:<24s} {e['naction_mse']:12.5f} {e['pos_err_mm']:11.2f} {shift:>17s}")
        print("  (real_reseeded = real obs, different initial noise: the sampling-noise floor)")

    out = pathlib.Path(args.out) if args.out else pathlib.Path(args.ckpt).with_name(
        pathlib.Path(args.ckpt).stem + "_elapsed_time_eval.json")
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
