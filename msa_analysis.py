"""Measurement-system analysis (MSA) of the robot multi-view petiole-length system,
using ONLY existing captures (no new measurement).

Blocks
  A  bias + linearity vs caliper (147 petioles, full observation pool)
  B  within-capture repeatability from DISJOINT 4-view subsets, per measurement procedure
     (sequential / random / coverage / uncertainty-optimal causal)
  C  interval gate: acceptance yield vs quality of accepted measurements
  D  cycle time: per-view time model from raw capture runs (file mtimes + joint deltas),
     plant-level cycle time and spec-conforming throughput per policy

Run:
  python msa_analysis.py            (needs the code repo, see README; set UOV_DIR if it is not a sibling folder)
Outputs -> out/
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
UOV = Path(os.environ.get("UOV_DIR", HERE.parent / "uncertainty-optimal-viewpoints")).resolve()
if not (UOV / "vision" / "view_selection.py").exists():
    sys.exit(f"code repo not found at {UOV}; clone github.com/min577/uncertainty-optimal-viewpoints there or set UOV_DIR")
sys.path.insert(0, str(UOV))
from vision import triangulate as tri, view_selection as vs, figstyle  # noqa: E402

OUT = HERE / "out"
OUT.mkdir(parents=True, exist_ok=True)
INPUTS = HERE / "inputs"
SESS = INPUTS / "sessions"
ABL = UOV / "data" / "ablation"
N_MC = 300
TAU = 5.0            # paper's acceptance criterion: 95% half-interval <= 5 mm
TOL = 10.0           # tolerance width used for P/T (= +-5 mm)
BUDGET = 4

try:
    from scipy import stats as sps
except Exception:  # noqa: BLE001
    sps = None

figstyle.apply(base=13)
import matplotlib.pyplot as plt  # noqa: E402

BLACK, RED = "black", "#C00000"
COH_STYLE = {"Laboratory": dict(color=BLACK, marker="o"), "Greenhouse": dict(color=RED, marker="s")}
PROCS = ["sequential", "random", "coverage", "uncertainty"]
PROC_LABEL = {"sequential": "Sequential", "random": "Random", "coverage": "Coverage-NBV",
              "uncertainty": "Uncertainty-opt."}


# --------------------------------------------------------------------------- data
def sessions():
    lab = [(p, "Laboratory") for p in sorted(SESS.glob("plant*")) if p.is_dir()]
    gh = [(SESS / "greenhouse_v1", "Greenhouse")] + [
        (SESS / f"greenhouse_p{i}", "Greenhouse") for i in range(2, 16)]
    return lab + gh


def load_petioles():
    rows = []
    for sdir, coh in sessions():
        intr, views, _ = tri.load_views(str(sdir))
        clicks = json.loads((sdir / "mv_clicks.json").read_text(encoding="utf-8"))
        gt = json.loads((sdir / "gt.json").read_text(encoding="utf-8"))
        gtmm = {k: float(v) for k, v in gt["gt_mm"].items()}
        eti = gt.get("etiolated", {})
        for pid in sorted(clicks, key=int):
            if pid not in gtmm:
                continue
            cr = [tuple(x) if x else None for x in clicks[pid]["crown"]]
            ju = [tuple(x) if x else None for x in clicks[pid]["junction"]]
            e = (pid in eti) if isinstance(eti, list) else bool(eti.get(pid, False))
            rows.append(dict(cohort=coh, plant=sdir.name, pid=pid, cr=cr, ju=ju, gt=gtmm[pid],
                             intr=intr, views=views, etiolated=e, sweep=len(views)))
    return rows


def tstat_ci(x):
    x = np.asarray(x, float); n = len(x); m = x.mean(); s = x.std(ddof=1)
    se = s / math.sqrt(n)
    tcrit = sps.t.ppf(0.975, n - 1) if sps else 1.96
    p = float(2 * sps.t.sf(abs(m / se), n - 1)) if sps else float("nan")
    return dict(n=n, mean=m, sd=s, ci_lo=m - tcrit * se, ci_hi=m + tcrit * se, p=p)


def ols(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float); n = len(x)
    X = np.c_[np.ones(n), x]
    beta, res, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta; sse = float(((y - yhat) ** 2).sum()); dof = n - 2
    s2 = sse / dof; cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov)); t = beta / se
    p = [float(2 * sps.t.sf(abs(ti), dof)) for ti in t] if sps else [float("nan")] * 2
    r2 = 1 - sse / float(((y - y.mean()) ** 2).sum())
    return dict(intercept=beta[0], slope=beta[1], se_int=se[0], se_slope=se[1],
                p_int=p[0], p_slope=p[1], r2=r2)


# --------------------------------------------------------------------------- A
def block_a(pet):
    res = {}
    for r in pet:
        m = vs.length_ci(r["cr"], r["ju"], r["intr"], r["views"], None, n=N_MC)
        r["L_full"] = m["L"]; r["ci_full"] = m["p95_mm"]; r["err"] = m["L"] - r["gt"]
    for coh in ("Laboratory", "Greenhouse"):
        sub = [r for r in pet if r["cohort"] == coh]
        err = np.array([r["err"] for r in sub]); gt = np.array([r["gt"] for r in sub])
        b = tstat_ci(err); lin = ols(gt, err)
        res[coh] = dict(
            n=len(sub), gt_min=gt.min(), gt_max=gt.max(), gt_sd=float(gt.std(ddof=1)),
            bias_mm=b["mean"], bias_ci=[b["ci_lo"], b["ci_hi"]], bias_p=b["p"],
            sd_err=b["sd"], mae=float(np.abs(err).mean()),
            mape=float(np.mean(np.abs(err) / gt) * 100),
            loa=[b["mean"] - 1.96 * b["sd"], b["mean"] + 1.96 * b["sd"]],
            linearity_slope=lin["slope"], linearity_slope_p=lin["p_slope"],
            linearity_intercept=lin["intercept"], linearity_r2=lin["r2"],
            pct_within_5mm=float(np.mean(np.abs(err) <= 5) * 100),
            r_pearson=float(np.corrcoef(gt, gt + err)[0, 1]),
            bias_wilcoxon_p=float(sps.wilcoxon(err).pvalue) if sps else float("nan"))
        # sensitivity: drop the single largest |error| petiole (reported, not used for the verdict)
        k = int(np.argmax(np.abs(err))); keep = np.arange(len(err)) != k
        b2 = tstat_ci(err[keep]); lin2 = ols(gt[keep], err[keep])
        res[coh]["drop_max"] = dict(
            dropped=f"{sub[k]['plant']}#{sub[k]['pid']}", dropped_err=float(err[k]), dropped_ci=sub[k]["ci_full"],
            n=int(keep.sum()), bias_mm=b2["mean"], bias_p=b2["p"],
            bias_wilcoxon_p=float(sps.wilcoxon(err[keep]).pvalue) if sps else float("nan"),
            loa=[b2["mean"] - 1.96 * b2["sd"], b2["mean"] + 1.96 * b2["sd"]],
            linearity_slope=lin2["slope"], linearity_slope_p=lin2["p_slope"])
    # figure A
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.6))
    for coh, st in COH_STYLE.items():
        sub = [r for r in pet if r["cohort"] == coh]
        gt = np.array([r["gt"] for r in sub]); L = np.array([r["L_full"] for r in sub])
        ax[0].plot(gt, L, linestyle="none", marker=st["marker"], markerfacecolor="none",
                   markeredgecolor=st["color"], markersize=6, label=f"{coh} (n={len(sub)})")
        ax[1].plot(gt, L - gt, linestyle="none", marker=st["marker"], markerfacecolor="none",
                   markeredgecolor=st["color"], markersize=6, label=coh)
        lin = ols(gt, L - gt); xs = np.array([gt.min(), gt.max()])
        ax[1].plot(xs, lin["intercept"] + lin["slope"] * xs, color=st["color"], lw=1.4,
                   label=f"{coh}: fit, error = {lin['intercept']:.1f} + {lin['slope']:.3f}·ref")
    lim = [20, 215]
    ax[0].plot(lim, lim, color="grey", lw=1, ls="--"); ax[0].set_xlim(lim); ax[0].set_ylim(lim)
    ax[0].set_xlabel("Caliper reference (mm)"); ax[0].set_ylabel("Robot measurement (mm)")
    ax[0].legend(loc="upper left", frameon=True, edgecolor="black", fancybox=False)
    ax[0].set_title("(a) Measured vs reference")
    ax[1].axhline(0, color="grey", lw=1, ls="--")
    for coh, st in COH_STYLE.items():
        lo, hi = res[coh]["loa"]
        ax[1].axhline(lo, color=st["color"], lw=0.9, ls=":"); ax[1].axhline(hi, color=st["color"], lw=0.9, ls=":")
    ax[1].set_xlabel("Caliper reference (mm)"); ax[1].set_ylabel("Error, measured - reference (mm)")
    ax[1].set_title("(b) Bias and linearity (dotted: 95% LoA)")
    ax[1].legend(loc="upper left", frameon=True, edgecolor="black", fancybox=False, fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "figA_bias_linearity.png"); plt.close(fig)
    return res


# --------------------------------------------------------------------------- B
def disjoint_subsets(r, proc, k=BUDGET, seed=0):
    cr, ju, intr, views = r["cr"], r["ju"], r["intr"], r["views"]
    pool = vs._obs_pool(cr, ju)
    sets = []
    if proc == "sequential":
        order = list(pool)
        sets = [order[i:i + k] for i in range(0, len(order) - k + 1, k)]
    elif proc == "random":
        order = np.random.default_rng(seed).permutation(pool).tolist()
        sets = [order[i:i + k] for i in range(0, len(order) - k + 1, k)]
    else:
        remaining = set(pool)
        while len(remaining) >= k:
            crm = [cr[i] if i in remaining else None for i in range(len(cr))]
            jum = [ju[i] if i in remaining else None for i in range(len(ju))]
            if proc == "coverage":
                S = vs.select_coverage(crm, jum, intr, views, k)
            else:
                S = vs.select_uncertainty_causal(crm, jum, intr, views, k, n=100)
            if len(S) < k:
                break
            sets.append(list(S)); remaining -= set(S)
    return sets


def block_b(pet):
    rows = []
    for r in pet:
        for proc in PROCS:
            for j, S in enumerate(disjoint_subsets(r, proc)):
                comp = vs._computable(r["cr"], r["ju"], set(S))
                m = vs.length_ci(r["cr"], r["ju"], r["intr"], r["views"], set(S), n=N_MC) if comp else None
                L = m["L"] if (m and m["L"] is not None and m["p95_mm"] < vs._BIG) else None
                rows.append(dict(cohort=r["cohort"], plant=r["plant"], pid=r["pid"], proc=proc,
                                 subset=j, views=" ".join(map(str, S)), computable=bool(L is not None),
                                 L=L, ci=(m["p95_mm"] if L is not None else None), gt=r["gt"],
                                 err=(L - r["gt"]) if L is not None else None))
    with open(OUT / "B_subset_measurements.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    def summarize(sub, gt_sd):
        by = defaultdict(list)
        for x in sub:
            if x["L"] is not None:
                by[(x["plant"], x["pid"])].append(x)
        n_sets = len(sub); n_ok = sum(1 for x in sub if x["L"] is not None)
        ss, dof, per_sd = 0.0, 0, []
        for v in by.values():
            if len(v) >= 2:
                Ls = np.array([x["L"] for x in v]); ss += float(((Ls - Ls.mean()) ** 2).sum()); dof += len(Ls) - 1
                per_sd.append(float(Ls.std(ddof=1)))
        ev = math.sqrt(ss / dof) if dof else float("nan")
        errs = np.array([x["err"] for x in sub if x["err"] is not None])
        tv = math.sqrt(ev ** 2 + gt_sd ** 2)
        return dict(n_subsets=n_sets, n_computable=n_ok, computable_pct=100 * n_ok / n_sets,
                    n_petioles_repeat=len(per_sd), dof=dof, EV_sd_mm=ev,
                    repeatability_5p15=5.15 * ev, median_petiole_sd=float(np.median(per_sd)) if per_sd else float("nan"),
                    pct_EV_of_TV=100 * ev / tv, ndc=1.41 * gt_sd / ev if ev else float("nan"),
                    PT_ratio_pct=100 * 6 * ev / TOL,
                    mae_vs_caliper=float(np.abs(errs).mean()), bias=float(errs.mean()),
                    pct_err_gt_5=float(np.mean(np.abs(errs) > 5) * 100),
                    pct_err_gt_20=float(np.mean(np.abs(errs) > 20) * 100))

    res = {}
    for coh in ("Laboratory", "Greenhouse"):
        gt_sd = float(np.std([r["gt"] for r in pet if r["cohort"] == coh], ddof=1))
        for proc in PROCS:
            sub = [x for x in rows if x["cohort"] == coh and x["proc"] == proc]
            res[f"{coh}|{proc}|all"] = summarize(sub, gt_sd)
            gated = [x for x in sub if x["L"] is not None and x["ci"] <= TAU]
            res[f"{coh}|{proc}|gated"] = summarize(gated, gt_sd) if gated else {}
            res[f"{coh}|{proc}|gated"]["accepted_pct"] = 100 * len(gated) / max(1, sum(1 for x in sub if x["L"] is not None))

    # figure B: per-petiole within-SD by procedure, lab / gh
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
    for ax, coh in zip(axes, ("Laboratory", "Greenhouse")):
        data, data_g = [], []
        for proc in PROCS:
            by = defaultdict(list); byg = defaultdict(list)
            for x in rows:
                if x["cohort"] == coh and x["proc"] == proc and x["L"] is not None:
                    by[(x["plant"], x["pid"])].append(x["L"])
                    if x["ci"] <= TAU:
                        byg[(x["plant"], x["pid"])].append(x["L"])
            data.append([float(np.std(v, ddof=1)) for v in by.values() if len(v) >= 2])
            data_g.append([float(np.std(v, ddof=1)) for v in byg.values() if len(v) >= 2])
        pos = np.arange(len(PROCS))
        bp = ax.boxplot(data, positions=pos - 0.18, widths=0.3, showfliers=False, patch_artist=True,
                        medianprops=dict(color=BLACK, lw=1.4))
        for b in bp["boxes"]:
            b.set(facecolor="white", edgecolor=BLACK, lw=1.2)
        bp2 = ax.boxplot(data_g, positions=pos + 0.18, widths=0.3, showfliers=False, patch_artist=True,
                         medianprops=dict(color=RED, lw=1.4))
        for b in bp2["boxes"]:
            b.set(facecolor="white", edgecolor=RED, lw=1.2)
        for c in ("whiskers", "caps"):
            for ln in bp[c]: ln.set(color=BLACK)
            for ln in bp2[c]: ln.set(color=RED)
        rng = np.random.default_rng(1)
        for i, (d, dg) in enumerate(zip(data, data_g)):
            ax.plot(i - 0.18 + rng.uniform(-0.09, 0.09, len(d)), d, linestyle="none", marker="o",
                    markerfacecolor="none", markeredgecolor=BLACK, markersize=3.2, alpha=0.6)
            ax.plot(i + 0.18 + rng.uniform(-0.09, 0.09, len(dg)), dg, linestyle="none", marker="s",
                    markerfacecolor="none", markeredgecolor=RED, markersize=3.2, alpha=0.6)
            ax.text(i + 0.18, 0.0062, f"n={len(dg)}", color=RED, ha="center", va="bottom", fontsize=9)
        ax.set_yscale("log"); ax.set_xticks(pos); ax.set_xticklabels([PROC_LABEL[p] for p in PROCS], fontsize=10)
        ax.set_title(f"{coh}: SD over disjoint {BUDGET}-view sets")
        ax.axhline(TAU / 1.96, color="grey", lw=1, ls="--")
    axes[0].set_ylabel("SD of repeated length (mm, log)")
    axes[0].plot([], [], color=BLACK, label="all computable sets"); axes[0].plot([], [], color=RED, label=f"sets with CI ≤ {TAU:.0f} mm")
    axes[0].plot([], [], color="grey", lw=1, ls="--", label=f"SD = {TAU:.0f}/1.96 = {TAU / 1.96:.2f} mm")
    axes[0].set_ylim(0.005, 150)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 0.06), frameon=True, edgecolor="black", fancybox=False, fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "figB_repeatability.png"); plt.close(fig)
    return res, rows


# --------------------------------------------------------------------------- C
def block_c(rows):
    taus = np.arange(1.0, 30.5, 0.5)
    res = {}
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.6))
    ax1r = ax[1].twinx()
    for coh, st in COH_STYLE.items():
        sub = [x for x in rows if x["cohort"] == coh and x["proc"] == "uncertainty" and x["L"] is not None]
        ci = np.array([x["ci"] for x in sub]); ae = np.abs([x["err"] for x in sub])
        acc, mae, esc = [], [], []
        for t in taus:
            m = ci <= t
            # MAE only where at least 10 sets are accepted; below that the curve is noise
            acc.append(100 * m.mean()); mae.append(float(ae[m].mean()) if m.sum() >= 10 else np.nan)
            esc.append(100 * float(np.mean(ae[m] > 10)) if m.any() else np.nan)   # escapes: accepted but >10 mm off
        res[coh] = dict(n=len(sub), taus=taus.tolist(), accepted_pct=acc, mae_accepted=mae, escape_pct_gt10=esc,
                        spearman_ci_abserr=float(sps.spearmanr(ci, ae).correlation) if sps else float("nan"),
                        at_tau5=dict(accepted_pct=acc[list(taus).index(5.0)], mae=mae[list(taus).index(5.0)],
                                     escape_pct=esc[list(taus).index(5.0)]),
                        mae_all=float(ae.mean()), pct_gt10_all=100 * float(np.mean(ae > 10)))
        ax[0].plot(ci, ae, linestyle="none", marker=st["marker"], markerfacecolor="none",
                   markeredgecolor=st["color"], markersize=4.5, alpha=0.8, label=f"{coh} (n={len(sub)})")
        ax[1].plot(taus, acc, color=st["color"], lw=1.8, label=f"{coh}: accepted (%)")
        ax1r.plot(taus, mae, color=st["color"], lw=1.8, ls="--", label=f"{coh}: MAE of accepted (mm, right; ≥10 sets)")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].axvline(TAU, color="grey", lw=1, ls="--"); ax[0].axhline(10, color="grey", lw=1, ls=":")
    ax[0].set_xlabel("Predicted 95% half-interval (mm)"); ax[0].set_ylabel("|error| vs caliper (mm)")
    ax[0].set_title("(a) Predicted interval vs actual error (4-view sets)")
    ax[0].legend(loc="upper left", frameon=True, edgecolor="black", fancybox=False, fontsize=9)
    ax[1].axvline(TAU, color="grey", lw=1, ls="--")
    ax[1].set_xlabel("Acceptance threshold τ on predicted interval (mm)"); ax[1].set_ylabel("Accepted (%, solid)")
    ax[1].set_ylim(0, 100); ax1r.set_ylim(0, 8); ax1r.set_ylabel("MAE of accepted (mm, dashed)")
    ax[1].set_title("(b) Yield-quality trade-off of the interval gate")
    h1, l1 = ax[1].get_legend_handles_labels(); h2, l2 = ax1r.get_legend_handles_labels()
    ax[1].legend(h1 + h2, l1 + l2, loc="lower right", frameon=True, edgecolor="black", fancybox=False, fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "figC_interval_gate.png"); plt.close(fig)
    return res


# --------------------------------------------------------------------------- D
SETTLE_STD_MS = 600


def run_dirs():
    """Raw capture runs; only manifest.json + capture_mtimes.json (image file times) are shipped."""
    return sorted((INPUTS / "runs" / "lab").iterdir()), sorted((INPUTS / "runs" / "greenhouse").iterdir())


def intervals(rdir):
    man = json.loads((rdir / "manifest.json").read_text(encoding="utf-8"))
    mtime = json.loads((rdir / "capture_mtimes.json").read_text(encoding="utf-8"))
    caps = sorted(man["captures"], key=lambda c: c["waypoint_index"])
    out = []
    for a, b in zip(caps, caps[1:]):
        ta = mtime[a["files"]["image"]]; tb = mtime[b["files"]["image"]]
        dq = np.abs(np.array(b["joints_at_capture"]) - np.array(a["joints_at_capture"]))
        out.append(dict(dt=tb - ta, dq_max=float(dq.max()), dq_l2=float(np.linalg.norm(dq))))
    return out, man


def path_joints(man):
    return {w["index"]: np.array(w["joints"], float) for w in man["waypoints"]}


def block_d(pet, throughput_csv=ABL / "_throughput5_causal.csv"):
    lab, gh = run_dirs()
    res = {}
    models = {}; all_x = []; n_fit = n_excl = 0
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.6))
    for coh, runs, st in (("Laboratory", lab, COH_STYLE["Laboratory"]), ("Greenhouse", gh, COH_STYLE["Greenhouse"])):
        # Three runs were captured with a 2.5 s settle; their intercept sits ~2.5 s higher. Fit the standard (600 ms) runs only.
        iv = []; iv_x = []; sweeps = []; man0 = None; n_x = 0
        for r in runs:
            o, man = intervals(r)
            if man["settle_ms"] != SETTLE_STD_MS:
                iv_x += o; n_x += 1; continue
            iv += o; sweeps.append(sum(x["dt"] for x in o)); man0 = man0 or man
        dt = np.array([x["dt"] for x in iv]); dq = np.array([x["dq_max"] for x in iv])
        lin = ols(dq, dt)
        models[coh] = (lin, path_joints(man0))
        res[coh] = dict(n_runs=len(sweeps), n_runs_excluded_settle=n_x, n_intervals=len(iv), sweep_s_median=float(np.median(sweeps)),
                        sweep_s_min=float(min(sweeps)), sweep_s_max=float(max(sweeps)),
                        per_view_s_mean=float(dt.mean()), per_view_s_median=float(np.median(dt)),
                        move_model=dict(intercept_s=lin["intercept"], slope_s_per_deg=lin["slope"],
                                        r2=lin["r2"], p_slope=lin["p_slope"]),
                        settle_ms=man0["settle_ms"], speed_percent=man0["speed_percent"])
        # 1~2 intervals per run carry an extra ~3 s pause; they stay in the fit (OLS keeps the mean, so run totals
        # are preserved) but are drawn separately so the line is not read as missing them.
        pause = dt - (lin["intercept"] + lin["slope"] * dq) > 1.5
        res[coh]["n_pause_intervals"] = int(pause.sum())
        xs = np.array([0, dq.max()])
        ax[0].plot(xs, lin["intercept"] + lin["slope"] * xs, color=st["color"], lw=1.0, zorder=1)
        ax[0].plot(dq[~pause], dt[~pause], linestyle="none", marker=st["marker"], markerfacecolor="none",
                   markeredgecolor=st["color"], markersize=4.5, zorder=2,
                   label=f"{coh}: t = {lin['intercept']:.1f} + {lin['slope']:.2f}·Δq  (R²={lin['r2']:.3f})")
        ax[0].plot(dq[pause], dt[pause], linestyle="none", marker=st["marker"], markerfacecolor=st["color"],
                   markeredgecolor=st["color"], markersize=4.5, zorder=2,
                   label=f"{coh}: interval with ~3 s pause (n={int(pause.sum())}, fitted)")
        all_x += iv_x; n_fit += len(sweeps); n_excl += n_x
    if all_x:
        ax[0].plot([x["dq_max"] for x in all_x], [x["dt"] for x in all_x], linestyle="none", marker="x",
                   color="#B0B0B0", markeredgecolor="#B0B0B0", markersize=3.5, zorder=0, label=f"settle 2.5 s runs (n={n_excl}), not fitted")
    ax[0].set_xlabel("Largest joint move between consecutive views, Δq (deg)")
    ax[0].set_ylabel("Time between captures (s)"); ax[0].set_title(f"(a) Per-view time model from {n_fit} capture runs")
    ax[0].legend(loc="upper left", frameon=True, edgecolor="black", fancybox=False, fontsize=9)

    # policies: reconstruct the acquisition order of each strategy per petiole (same code paths as the paper)
    tp = list(csv.DictReader(open(throughput_csv, encoding="utf-8")))
    petmap = {(r["plant"], r["pid"]): r for r in pet}
    cohort_name = {"laboratory": "Laboratory", "greenhouse": "Greenhouse"}
    strat_order = ["passive", "random", "coverage", "parallax_causal", "uncertainty_causal"]
    strat_label = {"passive": "Sequential", "random": "Random", "coverage": "Coverage-NBV",
                   "parallax_causal": "Max-parallax", "uncertainty_causal": "Uncertainty-opt.", "full": "Full sweep"}
    plant_union = defaultdict(set); plant_reached = defaultdict(int); plant_pet = defaultdict(int)
    for row in tp:
        key = (row["plant"], row["pid"]); r = petmap.get(key)
        if r is None:
            continue
        coh = cohort_name[row["cohort"]]; s = row["strategy"]; n = int(row["n_views"])
        pool = vs._obs_pool(r["cr"], r["ju"])
        if row["views"]:
            order = [int(x) for x in row["views"].split()]
        elif s == "passive":
            order = list(pool)[:n]
        elif s == "random":
            order = np.random.default_rng(0).permutation(pool).tolist()[:n]
        elif s == "coverage":
            order = vs._coverage_order(r["views"], pool)[:n]
        else:
            order = []
        plant_union[(coh, row["plant"], s)] |= set(order)
        plant_pet[(coh, row["plant"], s)] += 1
        plant_reached[(coh, row["plant"], s)] += int(row["reached"] == "True")

    def path_time(coh, idx_sorted):
        lin, pj = models[coh]
        t = 0.0
        for a, b in zip(idx_sorted, idx_sorted[1:]):
            dq = float(np.abs(pj[b] - pj[a]).max())
            t += lin["intercept"] + lin["slope"] * dq
        return t + (lin["intercept"] if idx_sorted else 0.0)   # first capture after arrival

    def tsp_time(coh, idx):
        """Open-path order minimising modelled move time: nearest-neighbour + 2-opt."""
        lin, pj = models[coh]
        if len(idx) < 3:
            return path_time(coh, sorted(idx)), sorted(idx)
        c = {(a, b): lin["intercept"] + lin["slope"] * float(np.abs(pj[b] - pj[a]).max()) for a in idx for b in idx if a != b}
        best_t, best_o = None, None
        for start in idx:
            o = [start]; rem = set(idx) - {start}
            while rem:
                nx = min(rem, key=lambda b: c[(o[-1], b)]); o.append(nx); rem.discard(nx)
            improved = True
            while improved:
                improved = False
                for i in range(1, len(o) - 1):
                    for j in range(i + 1, len(o)):
                        new = o[:i] + o[i:j + 1][::-1] + o[j + 1:]
                        if sum(c[(a, b)] for a, b in zip(new, new[1:])) < sum(c[(a, b)] for a, b in zip(o, o[1:])) - 1e-9:
                            o = new; improved = True
            t = sum(c[(a, b)] for a, b in zip(o, o[1:])) + lin["intercept"]
            if best_t is None or t < best_t:
                best_t, best_o = t, o
        return best_t, best_o

    table = []
    for coh in ("Laboratory", "Greenhouse"):
        plants = sorted({k[1] for k in plant_union if k[0] == coh})
        sweep_n = 27 if coh == "Laboratory" else 31
        for s in strat_order + ["full"]:
            times, reached, npet, times_opt = [], [], [], []
            for p in plants:
                if s == "full":
                    idx = list(range(sweep_n)); rc = sum(1 for r in pet if r["cohort"] == coh and r["plant"] == p and r["ci_full"] <= TAU)
                    npet.append(sum(1 for r in pet if r["cohort"] == coh and r["plant"] == p))
                else:
                    idx = sorted(plant_union[(coh, p, s)]); rc = plant_reached[(coh, p, s)]; npet.append(plant_pet[(coh, p, s)])
                times.append(path_time(coh, idx)); reached.append(rc); times_opt.append(tsp_time(coh, idx)[0])
            times = np.array(times); reached = np.array(reached); npet = np.array(npet); times_opt = np.array(times_opt)
            table.append(dict(cohort=coh, strategy=s, label=strat_label[s], n_plants=len(plants),
                              poses_per_plant=float(np.mean([len(plant_union[(coh, p, s)]) if s != "full" else sweep_n for p in plants])),
                              plant_time_s_mean=float(times.mean()), plant_time_s_sd=float(times.std(ddof=1)),
                              reached_pct=100 * reached.sum() / npet.sum(),
                              plants_per_hour=3600 / times.mean(),
                              spec_petioles_per_hour=3600 * reached.sum() / times.sum(),
                              plant_time_opt_s_mean=float(times_opt.mean()),
                              saving_pct=100 * (1 - times_opt.sum() / times.sum()),
                              spec_petioles_per_hour_opt=3600 * reached.sum() / times_opt.sum()))
    res["policy_table"] = table
    with open(OUT / "D_cycle_time_by_policy.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys())); w.writeheader(); w.writerows(table)

    # (b) bars: spec-conforming petioles per hour, lab/gh
    labs = [strat_label[s] for s in strat_order + ["full"]]
    x = np.arange(len(labs)); wdt = 0.36
    for i, coh in enumerate(("Laboratory", "Greenhouse")):
        vals = [next(t for t in table if t["cohort"] == coh and t["strategy"] == s)["spec_petioles_per_hour"] for s in strat_order + ["full"]]
        ax[1].bar(x + (i - 0.5) * wdt, vals, wdt, facecolor="white" if i == 0 else RED, edgecolor=BLACK if i == 0 else RED,
                  hatch="" if i == 0 else "", label=coh, lw=1.2)
    ax[1].set_xticks(x); ax[1].set_xticklabels(labs, rotation=20, fontsize=9.5)
    ax[1].set_ylabel(f"Petioles measured to spec (CI ≤ {TAU:.0f} mm) per hour")
    ax[1].set_title("(b) Effective throughput by viewpoint policy (τ = 5 mm)")
    ax[1].set_ylim(0, 85)
    ax[1].legend(loc="upper right", frameon=True, edgecolor="black", fancybox=False, fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "figD_cycle_time.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for i, coh in enumerate(("Laboratory", "Greenhouse")):
        st = COH_STYLE[coh]
        t0 = [next(t for t in table if t["cohort"] == coh and t["strategy"] == s)["plant_time_s_mean"] for s in strat_order + ["full"]]
        t1 = [next(t for t in table if t["cohort"] == coh and t["strategy"] == s)["plant_time_opt_s_mean"] for s in strat_order + ["full"]]
        ax.plot(x, t0, color=st["color"], marker=st["marker"], markerfacecolor="none", lw=1.6, label=f"{coh}: taught-path order")
        ax.plot(x, t1, color=st["color"], marker=st["marker"], lw=1.6, ls="--", label=f"{coh}: time-optimised order")
    ax.set_xticks(x); ax.set_xticklabels(labs, rotation=20, fontsize=9.5)
    ax.set_ylabel("Plant cycle time, motion + capture (s)"); ax.set_ylim(0, 300)
    ax.set_title("Visiting order matters more than the policy")
    ax.legend(loc="lower left", frameon=True, edgecolor="black", fancybox=False, fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "figD2_sequencing.png"); plt.close(fig)
    return res


# --------------------------------------------------------------------------- main
def main():
    pet = load_petioles()
    print(f"petioles: {len(pet)}  lab={sum(r['cohort']=='Laboratory' for r in pet)} gh={sum(r['cohort']=='Greenhouse' for r in pet)}", flush=True)
    A = block_a(pet); print("A done", flush=True)
    B, rows = block_b(pet); print("B done", flush=True)
    C = block_c(rows); print("C done", flush=True)
    D = block_d(pet); print("D done", flush=True)
    out = dict(A_bias_linearity=A, B_repeatability=B, C_interval_gate=C, D_cycle_time=D,
               settings=dict(N_MC=N_MC, TAU=TAU, TOL=TOL, BUDGET=BUDGET))
    (OUT / "msa_results.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print("->", OUT)


if __name__ == "__main__":
    main()
