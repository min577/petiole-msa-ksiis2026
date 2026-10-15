# petiole-msa-ksiis2026

Analysis script and inputs for the poster/paper **"Measurement System Analysis of a Robotic Multi-View
Strawberry Petiole-Length Gauge for Smart-Farm Phenotyping"**
(스마트팜 표현형 계측을 위한 로봇 다중뷰 딸기 엽병장 측정시스템의 측정시스템분석),
2026 Fall Conference of the Korean Society of Industrial Information Systems (KSIIS).

Minwoo Kim, Daehyun Jung - Department of Smart Farm Science, Kyung Hee University

No new measurements were taken. The analysis re-uses the 147 petioles (92 laboratory, 55 greenhouse)
captured for the companion study, whose code and data are published at
[min577/uncertainty-optimal-viewpoints](https://github.com/min577/uncertainty-optimal-viewpoints)
(Zenodo DOI [10.5281/zenodo.21995472](https://doi.org/10.5281/zenodo.21995472)).

## What the script computes

| Block | Question | Output |
|---|---|---|
| A | Bias and linearity vs caliper (t-test, error-on-reference regression, Bland-Altman LoA) | `figA_bias_linearity.png` |
| B | Repeatability from disjoint 4-view subsets for four view-selection procedures (EV, %GRR, ndc), with and without the interval gate | `figB_repeatability.png`, `B_subset_measurements.csv` |
| C | Interval gate: acceptance yield vs quality of accepted measurements | `figC_interval_gate.png` |
| D | Cycle time: per-view time model from capture logs, time per plant by policy, visiting-order optimisation (nearest neighbour + 2-opt) | `figD_cycle_time.png`, `figD2_sequencing.png`, `D_cycle_time_by_policy.csv` |

All numbers are written to `out/msa_results.json`. The outputs in `out/` are the ones used on the poster.

## Run

```bash
git clone https://github.com/min577/uncertainty-optimal-viewpoints.git
git clone https://github.com/min577/petiole-msa-ksiis2026.git
cd petiole-msa-ksiis2026
pip install -r requirements.txt
python msa_analysis.py          # about 3 min; Monte-Carlo steps are seeded, results reproduce exactly
```

If the code repository is not cloned next to this one, point to it with `UOV_DIR=/path/to/uncertainty-optimal-viewpoints`.
Tested with Python 3.11.

## Inputs

`inputs/` holds only the small non-image files the analysis needs (about 2.6 MB):

- `sessions/<plant>/` - `transforms.json` (camera intrinsics and hand-eye camera poses per view),
  `mv_clicks.json` (crown/junction pixels per view), `gt.json` (caliper reference, mm).
  `plant*` = laboratory, `greenhouse_*` = in-situ greenhouse.
- `runs/{lab,greenhouse}/<run>/` - robot capture-run `manifest.json` (waypoints, joint angles, speed, settle time)
  and `capture_mtimes.json` (image file timestamps, used to fit the per-view time model).

The per-petiole policy table (`data/ablation/_throughput5_causal.csv`) is read from the code repository.

## Caveats

- Repeatability is computed from disjoint view subsets of the **same** capture, which share the operator's clicks
  and the robot poses. It is a lower bound; an independent re-capture Gage R&R has not been done.
- The caliper reference is a single reading per petiole.
- Re-ordered visiting paths in block D are simulated; collisions and reachability were not checked.
