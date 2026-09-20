# 3x2 experiment — 3 HSPs × 2 base stations

Topology for this folder: **W = 3 buyers** and **K = 2 sellers**.
Capacity is **5 Mbps per cell**, reluctance is the frozen one-shot radio table, preference is max-normalized.
Fig. 2 subgradient steps: **0.01, 0.02, 0.04, 0.08**.

## How patients are allocated

Each eICU vital window is one uplink user. Users are placed uniformly on a 2 km × 2 km map (seed 42), associated to the strongest of 2 cells by max RSRP, and subscribed to HSP1, HSP2, HSP3 in a 5:4:3 share (seed 42+7) so hospitals sit in different preference bands. No disease-to-hospital map.

There are **74454** uplink users (eICU Demo vital windows). Unique ICU stays: **1821**.

### Users per hospital (5:4:3 split, seed 42)

| HSP | Users | Share |
| --- | ---: | ---: |
| HSP1 | 31023 | 41.7% |
| HSP2 | 24818 | 33.3% |
| HSP3 | 18613 | 25.0% |

### Users on each (BS, HSP) link

|  | HSP1 | HSP2 | HSP3 | BS total |
| --- | ---: | ---: | ---: | ---: |
| BS1 | 15416 | 12480 | 9421 | 37317 |
| BS2 | 15607 | 12338 | 9192 | 37137 |

Map (1.6k-user sample): `preference_sweep/fig_allocation_map.png`.

## Full-market preference and clearing

Official split is **5:4:3**. Preference rho = (C / mean C) * (1 + mean c), then **max-normalized** so the largest link is 1.

Preference matrix (`market/rho_wk.csv`):

```
,HSP1,HSP2,HSP3
BS1,1.0,0.81395185451342,0.6012773643515398
BS2,0.9984078017605572,0.7985354201290896,0.5935981209905625
```

Frozen reluctance (`market/omega_wk.csv`):

```
,HSP1,HSP2,HSP3
BS1,0.469699993299599,0.4707154751099731,0.4706818951803287
BS2,0.6143955009848018,0.6140127995017282,0.6147842341250133
```

| Metric | Value |
| --- | --- |
| Welfare | 7.477 |
| Total rate (Mbps) | 10.005 |
| Total payment | 3.402 |
| Gap | 9.603e-04 |
| Converged | True |

Auction traces: `figures/fig2_convergence.png` … `fig8_preference_vs_payment.png`.

Selected full-cohort ratios:

- **1:1:1** -> 24818/24818/24818 users: mean rho = HSP1 0.988, HSP2 0.991, HSP3 0.993
- **4:3:3** -> 29782/22336/22336 users: mean rho = HSP1 0.994, HSP2 0.749, HSP3 0.749
- **5:4:3** (official market) -> 31023/24818/18613 users: mean rho = HSP1 0.999, HSP2 0.806, HSP3 0.597
- **5:4:1** -> 37227/29782/7445 users: mean rho = HSP1 0.996, HSP2 0.795, HSP3 0.197
- **7:2:1** -> 52118/14891/7445 users: mean rho = HSP1 0.992, HSP2 0.282, HSP3 0.144

## Folder

```
3x2 experiment/
  README.md
  market/                 rho, omega, clearing, assignment
  figures/                auction Figs 2–8
  preference_sweep/       split experiment
```

Re-run:

```bash
python src/11_w3k2_final.py --only 3x2
```
