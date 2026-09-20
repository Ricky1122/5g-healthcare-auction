# 5x3 experiment — 5 HSPs × 3 base stations

Topology for this folder: **W = 5 buyers** and **K = 3 sellers**.
Capacity is **5 Mbps per cell**, reluctance is the frozen one-shot radio table, preference is max-normalized.
Fig. 2 subgradient steps: **0.01, 0.02, 0.04, 0.08**.

## How patients are allocated

Each eICU vital window is one uplink user. Users are placed uniformly on a 2 km × 2 km map (seed 42), associated to the strongest of 3 cells by max RSRP, and subscribed to HSP1, HSP2, HSP3, HSP4, HSP5 in a 5:4:3:2:1 share (seed 42+7) so hospitals sit in different preference bands. No disease-to-hospital map.

There are **74454** uplink users (eICU Demo vital windows). Unique ICU stays: **1821**.

### Users per hospital (5:4:3:2:1 split, seed 42)

| HSP | Users | Share |
| --- | ---: | ---: |
| HSP1 | 24818 | 33.3% |
| HSP2 | 19854 | 26.7% |
| HSP3 | 14891 | 20.0% |
| HSP4 | 9927 | 13.3% |
| HSP5 | 4964 | 6.7% |

### Users on each (BS, HSP) link

|  | HSP1 | HSP2 | HSP3 | HSP4 | HSP5 | BS total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BS1 | 6244 | 4884 | 3755 | 2524 | 1192 | 18599 |
| BS2 | 9331 | 7483 | 5530 | 3687 | 1833 | 27864 |
| BS3 | 9243 | 7487 | 5606 | 3716 | 1939 | 27991 |

Map (1.6k-user sample): `preference_sweep/fig_allocation_map.png`.

## Full-market preference and clearing

Official split is **5:4:3:2:1**. Preference rho = (C / mean C) * (1 + mean c), then **max-normalized** so the largest link is 1.

Preference matrix (`market/rho_wk.csv`):

```
,HSP1,HSP2,HSP3,HSP4,HSP5
BS1,0.668320173700553,0.5406168803660032,0.3997964704965316,0.2716572091894974,0.13056988453260143
BS2,1.0,0.810683015249635,0.6073142235072766,0.398910099279693,0.1941123246908643
BS3,0.9984927412567312,0.8065263275886658,0.6091341196323485,0.39223768559497285,0.20533598884414556
```

Frozen reluctance (`market/omega_wk.csv`):

```
,HSP1,HSP2,HSP3,HSP4,HSP5
BS1,0.3893842044336741,0.39020621752613116,0.3908671833087493,0.38998611129942995,0.39081968988500637
BS2,0.5627860492480317,0.5623865734948533,0.563096979179204,0.5622126779099973,0.5621197806610064
BS3,0.5582788331735243,0.5601252330438163,0.559994987399445,0.5581298090260519,0.5572575488733545
```

| Metric | Value |
| --- | --- |
| Welfare | 8.625 |
| Total rate (Mbps) | 14.981 |
| Total payment | 4.125 |
| Gap | 9.992e-04 |
| Converged | True |

Auction traces: `figures/fig2_convergence.png` … `fig8_preference_vs_payment.png`.

Selected full-cohort ratios:

- **1:1:1:1:1** -> 14891/14891/14891/14891/14890 users: mean rho = HSP1 0.871, HSP2 0.873, HSP3 0.872, HSP4 0.866, HSP5 0.865
- **5:4:3:2:1** (official market) -> 24818/19854/14891/9927/4964 users: mean rho = HSP1 0.889, HSP2 0.719, HSP3 0.539, HSP4 0.354, HSP5 0.177
- **7:5:3:2:1** -> 28954/20682/12409/8273/4136 users: mean rho = HSP1 0.887, HSP2 0.633, HSP3 0.381, HSP4 0.253, HSP5 0.124
- **15:1:1:1:1** -> 58779/3919/3919/3919/3918 users: mean rho = HSP1 0.890, HSP2 0.058, HSP3 0.062, HSP4 0.059, HSP5 0.060

## Folder

```
5x3 experiment/
  README.md
  market/                 rho, omega, clearing, assignment
  figures/                auction Figs 2–8
  preference_sweep/       split experiment
```

Re-run:

```bash
python src/11_w3k2_final.py --only 5x3
```
