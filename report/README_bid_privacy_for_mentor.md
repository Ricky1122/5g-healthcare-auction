# Bid privacy for the TPE — scoping memo

**To:** Satendra Kumar
**From:** Nakshatra Kanchan
**Re:** Can the third-party entity clear the market without ever learning the bid amounts?

**Status:** research scoping. No cryptography exists in the code today. Paper 1 remains a
cleartext TPE. This memo argues that bid privacy is a **second paper**, explains why the
problem is harder (and more interesting) than it first looks, and proposes a concrete
two-tier design.

This memo supersedes an earlier version that dismissed proxy re-encryption as a
"category error." That was too strong, and Section 6 corrects it.

---

## 1. The question in one paragraph

Our double auction was built so the TPE never sees the **utility functions** \(S_w\) (HSP)
and \(T_k\) (BS). Agents report only scalars — \(\varrho_{wk}\) from each HSP,
\(\zeta_{kw}\) from each BS — and the TPE clears on those. But the TPE *does* see those
scalars in cleartext. The question is whether we can keep the identical allocation while
hiding the numbers themselves. The answer is yes, but not by encrypting the bids: the
leak is not in the messages, it is in the **allocation rule and the price path**. Fixing it
means changing what the TPE is allowed to compute, not just what it is allowed to read.

---

## 2. Why this is not the usual sealed-bid auction problem

Most secure-auction papers protect a *sealed envelope*. Everyone submits one bid, the
auctioneer opens all envelopes once, announces a winner, and the protocol is over. The
literature here is mature: PP-SPEC, PROST, SDSA, ARMOR, PPS all hide bids during that
single winner-determination step.

Our market is a different animal on two counts:

- **The resource is divisible.** We are not asking "who wins the spectrum licence." We are
  asking "how do we split uplink rate across six BS–HSP links," and every link gets a
  positive share.
- **The price is discovered by haggling.** There is no single opening. The TPE posts prices,
  both sides respond with quantities, the TPE nudges prices toward the gap, and this repeats.
  Our 3 HSP × 2 BS market converges after **50 rounds**.

> **Analogy.** A sealed-bid auction is a locked ballot box opened once. Ours is a
> *haggling floor with a public price board*. You can keep every whisper private and still
> lose the game, because the board itself moves in response to those whispers.

That distinction is the whole memo. It also tells us which prior work is actually relevant:
not the sealed-bid crypto-auction line, but **encrypted iterative optimisation** (privacy-
preserving ADMM and distributed optimal power flow) and one paper on **divisible double
auctions** that explicitly protects "bid amount and prices in different iterations."

---

## 3. The leak, demonstrated on files already in this repository

This is the part worth showing rather than asserting. Convention: CSV rows are base
stations, columns are HSPs.

### 3.1 The bid table is already published — it is called `payment_wk.csv`

The paper's OPT2 stationarity condition is

\[
\tilde d_{wk} = \frac{\varrho_{wk}}{\tilde\mu_{wk}}
\qquad\Longrightarrow\qquad
\varrho_{wk} = \tilde d_{wk}\cdot\tilde\mu_{wk}
\]

that is, **bid = quantity × price**. But quantity × price is exactly what we already write
out as the payment. Taking the BS1–HSP1 link from `final result/market/`:

| quantity `d_wk.csv` | price `pi_wk.csv` | product | `payment_wk.csv` |
| --- | --- | --- | --- |
| 1.50078 | 0.36000 | 0.54028 | 0.54028 |

Verified across all six links: `pi ⊙ d == payment` to machine precision (max error `0.0`).
And because the pricing rule is pay-your-bid, \(\Gamma_w=\sum_k\varrho_{wk}\), the column
sums are each HSP's total bid: **1.43716 / 1.32671 / 0.63804**, summing to the reported
total payment 3.4019.

So under our own mechanism, the published payment table *is* the bid table. Asking the TPE
to settle payments but not learn bids is asking it to process a payslip without learning a
salary.

### 3.2 Worse: the clinical signal is recoverable too

The implemented solver uses \(d_{wk}=[\rho_{wk}/(\pi_{wk}+\lambda_k)-1/h_{wk}]_+\).
Invert it — \(\rho_{wk}=(d_{wk}+1/h_{wk})(\pi_{wk}+\lambda_k)\) — using only quantities,
prices and duals that we publish:

```
recovered rho          true rho (rho_wk.csv)
0.99877  0.81338  0.60063      1.0      0.81395  0.60128
0.99857  0.79869  0.59375      0.99841  0.79854  0.59360
```

Agreement to about 0.1%. The residual is only because the released table is the midpoint of
demand and supply rather than demand itself.

This matters far more than commercial confidentiality. Recall \(\rho_{wk}\) is the
max-normalised **criticality mass** on coverage set \(B_{wk}\) — LSTM-forecast patient
severity, aggregated per cell. Recovering \(\rho\) to three significant figures means
recovering **where the critically ill patients are, cell by cell**. That is health
information leaking out of a spectrum market, and it is a much stronger motivation for the
paper than "hospitals want to keep their bids secret."

### 3.3 The price path leaks even if all of the above is encrypted

Suppose we hide bids, quantities *and* payments, and publish only prices. The price update
is public and invertible:

```347:351:src/05_optimizer.py
        excess = d - r
        step = cfg.step0 / np.sqrt(t)
        scale = 1.0 + np.abs(excess)
        pi = np.clip(pi + step * excess / scale, pi_floor, cfg.pi_max)
```

Anyone who watches \(\pi\) between two rounds recovers the excess demand \(d-r\) exactly on
every link, because \(\alpha_t\) and the squashing map are known. Each round then contributes
fresh equations in the *same* unknown bids. Over 50 rounds and 6 links that is roughly
300 equations constraining a couple of dozen unknowns — the bid vector is not merely
recoverable, it is wildly over-determined.

> **Analogy.** You never let anyone see the thermometer. But the thermostat dial is on the
> wall, and everyone knows the control rule. Watch the dial get nudged up and down for a
> hundred minutes and you can reconstruct the temperature curve precisely. Encrypting the
> thermometer was never the point.

**Consequence:** prices cannot be public if bids are to stay private. This single sentence
is why the problem needs a protocol and not a library call, and it is the contribution hook
for paper 2.

---

## 4. What we are actually trying to achieve

Two requirements that are usually conflated, and need different tools:

1. **Confidentiality** — the TPE clears the market without ever holding a bid in plaintext.
2. **Verifiability** — agents can nonetheless check the TPE applied the rule honestly and
   did not quietly favour someone.

Requirement 2 is the "can it still verify the amounts?" part of the original question. It is
*not* solved by the same primitive as requirement 1.

---

## 5. The structural fact that makes this cheap

Here is the good news, and it comes from reading our own solver rather than the literature.

Every hard, nonlinear step in the clearing loop is **already local to an agent that knows
its own bid anyway**:

- In `solve_opt1`, HSP \(k\)'s bisection for \(\lambda_k\) touches only column \(k\) — its own links.
- In `solve_opt2`, BS \(w\)'s bisection for \(\mu_w\) touches only row \(w\) — its own links.

So the division, the \([\cdot]_+\) projection, and the 48-step bisection all belong to the
buyer or the seller. What is left for the TPE is purely **linear**: form \(d-r\), scale by
\(\alpha_t\), add to \(\pi\).

> **Analogy.** The TPE does not need to be a mathematician who understands the bids. It only
> needs to be a **blindfolded adding machine**: hand it two sealed numbers, it returns their
> sealed sum without ever opening either. Addition on sealed numbers is precisely what
> additively homomorphic encryption (Paillier) gives us, cheaply and *exactly*.

This is why the design below is realistic for a B.Tech prototype rather than a multi-year
cryptography project.

---

## 6. On proxy re-encryption — correcting the earlier memo

PRE (AFGH, BBS and successors) is **access delegation**. Alice encrypts to herself; a proxy
holding a re-encryption key converts the ciphertext so Bob can decrypt it. The proxy learns
nothing.

> **Analogy.** PRE is a courier who can re-address a locked box so that a *different*
> recipient's key opens it. Genuinely useful. But that courier cannot **count the cash
> inside**, and cannot **combine two boxes into a box holding the sum**. Our TPE needs to add.

So PRE cannot carry the market arithmetic — that part of the earlier memo stands. What the
earlier memo got wrong was calling it a category error. There is a published spectrum-auction
framework (Hu et al., *High-Confidence Computing*, 2021) that uses a **modified AFGH
re-encryption scheme specifically so that any bidder can verify the auction result without
compromising other bidders' privacy**, while homomorphic encryption does the computation.

**Correct framing:** PRE answers requirement 2 (verifiability), not requirement 1
(confidentiality). Your instinct pointed at a real tool; it just belongs in a different slot
of the design.

---

## 7. Proposed design: two privacy tiers

Rather than commit to one threat model, we report both as a tradeoff — the structure used by
Haydon et al. (ACISP 2024) for private network slicing. A regulator (e.g. TRAI) serves as the
second, non-colluding party; nearly every practical scheme needs one, and in a 5G healthcare
market that role is institutionally natural.

### Tier 1 — TPE-blind (Paillier only)

The TPE holds only a public key and the encrypted price state. Each round:

1. The regulator decrypts \(\pi_{wk}\) and delivers it **only to the two endpoints of that link**.
2. HSP and BS best-respond locally in cleartext — they already know their own bids.
3. They return \(\mathrm{Enc}(d_{wk})\) and \(\mathrm{Enc}(r_{kw})\).
4. The TPE homomorphically forms \(\mathrm{Enc}(\pi+\alpha_t(d-r))\) and iterates.

The TPE never holds a plaintext at any point. Two design details:

- **Run a fixed number of rounds** instead of testing \(\|d-r\|<\text{tol}\). The stopping test
  would itself leak; a fixed count is data-oblivious, and our market already terminates in 50.
- **Let the agent apply the price floor** on decryption, since \([\cdot]_+\) is not homomorphic
  and the endpoint needs the clipped value anyway.

*Cost:* a wrapper around the existing solver using `phe`. Roughly 20 s for 250 rounds against
~0.2 s cleartext — and **bit-identical welfare**, because fixed-point Paillier is exact.

*Residual leakage, stated honestly:* payments must settle, so the regulator learns
\(\Gamma_w=\sum_k\varrho_{wk}\). And an HSP watching its own link price for 50 rounds can
still infer its counterparty BS's cost. Tier 1 blinds the TPE; it does not blind competitors.

### Tier 2 — Fully blind (2-party MPC)

To close the peer leak, prices cannot be revealed even to the endpoints, so the best response
itself must run on secret-shared values. That needs secure division, secure comparison and
secure bisection — a rewrite of the solver in the MP-SPDZ DSL, not Paillier.

> **Analogy.** Tier 1 puts the auctioneer in a blindfold. Tier 2 puts every participant in a
> separate soundproof booth.

The cost driver is not arithmetic but **round depth**: 48 sequential bisection steps × 250
rounds ≈ 12,000 round trips, so we would be latency-bound. There is a clean algorithmic fix
that is a contribution in its own right — our capacity constraint binds on every cell, and a
constraint known to bind admits a closed-form water-filling dual, removing the bisection
entirely.

*Feasibility is not in doubt at our scale.* Haydon et al. handled **32 slices on 2 base
stations in under 5 ms** at moderate privacy and under 7 s at maximum privacy. We have
6 links.

### Verifiability layer (both tiers)

- **Pedersen commitments + Bulletproofs range proofs** so an HSP proves \(0\le\varrho_{wk}\le\)
  budget without revealing it (cf. Riggs, IACR 2023/1336: "bid ≤ collateral without disclosure").
- Pedersen is additively homomorphic, so \(\sum_k\mathrm{Com}(\varrho_{wk})=\mathrm{Com}(\Gamma_w)\)
  checks the payment against the bid sum without opening anything.
- **Outcome correctness comes almost free** because we deliberately kept the TPE's job linear:
  publish the ciphertext transcript, and any agent can replay the homomorphic operations and
  check equality. No ZK circuit needed. PRE (Section 6) is the alternative if we prefer
  bidder-side verification without a transcript.

---

## 8. What we will explicitly rule out, and say so

| Approach | Why it fails here |
| --- | --- |
| Encrypt in transit, decrypt at the TPE | TPE sees cleartext bids again. Does not address the threat model. |
| Commit–reveal | Prevents bid sniping, but the TPE sees everything at reveal time. |
| Differential privacy on bids | Distorts the KKT match and breaks the incentive-compatibility claim — our welfare number would move. Wrong tool for an exactness-critical mechanism. |
| Whole loop under FHE | Threshold decryption is now fast (Ajax, 2025: 0.9 ms), but comparison-heavy inner loops still benchmark around 0.07 orders/s in dark-pool work. Overkill and too slow. |
| Publish \((d,r,\pi,\lambda,\mu)\) and simply not name the bids | Algebraically invertible — demonstrated in Section 3. |

---

## 9. Recommendation on scope

**Bid privacy should be paper 2, not a section of paper 1.**

The reasoning is specific to our situation: the encrypted market is designed to produce the
*identical* allocation, so it contributes nothing to paper 1's experimental story — welfare
and cleared rate stay exactly as reported — while importing a threat model, a cryptography
section and a benchmark section. Paper 1's claim is dynamic preference from LSTM forecasts,
and that story is complete.

Paper 1 should, however, get a **corrected future-work paragraph**. The current text says
hiding the bid scalars is future work, which is true but misses the price-path argument of
Section 3.3 — the actual reason it is hard, and the hook for paper 2.

### Step 0, which we have not yet budgeted

`src/05_optimizer.py` is **not a bid protocol**. It loads \(\rho\), cost and \(\beta\)
directly and runs the centralized cousin; `main.tex` acknowledges this, and the "bid
evolution" figures compute \(\zeta_{kw}=\omega_{kw}(c/r_{kw}+\beta)\) *post hoc* from the
allocation rather than from bids any agent submitted.

So before anything can be encrypted, the genuine bid-based dual iteration has to exist:
agents submitting \(\varrho,\zeta\), the TPE clearing on those reports alone. This is modest
work and independently valuable — it is the artifact that lets us test the truthful-bidding
claim empirically instead of by construction.

**Proposed order:** (0) real bid-based loop → (1) Tier 1 Paillier wrapper + timing plots
against the cleartext baseline → (2) Tier 2 in MP-SPDZ → (3) verifiability layer.

---

## 10. Key references

**Closest prior work**
- Haydon, Lai, Yuan, Abuadbba, Rudolph. *Towards Private Multi-operator Network Slicing.*
  ACISP 2024, LNCS 14897, 359–379. — 2-party MP-SPDZ slicing; three privacy tiers; our cost model.
- *Privacy preserving divisible double auction with a hybridized TEE-blockchain system.*
  Cybersecurity, 2021. doi:10.1186/s42400-021-00100-x — divisible, iterative, protects
  per-iteration bid profiles; uses hardware trust and PSP rather than a KKT dual loop.

**Encrypted iterative optimisation (the right analogue for our loop)**
- Zhang, Ahmad, Wang. *ADMM Based Privacy-Preserving Decentralized Optimization.* IEEE TIFS
  14(3), 2018.
- Wu, Zhao, Zhang. *Privacy-Preserving Distributed Optimal Power Flow With Partially
  Homomorphic Encryption.* IEEE TSG 12(5), 2021.

**Sealed-bid spectrum auctions (adjacent, different problem)**
- PP-SPEC, IEEE IoT-J 2020; PROST, IEEE TIFS 2018; SDSA, arXiv:1810.07880; ARMOR, IEEE TMC
  2018; PPS, arXiv:1307.7792 — HE + garbled circuits + secret sharing, one-shot winner
  determination, "auctioneer + agent" two-party split.

**Verifiability**
- Hu et al. *A cloud-based framework for verifiable privacy-preserving spectrum auction.*
  High-Confidence Computing, 2021. doi:10.1016/j.hcc.2021.100037 — modified AFGH PRE for
  result verification. **This is the correct home for the PRE idea.**
- *Riggs: Decentralized Sealed-Bid Auctions.* IACR ePrint 2023/1336 — timed commitments plus
  range proofs.

**Theory: why some leakage is unavoidable**
- *Private From Whom? Minimal Information Leakage in Auctions.* arXiv:2511.10349 — for a fixed
  choice rule, revealing less to the auctioneer forces revealing more to bidders. Formal
  justification for reporting tiers instead of claiming one optimum.

**Ruled out**
- *Indifferential Privacy … Dark Pool Auctions.* arXiv:2502.13415 — FHE throughput benchmarks.
- *Ajax: Fast Threshold FHE without Noise Flooding.* IACR ePrint 2025/1834.
