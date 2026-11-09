# What predicts Ethereum 10 seconds to 5 minutes ahead, and why: research log

This project started as a port of `btc-predict`. The rule was **not to copy BTC's conclusions**:
every BTC finding was treated as a hypothesis and re-tested on ETH, and ETH-specific mechanisms
were tested from scratch.

All model numbers are out-of-sample walk-forward results on Binance ETHUSDT (spot) and the
ETHUSDT USD-M perpetual, unless stated otherwise:
* decisions: three DEV months (2025-11, 2026-01, 2026-03), monthly retraining, purging;
* checks: three untouched confirmation months (2025-12, 2026-02, 2026-04);
* headline results: the frozen holdout (2026-05 → 2026-10), see the README.

Mechanism tests use pre-holdout days only. Their scripts are in `scripts/research/mechanisms/`.

## 0. The central question: is ETH more predictable than BTC?

The working hypothesis at the start was "ETH should be more predictable, because BTC leads it."
Each reason ETH *could* be more predictable was tested separately, on both assets, with identical
code and data handling.

| hypothesis | test (pre-holdout data) | result |
|---|---|---|
| BTC leads ETH at seconds scale | cross-correlation of 1 s returns, 22 days | **No, the reverse.** ETH perp → BTC perp at +1 s: 0.139; BTC → ETH: 0.063. Spot: 0.148 vs 0.086. |
| BTC leads ETH at slower scales | lead-lag of 1-min and 5-min bars, 363 days | **No.** 1 min: ETH → BTC 0.028, BTC → ETH 0.004. 5 min: neither (−0.002 vs −0.003). |
| ETH is predictable *when BTC moves first* (BTC-specific news) | events where BTC perp moves > 4σ in 10 s and ETH has not followed | **Weakly.** ETH follows 54% at +10 s (+0.50 bp) and 50% by +60 s. In the mirror case BTC follows ETH 58% / 57% / 55% (+0.73 → +0.96 bp). |
| ETH prices are less efficient (more autocorrelation) | lag-1 autocorrelation and variance ratios from 1 s to 15 min, 363 days | **No.** BTC deviates more from a random walk at every scale up to 30 s. 1 s autocorrelation: BTC 0.118, ETH 0.089, and ETH is larger in 0 of 12 months. VR(60 s): 1.59 vs 1.35. |
| ETH 1-min returns have more structure | permutation entropy (order 3) of 1-min returns | **No.** ETH is closer to pure noise in 10 of 12 months (0.99923 vs 0.99888). |
| ETH has more leverage → stronger continuation after big moves | signed next move after a > 3σ minute | **No.** Spot: BTC +0.059σ, ETH +0.029σ at 1 min; perps are equal (+0.003σ). |
| ETH's moves are bigger, so a signal is worth more | 1-min volatility, 44 days | **Yes.** ETH 8.9 bp vs BTC 5.7 bp (1.56×). Fees are in bp, so the same IC buys ~1.5× more edge per trade. |

**Why ETH is not more predictable at these horizons.** Short-horizon predictability comes almost
entirely from *lagging* prices catching up with *leading* ones. ETH is the more active market:
* 26.4 perp trades per second vs 19.9 for BTC;
* 30% of perp seconds without a price change vs 40% for BTC;
* it leads BTC at 1 s and at 1 min.

So there is less "ahead of ETH" to forecast from. BTC benefited from cross-asset signals
precisely because *ETH* and Coinbase lead it; for ETH those same channels are weaker.

This agrees with recent high-frequency literature, which finds no consistent BTC lead and
often finds BTC in the lagging position. See
["Bitcoin, Ethereum, and the Ambiguity of Price Discovery: A Multi-Measure High-Frequency Analysis"](https://www.mdpi.com/1911-8074/19/9/678)
(JRFM 19(9):678). For efficiency comparisons, see
["On the efficiency and its drivers in the cryptocurrency market: the case of Bitcoin and Ethereum"](https://jfin-swufe.springeropen.com/articles/10.1186/s40854-023-00566-3)
(Financial Innovation) and the intraday multifractal study by
[Mensi et al. (2019)](https://www.sciencedirect.com/science/article/abs/pii/S1544612319300649).
Which of the two is more efficient flips with period and method; no study we found shows ETH
being more predictable intraday.

**Where ETH's predictability does live**
1. **ETH spot catching up with the ETH perp** (the basis family). That is the largest single
   source, exactly as for BTC.
2. **The Coinbase ETH premium.** It is the only cross-asset family that is individually
   significant for ETH.
3. **At 5 minutes, slow state features** (see §3). Unlike BTC, ETH's 5-minute forecasts improve
   when the model can condition on the day-scale state.
4. **Economically, the bp edge.** At equal hit rate ETH moves 1.56× more, so its edge in bp is
   equal to or larger than BTC's despite a lower IC.

## 1. Univariate evidence, ETH vs BTC (pre-holdout months, identical feature code)

Best feature per family: mean monthly IC with the scale-free spot +1 m target (share of months
with that sign). From `scripts/research/mechanisms/family_ic_scan.py`; full tables in
`reports/research/family_ic_{ETH,BTC}.csv`.

| family | BTC | ETH |
|---|---|---|
| perp–spot basis | 0.106 (100%) | 0.084 (100%) |
| Coinbase premium | 0.120 (100%) | 0.057 (100%) |
| other book of the same asset | 0.100 (FDUSD, 75%) | 0.054 (ETH/USDC, 100%) |
| 17-coin swarm | 0.072 (100%) | 0.040 (100%) |
| the other major's perp | 0.070 (ETH perp) | 0.023 (BTC perp) |
| the other major's spot | 0.063 (ETH) | 0.018 (BTC) |
| own perp microstructure | 0.073 | 0.043 |

Not one of the 101 features with |IC| > 0.01 is stronger for ETH than for BTC. For the **perp**
target, every ETH family is ≤ 0.02, because the ETH perp is the price-discovery venue.

## 2. ETH-specific hypotheses that were tested and rejected

* **Triangular arbitrage through the ETH/BTC book.** The idea: the ETHBTC book moves first
  (ratio traders, rotation flows) and arbitrage carries the move into ETHUSDT. The test used
  `tri_gap = log ETHBTC − (log ETHUSDT − log BTCUSDT)` on 44 days.
  * ETHBTC trades 1.2 times per second and **95% of its seconds have no price change**. It is a
    follower.
  * The gap is closed by the ETHBTC book itself: IC −0.35 on its own next move, negative in
    11 of 11 months.
  * The gap's IC for ETHUSDT is 0.002 at 10 s and 0.000 at 60 s, positive in only 5 of 11
    months. Rejected.
* **BTC spot features as a core ingredient.** For BTC, ETH spot was core. For ETH, adding BTC
  spot to the own-market model gives +0.002 IC at spot +1 m. Leaving it out of the full model
  costs −0.0011, p = 0.06. It is kept, because it is cheap and has a clear mechanism, but it is
  not a driver.
* **"Large trade" threshold.** The stored ETH perp seconds were first built with ≥ 1 ETH
  (~$3k) as a "large" trade, inherited from BTC's 1 BTC (~$100k). That is not like-for-like.
  The data were rebuilt at **25 ETH**: large prints are 55% of ETH perp volume vs 60% for BTC
  at 1 BTC, so the threshold now has the same meaning.

## 3. Feature families: leave-one-family-out on DEV months (spot +1 m)

The reference is everything (`sat_all`, IC 0.0568). Removing one family gives the following.
The paired day-block bootstrap is over 92 days.

| family removed | ΔIC | 95% CI | months worse |
|---|---|---|---|
| Coinbase premium | −0.0027 | [−0.0042, −0.0012] | 3/3 |
| BTC perp | −0.0011 | [−0.0024, +0.0002] | 2/3 |
| BTC spot | −0.0011 | [−0.0026, +0.0003] | 2/3 |
| 5-coin swarm | −0.0004 | [−0.0018, +0.0010] | 2/3 |
| perp premium index | −0.0002 | [−0.0012, +0.0009] | 2/3 |
| 17-coin swarm | −0.0001 | [−0.0011, +0.0008] | 1/3 |
| ETH/USDC book | **+0.0004** | [−0.0009, +0.0017] | 1/3 |

Decision (simplest set not worse than everything):
**satellite = core + Coinbase + BTC perp + 5-coin swarm.**
* It equals `sat_all` at spot +1 m (ΔIC +0.0002) and perp +1 m (+0.0003).
* It beats `sat_all` at spot +10 s (+0.0011, CI excludes 0, 3/3 months).
* It drops the ETH/USDC book and the 17-coin swarm. That is 13 fewer live requests per minute,
  and fewer venues whose structure can break (the FDUSD lesson from BTC).
* Removing *both* swarms hurts (−0.0011, p = 0.06; perp +1 m −0.0017, p = 0.04). The 5-coin
  swarm carries the market factor that the 17-coin swarm only duplicates.

The satellite beats the core on every target tested (DEV, all p < 0.001, 3/3 months): spot +1 m
+0.0042, perp +1 m +0.0052, spot +10 s +0.0054, perp +10 s +0.0059.

<!-- sections 3b+ are completed after the confirmation months and holdout -->
