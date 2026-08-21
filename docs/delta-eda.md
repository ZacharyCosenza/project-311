# Delta alerting — design notes

Reference for `compute_call_deltas` (`inference/nodes.py`) and the two tweets built on
it (`tweet_delta_summary`, and `tweet_daily`'s district selection). Built from an EDA
over 291 weeks of real call history across 71 districts (2020–2026).

## What delta means here

`delta_tgt_calls` = a district's predicted next-week total minus its own trailing
4-week average of actual calls. Only **increases** count — a predicted decrease gets no
`delta_rank` at all, never a top-5 slot. 4 weeks was chosen from a sensitivity sweep
across K = 1, 2, 4, 8, 12, 26: shorter windows are noisier (single-week volatility),
longer ones start blending across seasons and re-inflate the delta for reasons
unrelated to that specific week. K=4 was the low point of that curve.

Candidate categorization tiers (raw call counts, K=4, artifact excluded — see below):
minor <58, notable 58–124, major 124–212, extreme 212+.

## The outlier filter

One district (`12 BRONX`) is the #1 weekly mover in 58% of all 291 weeks in the raw
data — not real volatility, a recurring data artifact. Every flagged week has the same
signature: `Noise - Residential` alone at 90%+ of that week's total (confirmed directly
against the Socrata API, e.g. 23,282 Noise complaints in one week against <700 for
everything else combined), recurring across 2021, 2022, 2024, and 2025.

A pure magnitude threshold can't safely filter this: a genuine citywide snowstorm week
(multiple boroughs, `Snow or Ice` complaints) shows the *same* single-category-dominance
signature and comparable z-scores. The actual distinguishing signal is **isolation**:
the Bronx artifact weeks have 0–4 other districts also showing an elevated reading the
same week; the snowstorm week has 8–10. `_winsorize_isolated_outliers` caps a board-week
to that board's own historical median only when it's both an extreme outlier (modified
z-score, MAD-based) *and* isolated (fewer than `outlier_min_corroborators` other boards
elevated that week) — `outlier_z_threshold: 8.0`, `outlier_min_corroborators: 5` in
`conf/base/parameters.yml`, tuned against these two specific cases.

This only protects the delta/baseline computation. It does not touch training data —
if `12 BRONX`-style artifacts are affecting model quality (not just the alerting
signal), that's a separate, not-yet-investigated question.

## What's out of scope so far

- Category-level deltas (biggest mover *within* a complaint-type group, not just total
  calls) — the architecture supports it (per-group predictions already exist in
  `models`), not built yet.
- Scheduling `tweet_delta_summary` for real — it's a runnable pipeline
  (`kedro run --pipeline tweet_delta_summary`), not yet wired into
  `deploy/workflows/*.yaml`.
