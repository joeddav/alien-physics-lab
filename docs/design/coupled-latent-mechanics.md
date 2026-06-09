# Alien Physics Lab → Coupled-Latent Tech-Tree (CLTT): Design Doc

**Status:** proposal for owner review · **Date:** 2026-06-09 · **Gating flag:** `--tech-tree` (default OFF = byte-identical to today)

## 1. Vision

Today the lab is, by its own CLAUDE.md admission, "effectively one prompt": one latent (`gravity_m_s2`) varies, and the only real skill is *aggregating noisy readings*. We expand it into a **distribution over hidden causal structures** where the skill becomes the *discovery procedure itself*: pick the right instruments out of a large confounded catalog, sequence experiments to identify a latent, and chain inferences (infer A and B to get C). Crucially we do this **without inventing new physics** — the repo already simulates the coupling we need, it's just short-circuited.

The named architecture: **CLTT = Coupled-Latent Tech-Tree with a Confounded Instrument Catalog.**

- **Spine:** a per-seed latent **DAG**. Leaves (`g`, `R`, `latitude`) need one experiment; tier-1 (`spin` = latitude-then-Ω) needs two; tier-2 derived nodes (`mass`, `density`, `orbit_period`) compose leaves. Tasks scale as DAG nodes, not hand-written one-offs.
- **The live coupling** is real, not contrived: `world.py:30-39` already defines `effective_gravity_m_s2 = G·M/R² − Ω²·R·cos²(lat)`. Today every world bypasses it by setting `gravity_m_s2` explicitly (`env.py:61`). Setting it to `None` and drawing the roots makes `g` a genuine deterministic function of four latents — verified live in `world.py`.
- **Confounded catalog:** ~12-14 instruments where most are *observability-degenerate* for the asked node. "Which experiment matters" becomes a well-posed identifiability question, graded **for free** by the existing relative-error verifier.
- **Semi-OOD eval:** a named DBCA combination-split suite (S1-S6) over disjoint seeds, with an atom-coverage manifest, so generalization can't be silently contaminated into memorization.

We **reject** the multi-box "battery" alternative (PCB): it would rewrite the verifier into multi-target partial credit, add a spray-many-`\boxed{}` reward-hack surface, and abandon the one-number/one-relerr robustness that is the project's moat. Chained single-answer targets (mass) recover the same in-episode measurement reuse with none of that risk.

## 2. Design-space map

Four axes; four candidate architectures (CLPT / CIC / PCB / Minimal-Extension). They differ in **ambition/risk, not direction** — they agree on every load-bearing mechanic.

| Axis | Recommendation | Why |
|---|---|---|
| **Episode shape** | **One boxed number/episode** (reuse `parse_boxed_value` grpo_env.py:131 + `score_value` env.py:217). Reject PCB battery. | Battery multiplies tasks fastest but rewrites the verifier and opens a multi-box spray hack. Chaining recovers reuse without it. |
| **How "which tool matters" is induced** | **Structural** (wrong tool → wrong number → score 0) **+ a small per-experiment cost**. NEVER reward tool choice. | Unanimous across candidates and the active-sensing literature (Act-Then-Measure; process-reward over-optimization). The current count-based `measurement_reward` is the anti-pattern. |
| **Interrelation mechanism** | **Activate the dormant `effective_gravity` coupling** (`gravity_m_s2=None`). | Highest-confidence move; zero new physics. Real "infer A before B" chains fall out of the existing formula. |
| **OOD split granularity** | **Combination-disjoint** (DBCA / gSCAN style), not just seed-disjoint. | Current `make_splits` (grpo_data.py:176) is instance-split only; a memorized single procedure passes it. |

## 3. Recommended architecture & mechanics

Everything below is **CRN-safe** (each per-world value a deterministic `f(seed)` with its own decorrelated prime-offset RNG, extending the `_TOL_RNG`/`_TOOLS_RNG`/`_DIAM_RNG` pattern at grpo_data.py:59-63) and **gated behind `--tech-tree`** (OFF = byte-identical to today).

### 3.1 Latent graph (`world.py`)
- On `--tech-tree` derived worlds, set `gravity_m_s2=None` so `effective_gravity_m_s2` is live.
- **World generation — FIXED per adversarial finding #1 (see §5).** Do **not** draw `M` and `R` independently. Draw `g` directly log-uniform in the trained `[0.4, 2.2]×` Earth band, draw `R` log-uniform `[1.5e5, 6.371e6]`, set `M = g·R²/G`, then **rejection-sample on induced density** (keep only worlds with `ρ = 3g/(4πGR) ∈ [1500, 15000] kg/m³`). The rejection loop is seeded per-world (CRN-safe). Verified: this pins `g` exactly in-band by construction and yields physically plausible densities (p5 ≈ 3819, median ≈ 9591, p95 ≈ 14390 kg/m³) at a 22% accept rate.
- Add read-only derived properties: `world_density_kg_m3 = M/((4/3)πR³)`, `low_orbit_period_s = 2π·sqrt(R/g)`, `base_gravity_m_s2 = G·M/R²`. Add causally-disconnected nuisance latents `temp_k`, `surface_pressure_kpa` for distractor tools.
- **DAG edges:** `{M,R}→base_g`; `{Ω,lat,R}→centrifugal`; both → `effective_g`. `mass = g·R²/G` needs `g`(drop/pendulum) + `R`(horizon_dip). `density` needs `M`+`R` (3-hop). `orbit` reuses `R`+`g`. **`R` is the shared prerequisite** across diameter/mass/density/orbit — the compositional skill-reuse the design wants.

### 3.2 Tool catalog (~12-14 instruments)
**All declared as methods on `AlienPhysicsGRPOEnv`** because TRL freezes the tool schema at trainer construction; per-world relevance is via the existing **soft-disable** path (grpo_env.py:295-310), NOT schema change. Per-world advertised subset is a deterministic `f(seed)` extension of `TOOL_SUBSETS`.

- **Reachable today:** `drop_ball`, `pendulum_period`, `measure_horizon_dip`, `calculator`.
- **Wire-up (physics already in env.py, methods absent — #1 free win):** `foucault_pendulum` (returns Ω·sin(lat) deg/hr, env.py:191), `star_tracker` (returns lat, env.py:202).
- **New, three confound TYPES** (distractors must be semantically *adjacent* physics instruments so topic-filtering can't trivially reject them):
  - **IRRELEVANT:** `thermometer()→temp_k`, `barometer()→pressure_kpa` (read nuisance latents).
  - **INSUFFICIENT:** `laser_rangefinder(target)→distance_m` (length, no timing); `star_tracker` gives lat but not Ω.
  - **CIRCULAR/DEGENERATE:** `scale` — **redesigned per finding #3 (see §5).**
  - **Joint-read:** `orbit_timer(altitude_m)→period_s` (reads R,g jointly).
  - **EVAL-ONLY (S2):** `incline_timer(angle_deg,length_m)`, `spring_oscillator(stiffness)`.
- All readings route through `_measure` (env.py:274) so noise/CRN are uniform. Every tool docstring is strictly target-neutral and relation-only (extends the no-leak discipline of `SCENARIO_INTROS` grpo_env.py:51).

### 3.3 Target & verifier
- `true_target_value` (env.py:209) gets branches for `latitude`, `spin` (exists), `mass`, `density`, `base_gravity`, `orbit_period`. `score_value` (env.py:217) stays target-agnostic and **untouched** — relative error is dimensionless, so reward scale auto-balances across m/s² vs m vs kg.
- **No learned judge** (we explicitly reject NewtonBench's LLM-judge equivalence check). Optional deterministic discrete regime-tag gate is **deferred** (open question 2).

### 3.4 Reward (`grpo_env.py`)
- Keep `physics_reward` (accuracy, dominant) + `validity_reward`, **but harden validity per finding #9:** make it a **multiplicative gate** on `physics_reward` (or cap it well below the minimum non-zero physics increment), so when a whole group scores ~0 the policy can't optimize the shaping term instead of accuracy.
- **Replace `measurement_reward` with `efficiency_reward`.** Note: today's `measurement_reward` is already *saturating* (CAP·(1−DECAY^(n−1)), grpo_env.py:374), not purely linear — but it is **accuracy-blind**, which is the real defect. **FIXED per finding #7:** do NOT use a global absolute per-experiment cost (it under-aggregates on tight/noisy `vary_precision` worlds — a count-ANTI-hack mirroring the documented n≈2 collapse). Instead make cost **lexicographic / tie-breaking**: accuracy first, then prefer fewer calls **only among answers in the same success bucket**, or cost only experiments *beyond* what the world's noise/tolerance justifies. Default `COST=0.0` (byte-identical); introduce only from Phase 2, swept small.

### 3.5 Episode & guards
- Episode loop **unchanged** (ReAct; terminate on no-tool-call + `\boxed{number}`). One number per episode; chains box one final derived value.
- **Observability + budget guard** (dataset-build time): assert the advertised tool subset is **identifying** for the target (mass-world MUST advertise an R-tool AND a g-tool) and **budget-feasible** (`chain_depth × aggregation_count ≤ 12`), AND — per finding in verifier-gaps — that **tight-tolerance worlds are winnable** (sqrt(n) error reduction reaches `tol` within budget; verify numerically per world). Guard against the `effective_gravity` 0.001 clamp (env.py:39) producing degenerate `orbit`/`density` targets.

## 4. Semi-OOD split scheme

A NEW `scripts/eval_cltt.py` replaces the **fully broken** `eval_grpo.py` (it still calls `submit_answer`/`_EpisodeComplete`/`env.last_result`/`env.submitted` at eval_grpo.py:57,85-93,113-117 — none exist post-boxed-migration — and hard-codes a `9.80665` earth-prior detector at eval_grpo.py:124 that is meaningless once g is 0.4-2.2× Earth). It drives the boxed ReAct loop and scores via `env._lab.score_value(parse_boxed_value(text))`. All eval seeds disjoint (`EVAL_SEED_START=1_000_000`).

**ATOMS** = {each tool, each target latent, each inversion step: square-then-invert, deg→rad, divide-by-sin, `M=gR²/G`}. **Manifest assertion in `make_splits`:** every atom appears in ≥1 TRAIN world; only the COMBINATION/chain is held out. A train seed drawing any held-out (target,tool) pair is a hard assertion failure (locked by a unit test).

| Split | What's held out | Tests |
|---|---|---|
| **S1** unseen-seed | seeds only (today's split) | Memorization control / baseline. A clean S1 with failing S2/S3 = the memorization signature. |
| **S2** unseen-TOOL / seen-target | `incline_timer` (declared as a train method, NEVER advertised in train) used zero-shot for gravity | **Reframed per finding #6:** report as *formula-substitution*, not composition. Gate on a base-model spot-check; if base ≈ trained, S2 is capability/leakage and is dropped from the profile. |
| **S3** unseen-TARGET / seen-tools | the **mass** node (compose trained `R` via horizon_dip + trained `g` via drop_ball → `M=g·R²/G`) | **HEADLINE compositional test.** Measure ZERO-SHOT mass composition *before* training on it. Separates "learned to invert g" from "learned the discovery procedure." |
| **S4** unseen-RELATION | a tool recombining a trained inversion onto a seen latent | Relation transfer. |
| **S5** unseen-CONFOUND-STRUCTURE | a train-identifying tool becomes degenerate at eval (redundant 2nd timing tool), or an IRRELEVANT tool never paired with the target appears | Confound-rejection generalization. **The centrifugal/equatorial sub-variant is DROPPED — see finding #5.** |
| **S6** longer-chain | train depth-2 (mass), eval depth-3 (density) | Chain-depth generalization. |

**Plus** a DiscoverPhysics "randomized-mode" flag (env auto-picks experiments, agent only analyzes): if randomized-mode reward ≈ agent-chooses reward, the agent is NOT doing active sensing — separates analysis skill from discrimination skill.

**Disclosed atom-overlap caveats** (so the profile isn't oversold): (a) `drop_ball` and `pendulum_period` are degenerate-identical for g (env.py:155/173) — don't count them as two exercised atoms; (b) diameter and mass/density share the SAME R-recovery atom, so if diameter is heavily trained, S3 tests only the `M=g·R²/G` combine step, not R recovery.

## 5. Adversarial fixes folded in (verified against this repo)

Two findings are **FATAL as originally specified** and are now fixed *in Phase 0*, not deferred:

1. **Induced-g miscalibration (FATAL — verified).** Drawing M and R independently log-uniform gives **median g = 39.6× Earth, p95 = 1269×, 37.6% of worlds > 100×, only 11.5% in the trained [0.4,2.2]× band** (200k-sample sim against world.py:36). This reopens the exact prior-guess hack invariant #4 must kill: a policy guessing the log-median (~390 m/s²) lands in the 12% relative-error band on a large fraction of worlds with zero tool calls, and drop times become 0.002-0.2 s (physically bewildering). **Fix:** draw `g` directly in-band, set `M=g·R²/G`, rejection-sample on density ∈ [1500,15000] (§3.1) — verified to pin g in-band with realistic densities. Add the test the design originally mis-specified: assert the **empirical** induced-g distribution over the *actual joint draws* stays in-band at p1/p99.

2. **`parse_boxed_value` truncates scientific notation (FATAL — verified).** `\boxed{5.97 \times 10^{24}}` → **5.97**; `\boxed{597 \times 10^{22}}` → **597.0**; `\boxed{1/2 \cdot 9.8}` → **1.0** (it takes the first number after `\boxed`, grpo_env.py:146, `_NUM_RE` at grpo_env.py:108). Every large/derived target (mass ~1e24) scores rel_err≈1.0 → 0 for the *natural* LaTeX answer form, and the S3 headline eval reads ~0 even when the model composed correctly. **Fix (ship in Phase 0):** rewrite `parse_boxed_value` to strip LaTeX (`\times`,`\cdot`,`\,`,`\text{}`,`^{}`), normalize `10^{24}`→`1e24` and `a × 10^b`→`a e b`, then evaluate the cleaned string through the existing sandboxed `_safe_calculate` AST (env.py:356). Add unit tests with the exact LaTeX forms above. Also log raw boxed text on near-1.0 rel_err to distinguish "wrong physics" from "parse truncation."

Major fixes:

3. **`scale` is a one-call g oracle, not a "provably useless" confound (verified):** `weight/mass == g` exactly for any agent-supplied mass. It cannot be both a valid scored g-path AND a decoy. **Decision required (open question), with a default:** make it a genuine DECOY — return `m·g_ref` for a FIXED reference g unrelated to this world (so `weight/mass ≠ local g`), audited by asserting `weight/mass ≠ effective_gravity`. If instead kept as a valid alternate path, it must be excluded from all S2 "held-out tool" evals and gated by the cost so it doesn't dominate `drop_ball`.

4. **"Which tool matters" is FALSE for gravity (verified):** `drop_ball` and `pendulum_period` both read only `effective_gravity` (env.py:155/173) — interchangeable, so on a gravity world there is no discrimination decision and IRRELEVANT decoys are trivially dimensionally-rejectable. Active sensing is only real on `spin` (lat+Ω) and `mass` (g-tool+R-tool). **Fix:** make selection load-bearing on heavily-trained targets via PARTIALLY-informative/biased tools (e.g. a projectile-range tool needing an extra measured input) or redundant-tool budget waste (S5), rather than relying on irrelevant decoys to train discrimination.

5. **Centrifugal term is sub-noise → base_gravity is a fake task (verified):** at 5× Earth spin, centrifugal maxes at 0.85 m/s² ≈ 0.2% of a 390 m/s² base-g, far under the 3% noise floor. **Fix:** DROP `base_gravity` and the equatorial/polar S5 sub-split. Keep the spin chain only as the **sidereal-day** target (already coded, genuinely needs lat-then-Ω). If a centrifugal effect is ever wanted, raise spin to ~50× (centrifugal then ~8.6× Earth-g) and re-check it doesn't drive equatorial g into the 0.001 clamp. Add a test asserting any "derived" node differs from its parent by ≫ noise across the seed range.

6. **S2 leaks via the stated formula:** if `incline_timer`'s briefing states `g=2L/(t²·sinθ)` it's substitution (the same square-then-invert atom plus a sin factor), not composition; if not stated it's unsolvable. **Fix:** gate the whole suite on a base-model (no-GRPO) spot-check; report S2 as formula-substitution; weight S3/S6 as the true-composition headline.

7. **Global per-experiment COST under-aggregates on tight worlds.** **Fix:** lexicographic/relative cost (§3.4), COST=0 until leaves stable, sweep per worker-distribution, validate with the intervention probe before trusting.

8/9. Easy-target gradient sink → adopt regret-based down-weighting earlier; `validity_reward` shaping-takeover → multiplicative gate (§3.4).

## 6. Invariants preserved (+ byte-identity lock)

1. **VERIFIABLE / no learned judge:** every node computed from `WorldParams`, scored by `score_value` relative-error on ONE boxed number (env.py:217-237). Wrong tool/chain → wrong number → 0.
2. **CRN-safe:** target node, advertised subset, distractor set, chain depth, cost, nuisance latents, and all roots are each `f(seed)` with decorrelated prime-offset RNGs. Soft-disable never perturbs the noise stream (locked by `test_soft_disable_does_not_perturb_noise_stream`, test_diversity.py:111).
3. **UN-CONFUSING vs must-choose:** briefings stay DIAMETER_BRIEFING-simple — list advertised instruments + honest relations + the boxed directive; never name which tool is relevant. Difficulty lives in hidden structure. (This is now *actually true* because finding #1's fix keeps g in a sane, model-priorable range.)
4. **NO trivial dominant strategy:** the fixed induced-g range kills the prior-guess hack (asserted at p1/p99); lexicographic cost kills both "call everything" and "call one degenerate tool"; `scale` redesigned so it can't collapse the g-leg; score the NUMBER not the route so valid alternate paths aren't punished.
5. **One episode = experiment → aggregate → box:** UNCHANGED. Chaining ADDS a "pick + sequence" cliff on top; single boxed answer keeps `parse_boxed_value` the only parse point.

**BYTE-IDENTITY LOCK:** all gated behind `--tech-tree`; OFF → dataset/env/reward byte-identical. `ORIGINAL_INSTRUCTIONS` byte-equality (test_diversity.py:22-49) and off-path equality still pass; COST defaults 0.0. New tests: tech-tree CRN-determinism, observability+budget+winnability guard, **empirical induced-g-range assertion**, **`parse_boxed_value` LaTeX/scientific-notation unit tests**, every-catalog-tool-is-a-method assertion (locks the frozen-schema constraint), and split disjointness/atom-coverage.

## 7. Incremental phase plan (starts from current code)

**Phase 0 — Working boxed eval harness + the two FATAL fixes (ship FIRST, zero training risk, ~1-2 days).**
Rewrite the broken `eval_grpo.py` → `scripts/eval_cltt.py` for the boxed protocol with the S1 split; redefine the prior-hack detector relative to the per-world induced-g distribution (not Earth g). **Plus, brought forward from later phases:** fix `parse_boxed_value` (LaTeX/scientific-notation, via `_safe_calculate`) and add its unit tests; fix the induced-g world generator and add the empirical p1/p99 in-band assertion. Add the base-model S2 spot-check on `incline_timer` (open question 1). Validate the harness against the existing gravity/diameter checkpoints. This gives the first working held-out harness (today all numbers are in-distribution training reward).

**Phase 1 — Wire spin + latitude; mixed-target training (~2-3 days incl. one run).**
Add `foucault_pendulum`/`star_tracker` as env methods (physics exists env.py:191-207); add latitude + spin briefings; lift `target` from a dataset-wide scalar to a per-seed draw (`_TARGET_RNG`). Train over {gravity, diameter, spin, latitude}, balanced mix. Spin is the FIRST genuine 2-tool which-matters task. Monitor per-target n_exp/accuracy/**success-rate** separately for the over-anchor signature. Don't assume the gravity beta/lr recipe transfers (docs show diameter ≠ gravity optimum).

**Phase 2 — Catalog + confound types + lexicographic efficiency cost (~3-4 days).**
Add the ~10 catalog tools (incl. the REDESIGNED `scale`) + nuisance latents. Advertise 1-2 relevant + 1-2 decoys. Replace `measurement_reward` with the lexicographic `efficiency_reward` (introduce COST here, swept ~0.02). Add the observability+budget+winnability guard. **Validate with the intervention probe** (swap a relevant tool for a decoy on a fixed world; assert tool CHOICE and n shift) before trusting the cost — a flat reward curve is NOT evidence of correct diagnosis. Ship S2/S5.

**Phase 3 — Activate coupling: tier-2 derived nodes (~1 week).**
On `--tech-tree`, generate worlds via the fixed g-direct/density-rejection sampler. Add `density`/`orbit_period` properties + branches (NOT `base_gravity` — dropped). Train mass/density/orbit ONLY after leaf skills are individually solved. Ship S3 (headline — measure zero-shot mass BEFORE training on it), S4, S6, and the randomized-mode flag. Per-node beta/lr sweep; calibrate that depth-3 × precision × noise stays inside the 12-turn budget.

**Phase 4 — Curriculum + full generalization profile (~3-4 days).**
Regret-based world selection (gate on GRPO group-internal regret = group-max − group-mean from `reward_dist.jsonl`; oversample mid-regret, down-weight all-0/all-1 worlds) — adopt the down-weighting earlier if the easy-target sink appears. Run the full S1-S6 suite → a per-split generalization PROFILE (not a scalar) + intervention probe + randomized-mode ablation, written up in a per-date `docs/results/` doc.
