# Expansion design — "Alien Lab" as an active-sensing / model-selection benchmark

**Status: design exploration, PINNED (2026-06-09). Nothing here is implemented yet.** Captured so
the thread is recoverable. The current trained tasks remain gravity + diameter (see `docs/results/`).

This folder holds the design work for expanding the lab from a toy (one latent — `g` — recovered by
averaging noisy readings) into an expansive, procedural task space with a curriculum.

## The core idea — a "ladder of doubt"

Every task is one skill at increasing depth: **infer the generative model of your world.** Parameter
estimation ("what is g?") is the easy case where the world-model is handed to you; the interesting
tasks strip that away — *is* this a planet? a centrifuge? a moon? a sim with a lying instrument? So
the whole space is **hypothesis discrimination by experiment design** ("which measurement best
separates the worlds I might be in") — i.e. active sensing / optimal experimental design.

The curriculum widens what the agent may question at each tier:

| Tier | Theme | Skill |
|---|---|---|
| T0 First Light | one world, one number | estimate by repetition (today's toy) |
| T1 Two-Instrument Composition | interrelated latents | combine two measurements (mass = g·R²/G) |
| T2 Confound & Degeneracy | identifiability | the obvious reading is biased → find a third route |
| T3 Set-Identification | multi-output | how many moons/rings/suns + each period; 3D position |
| T4 Integrative Missions | cross-domain action | time a launch / ration life-support — success **re-simulated** |
| T5 The World-Model Court | pure model selection | "what kind of place is this?" — prove it with the discriminating measurement |

It stays cleanly **verifiable** the whole way up: every answer is a number (rel-error), a discrete
label (exact match), a set/vector (per-field match), or an **action re-simulated from hidden state**
(orbit achieved? crew survived?). No human/LLM judge.

## The docs

- **[`curriculum.md`](curriculum.md)** — the primary vision: ~79 task families across 8 domains
  (mechanics, celestial, galactic positioning, atmospheric chemistry, geophysics, **epistemic
  curveballs**, integrative missions, exotic worlds), the 6-tier ladder + 7 difficulty axes, the
  capstones, the OOD-split design, and a buildable-now vertical slice.
- **[`data-requirements.md`](data-requirements.md)** — literature-grounded answer (Reasoning Gym,
  RLVE, Only-IF, 1-shot RLVR, LIMR, Procgen, the sharpening-vs-expansion debate) to "how procedural,
  and how much data for gains vs generalization." Citations + confidence flags + verifier corrections.
- **[`coupled-latent-mechanics.md`](coupled-latent-mechanics.md)** — an earlier, narrower pass (the
  "CLTT" coupled-latent tech-tree). Superseded in *scope* by `curriculum.md`, retained for its
  concrete coupled-latent mechanics, the exact OOD-split scheme (S1–S6), and the **two bug findings
  below**.

## ⚠️ Bugs surfaced by the design red-team (fix regardless of the expansion)

1. **`parse_boxed_value` truncates scientific notation** (`grpo_env.py`). `\boxed{5.97 \times 10^{24}}`
   → `5.97`; `\boxed{1.27 \times 10^7}` → `1.27`. It grabs the first number after `\boxed`. This may
   be **silently zeroing correct large-magnitude answers in the current diameter results** (diameters
   ~1e7; any LaTeX-formatted answer truncates), which would mean the ~32–39% measured success is
   *understated*. Mass (~1e24) would always read ~0. Fix: strip LaTeX, normalize `10^{n}`→`1e n`,
   evaluate via the existing `_safe_calculate` AST; add unit tests. (Low effort, no GPU.)
2. **Independent (M, R) sampling miscalibrates g** — *only relevant when the coupled-`effective_gravity`
   mode is activated.* Drawing mass and radius independently log-uniform gives median g ≈ 40× Earth,
   reopening the prior-guess hack. Fix: draw `g` directly in-band, set `M = g·R²/G`, rejection-sample
   on plausible density. (Not a current-code bug; a fix to bake in before Tier-1+.)

## Key open decisions (when we unpin)

- **Reward shaping above T1.** The count-based `measurement_reward` *mis*-incentivizes model-selection
  (rewards call volume; the skill is the *right few* calls — and for the sim/quantization curveball,
  *more* averaging destroys the evidence). Replace with a task-kind-aware "ran the discriminating
  experiment" bonus, or drop shaping and rely on outcome reward + KL.
- **One curriculum-scheduled policy vs per-tier checkpoints.** (Mixed-target groundwork already exists:
  per-rollout `target`/`world_diameter` logging.)
- **The ephemeris propagator** — the single biggest engine cost; gates the orbital/launch missions.
  Keep those closed-form/north-star, or invest in a real propagator.
- **The 4-axis held-out split** (the highest-leverage methodology item): seeds *(weak, exists)* +
  **difficulty + tools + whole families**. A seed-only split does NOT detect generator-memorization —
  the failure already observed (the β=0.01 n=1 freeze; the "effectively one prompt" pathology).

## Data-requirements punchline

"80 problems" is the wrong unit — every reference system is a *generator library* (Reasoning Gym 100+,
RLVE 400, InternBootcamp 1000+). **Gains** (in-dist reward up) need ~1 prompt / a few hundred steps and
prove little (mostly sharpening — the current runs are here). **Generalization** (the real goal) needs
**diversity, not count**: ~10–30 families × 2–10k seeds each clears the ~300–1000-distinct-*structures*
band, joint-mixture ~800–2000 steps with easy worlds faded. Lean into this env's structural advantage:
**post-pretraining-cutoff hidden latents make it contamination-proof.** Run three controls before
trusting any gain: pass@k (not just pass@1), a non-Qwen base, and scrambled physics constants.

## Buildable-now vertical slice (~3–4 days, current engine)

1. `mass_of_a_world` (T1) — reuse the proven `target='diameter'` pattern; trap: reporting g *or* R
   alone scores ~0. Proves composition trains.
2. A tiny answer-form layer (`parse_boxed_label` + `score_label`, exact-match) — unlocks the whole
   classification / model-selection track.
3. `centrifuge_or_planet` — first curveball: planet g flat with height, centrifuge g falls with height
   (+ Coriolis), tuned identical at the bench so a single reading is degenerate. 50/50 prior, with a
   transcript-checked "sampled ≥2 heights" evidence bonus.
