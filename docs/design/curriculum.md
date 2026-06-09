# The Alien Physics Lab — Curriculum Design Doc

> *"You wake up in a lab on an alien planet, surrounded by instruments. Use them to figure out the world."*

Today the lab asks one question — **what is g?** — and answers it by averaging noisy drop/pendulum readings. That single loop (call a noisy instrument → repeat → aggregate → `\boxed{number}`, scored by `score_value` rel-error against hidden `WorldParams`) is already proven to train under multi-turn GRPO (gravity and the new diameter/horizon-dip task both reach physics ~0.5–0.7, n→6–12, stable). This doc turns that one loop into a **world**: dozens of crisply-specified instruments across six domains, a six-tier ladder that composes skills bottom-up, and — the part we are most excited about — a **"what kind of place is this?" epistemic thread** that runs through every tier and culminates in the agent putting the setting itself on trial.

The whole design holds four invariants sacred, because they are what make this RLVR and not a vibe check:

- **Verifiable, no judge.** Every answer is a number (rel-error), a discrete label (exact match), a set/vector (bipartite/per-field match), or an **action re-simulated from hidden state** (orbit achieved? crew survived?). Ground truth is always a deterministic function of `WorldParams`.
- **Un-confusing but hard.** Each instrument returns a clean noisy reading of one definite quantity. Difficulty comes from hidden structure, confounds, inference depth, and epistemic traps — *never* from ambiguous instructions.
- **Active sensing.** The catalog deliberately offers irrelevant / confounded / degenerate / seductive-but-wrong instruments. Choosing the right experiment (or designing the right sequence) is the skill.
- **CRN-safe.** Every per-world latent is a deterministic `f(seed)` drawn from its own decorrelated RNG (the existing `_DIAM_RNG = (86028121, 11)` pattern), so all rollouts in a GRPO group share the same hidden world and the same frozen noise stream. Noise is part of the *task*, not per-rollout luck.

---

## The vision in one picture

```
        DOMAINS (what you measure) ─────────────────────────────────────────
        Gravimetry · Celestial · Galactic positioning · Chemistry · Geophysics
                              │
        TIERS (how hard) ─────┼──────────────────────────────────────────────
   T0 First Light → T1 Two-Instrument → T2 Confound/Degeneracy →
   T3 Set-ID & Campaigns → T4 Applied Action Missions → T5 The World-Model Court
                              │
   CURVEBALL TRACK (vertical) ┴─ "question the setting" injected at EVERY tier,
        escalating from "the obvious reading misleads" → "your instrument lies"
        → "this isn't a planet" → "this isn't even a world."
```

Skills compose strictly bottom-up; each tier's graduation skill becomes a **reused subroutine** in the next. The clearest example: `measure_horizon_dip` → radius `R`, minted as a building block in T1, is then *consumed* by mass, density, core-radius-from-shadow, heat-power surface area, orbital velocity, and landing geometry across four later tiers. That reuse spine is why the ordering is opinionated and not negotiable.

---

## The difficulty axes

Seven independent dials. A task's tier is roughly *how many it turns up at once*.

| Axis | Low | High |
|---|---|---|
| **Inference depth** | One latent, one instrument, average repeats (`first_light_gravity`). | 3–5 chained latents where order matters and intermediates are consumed downstream (`spin_correction_true_g`: Foucault→lat→R→add centrifugal; `full_interior_profile`: R feeds *both* core-radius and heat-power). |
| **Identifiability** | All instruments relevant; the only skill is pooling two same-latent tools. | A catalog where most tools are irrelevant/degenerate/seductive-but-wrong and the load-bearing pair is non-obvious (`cross_reactive_reagent`: A − B isolates SO₂, but reagent C *looks* selective and adds a 3rd unknown; `quasar_triangulation`: quasar-to-quasar angles are position-invariant at infinity — useless). |
| **Noise & earned precision** | noise 0.005, 3% tol, 2–3 repeats suffice. | noise 0.08+, tight per-world tol, signal buried below surface g (`tidal_two_body_pull`, `cmb_dipole`) — needs a differential instrument, long baseline, *and* ~8–12 repeats. |
| **OOD novelty** | Held-out **seed** (existing `EVAL_SEED_START=1_000_000`). | Held-out **task / world-kind / curveball**: an instrument, confound mechanism, or entire hypothesis class never trained on. |
| **Epistemic / model selection** | World model fixed and trusted; estimate a parameter. | Select among generative hypotheses about what kind of place this is; the most-natural reading is engineered degenerate, so you must design the discriminating experiment and confirm with a second orthogonal probe. |
| **Cross-domain integration** | Single instrument family. | One episode fusing mechanics + astronomy + chemistry + a committed action (`life_support_ration`: O₂ assay × Hohmann ETA → survival). |
| **Open-endedness / action commitment** | Report one number, graded smoothly. | Commit an irreversible action re-simulated from hidden state, with narrow/two-sided windows, sometimes **no valid answer** (`ballistic_range_table`: declare *unreachable*; aerocapture: too-shallow AND too-steep both fail). |

---

## The tier ladder

Each tier lists its theme, what it teaches, a sampler of curated tasks (by domain), and a **graduation criterion** that is itself programmatically checkable. Graduation always requires clearing **all three OOD levels** (seed / sibling-task / world-kind), not just held-out seeds.

### T0 — First Light *(intro, single latent)*
**Theme:** One world, one number, earned by repetition. The byte-identical-briefing toy, generalized to interchangeable single-latent estimates.
**Teaches:** the atomic loop — noisy read → repeat → aggregate → box — and the proven `measurement_reward` cliff that makes n≥2 worth it. Recognize when two instruments measure the *same* latent so you can pool independent estimates.
**Tasks:** `first_light_gravity` (drop + pendulum both give g) · `pole_star_latitude` (the star-trail *pivot* = latitude; instantaneous altitudes drift = soft confound) · `partial_pressure_o2` (barometer + O₂-scrubber Δ) · `seismic_moho_depth` (shoot a *spread* of offsets, find the crossover; one near-offset shot is degenerate with a halfspace) · `count_the_suns` (sample sky_camera across a full day-cycle).
**Graduation:** on held-out seeds, accuracy ≥ the proven gravity/diameter bar (physics ~0.6, high success-rate) **AND** median n_experiments ≥ 3 (demonstrably aggregating) **AND** correctly pools two same-latent instruments when both are offered. Across all `{number}` tasks in the tier.

### T1 — Two-Instrument Composition *(core, interrelated latents)*
**Theme:** No single instrument suffices; the answer emerges only from combining measurements of *different* latents via a closed-form chain.
**Teaches:** compose across instruments/domains (M = g·R²/G needs a gravity tool **and** horizon-dip); spot algebraic shortcuts that skip an intermediate (ρ = 3g/(4πGR) skips computing M); detect and avoid **zero-information experiments** (incline too shallow → "no motion"). Introduces the **classification** answer-form and the canonical **R subroutine**.
**Tasks:** `mass_of_a_world` (g + R → M; either alone scores ~0) · `density_decides_iron_or_ice` (g + R → ρ → ice/rock/iron) · `altitude_g_gradient_radius` (choose a *large* altitude baseline) · `incline_friction_and_g` (two angles, or drop_ball bypasses friction) · `core_shadow_zone` (shadow half-angle + R) · `year_length` / `axial_tilt` (sample the slow annual oscillation; ignore constant latitude).
**Graduation:** held-out seed success on ≥2-latent chains where reporting *either* single latent scores ~0 (proves composition); produces the first exact-match classifications; does **not** fire a zero-information experiment twice in a row.

### T2 — Confound & Degeneracy Resolution *(advanced, identifiability)*
**Theme:** The obvious reading is biased, degenerate, or shared. Earn an un-confounded answer by adding a third route, switching instruments, or sampling a discriminating dimension. *This is where the catalog's richest single-domain gems live.*
**Teaches:** identifiability — two-equations-two-unknowns, underdetermination; **more data on the wrong instrument cannot fix a bias or a spectral alias** (variance vs bias); calibrate out a multiplicative bias with a known standard. **This is the prerequisite for everything epistemic: you cannot question the *setting* until you can question an *instrument*.**
**Tasks:** `spin_correction_true_g` (bench g biased low by spin) · `spectral_alias_co2_vs_n2o` (low-res band is permanent BIAS — switch to high-res or the orthogonal 15 µm CO₂-only band) · `cross_reactive_reagent` (A − B isolates SO₂; reagent C is a trap) · `isotope_ratio` (bracket with a KNOWN standard to back out mass-bias; ion-source-current is a no-op knob) · `dynamo_field_strength` (B degenerate between strength & maglat; read **inclination** first) · `galactic_coords_from_quasars` (pick a well-*separated* triple; collinear is rank-deficient) · `ballistic_landing_with_drag` (drag identifiable only by **varying mass**; the wind_sock is zero-mean-noise decoy).
**Graduation:** held-out seed **and** held-out confound-instance where the naive single-instrument estimate is provably outside tolerance; adds the third/orthogonal route when two channels disagree beyond noise; stops over-averaging a degenerate instrument and switches; handles a multi-number answer (offset + corrected composition) with **both** fields in band.

### T3 — Set-Identification & Structured Inference *(advanced→capstone, multi-output)*
**Theme:** The answer is a **set or vector whose cardinality you must also infer**: how many moons/rings/suns, each one's period, a 3D position, a full interior profile.
**Teaches:** cardinality + per-element estimation under a bipartite-match scorer; active design of observing **baseline and cadence** (catch each periodic body ≥twice to fix a period); choose geometrically non-degenerate references; statistical model selection over a point process; compose a whole **campaign** that reuses a shared scale.
**Tasks:** `how_many_moons_and_periods` (long dense baseline, period-fold, disentangle overlapping dip-combs; a 2:1 resonance hides a moon → alternating dip depths) · `ring_shadow_count` (sweep photometer azimuth; sub-beam ringlets blur, sub-threshold ringlet must be *excluded*) · `pulsar_timing_position_fix` (3 pulsars with independent sky directions; coplanar triad is ill-conditioned) · `plate_activity_classification` (joint depth+belt+b-value distribution; one big shallow quake is NOT activity) · `full_interior_profile` (core radius + state + Moho + heat + ocean from a sequenced campaign reusing R).
**Graduation:** bipartite-match success requiring **both** correct cardinality and every element in tolerance; demonstrates baseline/cadence design; rejects degenerate reference geometry; on the composite capstone hits every per-field tolerance with a sensible instrument **order** (shared quantities measured once, decoys skipped).

### T4 — Integrative Applied Missions *(capstone, cross-domain action)*
**Theme:** Goal-directed missions whose **success is re-simulated from hidden state**: time a launch, schedule an uplink, land a probe, ration life support, predict an eclipse to the minute.
**Teaches:** chain a full estimation pipeline into a committed action with end-to-end simulated verification and **unforgiving / two-sided / sometimes-unsolvable** windows. Learn that maximizing one quantity can FAIL (over-braking burns up in aerocapture) and that the right answer is sometimes *unreachable / abort*.
**Tasks:** `launch_to_circular_orbit` (g + R → v_circ; star_tracker & Foucault are red herrings) · `ballistic_range_table` (measure g first; detect unreachable + two-angle ambiguity) · `moon_transfer_window_burn` / `eclipse_geometry_predict_next` (Kepler from µ, phasing; node-regression drift; the angularly-*larger* not brighter moon causes totality) · `life_support_ration_schedule` (O₂ mole-fraction × transfer ETA → survival; total pressure is the trap) · `comms_uplink_light_delay` (direct radar over the angular-size shortcut; account for motion between measure and act).
**Graduation:** held-out seed **and** held-out mission: simulated-outcome success at a target rate, **including the unsolvable/abort instances answered correctly**; measures all load-bearing latents before committing; ignores named decoys; respects two-sided corridors (does not over-maximize); carries the right-target/right-body/right-sense (retrograde, totality) selection through to the action.

### T5 — The World-Model Court *(capstone, pure model selection)*
**Theme:** "What KIND of place is this?" No labels, full catalog, decoys seeded — rule out **planet / moon / centrifuge / dome / simulation** (and honest-vs-miscalibrated instruments) and **prove** the verdict with the one discriminating measurement, confirmed by a second.
**Teaches:** everything above, inverted into hypothesis space. The most-natural reading is engineered degenerate (single-spot gravimeter reads Earth-normal in *both* planet and centrifuge). Run a **decision tree** of experiments chosen to maximally reduce hypothesis entropy, distrust internal consistency, and — for the simulation curveball — **interrogate the noise structure rather than average it away** (the inverted aggregation lesson).
**Tasks:** `full_world_model_court` (1-of-5 with seeded decoys; confirm with a second orthogonal probe) · `centrifuge_or_planet_gravity_gradient` / `coriolis_sidewall_test` (g-vs-height gradient; lateral Coriolis scaling h^1.5) · `simulation_floating_point_seam` / `atmosphere_is_simulated` (quantization grid, PRNG period, conservation closure; ignore the self-report flag) · `miscalibrated_meter_unit_trap` (out-of-band invariant breaks it; agreement is the trap) · `are_you_under_a_real_sky` / `skyless_dome` (zero-parallax-at-infinity test; muon overburden).
**Graduation:** held-out seed **and** held-out world-kind/curveball: exact-match verdict at a high rate, with **partial credit contingent on the canonical `discriminating_measurement` tool-call pattern appearing in the transcript before the answer** (the right verdict via the right evidence, not a lucky guess); never stops at the first suggestive reading on a decoyed instance; on the simulation curveball over-samples-for-structure instead of averaging.

---

## The curveball / epistemic track *(the spine the owner cares most about)*

The "question the setting" skill is **not confined to T5.** It is a vertical thread injected at every tier as a small, escalating fraction of episodes, so the policy learns to distrust its prior *continuously* rather than meeting model-selection as a separate genre only at graduation.

| Tier | Curveball flavor | Example |
|---|---|---|
| **T0** | *Soft* — the most-natural reading misleads, but the world model is honest (still a parameter estimate). | `pole_star_latitude`: instantaneous altitudes drift; only the trail *pivot* is invariant. `count_the_suns`: one frame undercounts; one "sun" is a reflecting moon. |
| **T1** | Breaks a **textbook prior** you must measure instead of assume. | `parallax_distance_to_beacon`: 1 arcsec ≠ 1 pc here because the orbital baseline isn't 2 AU — **measure it.** Year length is non-integer; resist rounding. |
| **T2** | First true **binary model-selection embedded as a confound**. | `is_there_a_dynamo_or_remanent`: same field, two generative models → need spatial dipole-law + secular-variation. `instrument_liar_crosscheck`: a third route breaks a calibration tie, and **"none faulty" is a valid answer.** |
| **T3** | Model selection over **structure/statistics**. | `plate_activity_classification` over a point process; resonance / sub-threshold cases where "more sensitivity ≠ more objects." |
| **T4** | A **regime-confirmation gate** inside an action mission. | `centrifuge_or_planet_launch_abort`: classify THEN launch with the regime-correct model, abort if unsafe. `tidally_locked_terminator_landing`: the clock never reaches "morning." |
| **T5** | The full multi-hypothesis court — model selection IS the whole task. | `full_world_model_court`. |

**The anti-collapse rule (a hard graduation gate at EVERY tier):** ~15–20% of episodes in T1–T5 are **honest control worlds** where the curveball is *absent* (k=1, real sky, planet, "nothing is wrong"). The policy must learn the curveball is a **hypothesis to test, not a reflex to always cry "centrifuge!/simulation!"** The **false-alarm rate on honest control worlds is a tracked metric and a graduation sub-criterion.** Without the control worlds this collapses to a degenerate alarm strategy; with them it is healthy.

---

## Capstone missions (the north stars)

1. **`full_world_model_court`** — the apex of T5 and of the whole curriculum. No labels, full catalog, hidden `world_kind` ∈ {planet, moon, centrifuge, sealed_dome, simulation} with decoy signatures seeded. Scored exact-match on the verdict **plus** transcript-verified partial credit for emitting the canonical discriminating measurement. Integrates active sensing, identifiability, the curveball track, and a decision-tree experiment plan.
2. **`interplanetary_window_and_aerocapture`** — the apex action mission. Infer home µ (g+R), *both* planets' heliocentric elements (ignoring a decoy third planet), and the target's remote atmospheric scale-height, then commit {departure_epoch, injection_dv, aerocapture_periapsis}, re-simulated end-to-end with a **two-sided failure corridor**. Spans local mechanics + orbital ephemerides + remote sensing in one episode.
3. **`full_interior_profile`** — the apex campaign mission. Deliver core_radius + core_is_liquid + moho_depth + total heat + has_subsurface_ocean from a self-planned, **order-dependent** instrument sequence that reuses horizon-dip R across multiple sub-inversions, budgets repeats for noisy fields, and skips decoys. Composite per-field scoring.
4. **`life_support_ration_schedule`** — the apex cross-domain fusion. A chemistry measurement (O₂ mole fraction, where total pressure is the lethal trap) and an astrodynamics estimate (Hohmann ETA from g+R) fused into ONE resource constraint, verified by simulating depletion against the true budget.
5. **`tidally_locked_terminator_landing`** — fuses regime-confirmation (verify spin==orbit so the thermal map is static), spatial search (thermal_mapper for the survivable band), and mechanics (size the burn from g+drag), with the epistemic trap of a mission clock that never advances to morning.

---

## OOD-eval design (three escalating levels, every tier, all judge-free)

Reuses the existing disjoint-seed contract (`EVAL_SEED_START=1_000_000`) and the batched `eval_grpo.py` harness, extended to held-out tasks/worlds.

- **Level 1 — held-out SEED** *(floor for graduation)*: same task family, hidden latent from the disjoint eval seed range. Because every per-world value is CRN-safe `f(seed)`, eval worlds never overlap training worlds even though they share structure.
- **Level 2 — held-out TASK within the tier's skill** *(tests SKILL transfer, not memorized procedure)*: train on a subset, eval on a sibling exercising the same skill with novel surface form — e.g. T2 trained on `spectral_alias` + `cross_reactive_reagent`, eval on `isotope_ratio` mass-bias (same "a degeneracy needs an independent/known anchor; more data can't fix bias" skill, new instrument); T3 trained on moons-and-periods, eval on ring-count; T1 trained on `mass_of_a_world`, eval on `core_shadow_zone` (same "compose with R" skill). Report the **transfer gap** (sibling minus seed-held-out accuracy — small gap = real skill).
- **Level 3 — held-out WORLD-KIND / CURVEBALL** *(the epistemic OOD)*: hold out an entire hypothesis class — train model-selection on {centrifuge-vs-planet, real-vs-simulation}, eval on {dome-vs-underground, dynamo-vs-remanent, miscalibrated-meter}; or hold out a confound *mechanism* (trained on multiplicative-timer-bias, eval on wavelength-axis-offset). At T4/T5, also hold out the **unsolvable/abort instances** (unreachable target, unsafe-launch abort, all-honest control) to verify the policy can say "impossible/abort/nothing is wrong" rather than always producing a confident verdict.

Each tier reports: L1 accuracy/success-rate · L2 transfer gap · L3 curveball verdict accuracy + discriminating-measurement-correct rate + **false-alarm rate on honest controls**. Advancement requires clearing all three.

---

## Buildable-now vs aspirational — an honest map

### Ships almost verbatim on today's engine (numeric, `score_value`, boxed single number)
The **gravimetry numeric chain is the strongest cluster** and follows the *exact* proven `target='diameter'` + `_DIAM_RNG` extension pattern — add a `target=` branch to `true_target_value()` + a per-world log-uniform latent draw in `grpo_data`. No new answer form, no new noise model:
- `mass_of_a_world`, `density_*` (numeric ρ backstop), `altitude_g_gradient_radius`, `core_shadow_zone` — all reuse the existing `measure_horizon_dip` + a closed-form g.
- `spin_correction_true_g`, `oblate_high_spin` — `foucault_pendulum` + `star_tracker` *already exist* and return Ω·sin(lat) and lat; `effective_gravity` already subtracts the centrifugal term. Only real work: make `lab_latitude_deg`/`spin` per-world draws (currently fixed at 35°/earth-spin) + add `target='g_grav'`.
- `incline_friction_and_g` — one new tool `incline_run(angle_deg)`, ~30 lines, genuine T1/T2 confound.
- `seismic_moho_depth` — one tool `seismic_refraction(offset_m)`, piecewise-linear closed form, honest crossover-design active sensing.
- **Astronomy/chemistry/geophysics numeric**: `year_length` & `axial_tilt` (one time-parameterized closed-form `gnomon_noon_altitude(day_index)`); `pole_star_latitude` (trivially `target='latitude'` *today*); `parallax_distance_to_beacon`, `galactocentric_distance`, `cmb_dipole` (new closed-form tools); the chemistry partial-pressure cluster (`partial_pressure_o2`, `spectral_alias`, `cross_reactive_reagent`, `isotope_ratio` — needs a per-world gas-mixture dict, the only structured-latent addition, + ~6 selective-readout tools); `dynamo_field_strength`, `internal_heat_flow` (new closed-form tools reusing R).
- `ballistic_range_table` as a **numeric variant** — score the submitted angle directly vs the analytic θ, sidestepping the action layer while keeping the measure-g-first chain and the unreachable/two-solution epistemic content.

### Needs engine work (the high-leverage additions, roughly in build order)
1. **Answer-form layer** *(highest leverage, ~1–2 days, unlocks ~60% of the catalog and gates the entire model-selection track)*: `parse_boxed_label` + `score_label` (exact-match for classification, integer-count, and sentinels) · structured-JSON parse + `score_set` (tolerance-gated bipartite match, cardinality AND values) + `score_vector` (per-field rel-error mean) · a per-task `answer_kind` router so the reward fns pick the right scorer. Map integer-count tasks onto the *same* label scorer (exact-int) to avoid the rel-error-forces-integer hack.
2. **Position/altitude-resolved gravity model** *(~1–2 days)*: add `world_kind` + g(h, position) — planet g₀(R/(R+h))², centrifuge Ω²(R_c−h) — plus a Coriolis lateral-deflection output. Makes `drop_ball` / a new `gravimeter` consume height. Unlocks the signature centrifuge/planet/coriolis/oblate cluster. (Note: `_drop_ball` currently *ignores* `height_m` for g; `effective_gravity` has no h term — this is the gap that makes the canonical curveball not yet exist.)
3. **Time axis + lightweight periodic evaluator** *(cheap for closed-form; medium-heavy for multi-body)*: a mission-clock arg + per-call deterministic time-evolved readings. Cheap tier = single sinusoid/box closed forms (gnomon, single-moon transit, sun-count, tidal-breathing). Expensive tier = a small ephemeris propagator for superposed periodic bodies (overlapping transit combs, binary beats, node regression).
4. **Structured latents** in `WorldParams` *(light-medium)*: small immutable structured fields drawn CRN-safe — gas dict (trivial); variable-cardinality catalogs (moon/ring/pulsar lists with a per-world count draw + guard-band rejection so elements stay separable).
5. **Action re-simulation hook** *(light for closed-form, heavy for orbital)*: `score_action(answer)` runs a deterministic post-sim from hidden state → success + margin. Build the cheap ones first (`launch_to_circular_orbit`, `ballistic`, `comms_uplink`) to validate the action-reward plumbing; defer multi-body missions.
6. **Miscalibration / out-of-band-invariant model** *(~1 day)*: per-world multiplicative instrument bias + a calibration-shot/known-standard/atomic-invariant tool. Unlocks the `miscalibrated_*` family.
7. **`measurement_reward` generalization** *(~½ day, but important)*: today it hard-codes n = #drop_ball + #pendulum and rewards **count**. On identifiability/model-selection tasks the skill is choosing the *right* instrument (often few calls), and on simulation/quantization tasks **more averaging is actively wrong**. Make the shaping reward task-kind-aware: count any non-calculator experiment, and **gate/disable count-shaping for model-selection and curveball tiers** (replace with a "ran ≥1 discriminating experiment" transcript bonus).

### Aspirational (tagged `needs_bigger_model`; sequence last)
`interplanetary_window_and_aerocapture` (real multi-body astrodynamics + remote sensing + two-sided corridor) · the `quasar_triangulation` / `which_galaxy_am_i_in` family (faithful 3D sky model with rotation/orientation solve) · `full_world_model_court` (requires *every* discriminator sub-model to exist simultaneously + the evidence-key scorer) · the **simulation/quantization curveballs** (invert the engine's whole clean-Gaussian-noise philosophy — a deliberately-non-physical alternate world mode + a scorer that verifies the discriminating regime was probed) · to-the-minute eclipse prediction with node regression. Each is buildable *once its constituent sims exist*; treat them as the engine's growth targets, with cheaper stepping stones (parallax_distance, rotation-curve R, single-moon period-fit, the cheap action missions) shipped first.

---

## Verifiability & collapse risks (and the fixes that keep the invariants intact)

These are the ways a task can secretly become un-winnable, judge-dependent, or hackable. Every fix is mechanical and CRN-safe.

- **`discriminating_measurement` partial credit looks judge-like.** Do **not** parse free-text justification. Score it from the **transcript**: define a canonical discriminating tool-call pattern per world-kind (e.g. centrifuge ⇒ a gravimeter at ≥2 distinct heights spanning a min gap) and award the bonus iff that pattern appears before the answer. Hidden sim state, fully checkable — converts fuzzy intent into a deterministic pattern match.
- **"Declare unreachable / abort / nothing is wrong" has no vocabulary.** Define a fixed sentinel set (`unreachable`, `abort`, `none`, `real_sky`, `honest`) accepted by the label scorer; for unsolvable instances the ground truth *is* that sentinel. Track the success rate on sentinel instances separately — it's the honesty metric.
- **Set-ID bipartite matching is ill-posed when two true elements are within tolerance** (the 2:1 resonance, sub-beam ringlets). When drawing worlds, **reject seeds whose elements are closer than ~2× tol** so the match is unambiguous by construction; score cardinality and matched-value-error as *separate* sub-rewards (smooth gradient).
- **Classification near a threshold** (ice/rock/iron, breathability bands, spectral class) is genuinely ambiguous even with perfect inference. **Reject seeds within a guard band of any class threshold** (reuse the disjoint-seed machinery to resample). Keep the numeric backstop secondary, never primary.
- **Action success bands must be achievable yet non-trivial.** Calibrate each band against a **reference solver** that uses the true latents + the expected measurement error after k feasible repeats — exactly as `vary_precision` already ties tolerance to noise.
- **Bias "within the noise band" makes a task unwinnable.** Draw the bias magnitude clearly *above* the after-k-repeats noise floor, so it's detectable with diligent aggregation. The curveball lesson ("repetition can't fix bias") survives because the bias doesn't *shrink* with averaging — but it must be resolvable in principle.
- **2-class guessing collapse** (planet/centrifuge, real/sim, etc.): a policy can farm 50% by always guessing the majority. Mitigate by (a) balancing the label prior to exactly 50/50, (b) the 15–20% honest-control worlds + tracked false-alarm rate, (c) gating the shaping reward on having run the discriminating experiment.
- **Redundant-instrument "spam both tools" habit** if T0's pooling lesson leaks upward: at T2+ make redundant instruments genuinely redundant and disable count-shaping above T1.
- **Sin(lat)→0 traps:** for spin tasks, draw `lab_latitude_deg` away from the equator (|lat| ∈ [20,70]) so the Foucault signal is recoverable; keep the equator trap only as an explicit advanced variant.
- **Height-baseline tasks may collapse to noise** at planet-sized R within the 0.1–1000 m cap (g-vs-altitude over ≤1 km is ~1e-4 relative). Widen the height range, shrink the drawn R, or verify with the reference-agent sim before shipping.
- **No reward-form leakage in the briefing:** state only the answer schema + the fixed decision rule/thresholds — never the hidden value, never which hypothesis is true. Mirror the byte-identical-default discipline; add a test that a world-kind briefing does not contain the ground-truth label string.

**The single cross-cutting tool that de-risks all of the above:** a **reference-agent winnability harness** (light, essential). For each new world distribution, run an oracle with the true latents + expected post-k-repeat error and confirm (a) the success band is achievable and (b) the naive/degenerate strategy provably misses — *before* a training run. Extend the existing `eval_grpo` / `analyze_aggregation` scaffolding.

---

## Why this ordering (the progression logic)

T0 mints the primitive (read→repeat→aggregate→box). T1 turns one primitive into a **chain** and mints the reusable **R subroutine** + the classification answer-form. T2 makes the chain links **unreliable** (bias/degeneracy/alias) — the prerequisite for all epistemic work, because *you can't question the setting until you can question an instrument*. T3 scales the output from scalar to **set/vector** and the input from one reading to a designed **campaign**. T4 attaches a committed **action** to the end of a full pipeline (reusing T0 aggregation, T1 chains, T2 confound-rejection, T3 planning) with simulated verification. T5 is pure **multi-hypothesis arbitration** that reuses the discriminating experiments the curveball track has been teaching since T0 — the integration exam. The curveball thread runs orthogonally through all six so the epistemic muscle is mature by the time T5 makes it the whole task.</curriculum_doc_markdown>
</invoke>
