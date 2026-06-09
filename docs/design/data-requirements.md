# Procedural task space + how much data for gains vs generalization — answer for the alien-physics-lab

## TL;DR (direct answers)

1. **Can the task space be made procedurally meaningful (vs ~80 hard-coded problems)? YES — and "80 problems" is the wrong unit; kill it.** Every reference system in this space is a *generator library*, not a list: Reasoning Gym = 100+ generators / ~10 categories with "virtually infinite" non-repeating instances ([2505.24760](https://arxiv.org/abs/2505.24760)); RLVE = 400 verifiable environments ([2511.07317](https://arxiv.org/abs/2511.07317)); InternBootcamp = 1000+ generators (704 retained, [2508.08636](https://arxiv.org/abs/2508.08636)); SynLogic = 35 generators ([2505.19641](https://arxiv.org/abs/2505.19641)). The meaningful unit is **TASK FAMILIES (generators) × WITHIN-FAMILY STRUCTURAL DIVERSITY**, where each per-seed instance is a deterministic `f(seed)` over {latents, advertised tool subset, confounds/distractors, chain depth}. **This project is already on that footing** (verified: `world.py:31-38` computes `effective_gravity` from {mass, diameter, spin, latitude} so one physics core spawns many families; `grpo_data.py` draws latents+noise+tolerance+tool-subset+prompt-template as decorrelated `f(seed)`). 80 *fixed instances* would be memorization bait; the same 80 concepts re-expressed as ~10-30 generators emitting thousands of seeds each is effectively unbounded and memorization-resistant.

2. **How much data for GAINS (in-distribution reward up)?** Shockingly little — the floor is **1 prompt**, a few hundred GRPO steps. This is the regime the current runs are already in. **But this is the weakest evidence of anything** — it is mostly *sharpening/format-correction* of latent base ability, not new capability, and on Qwen specifically even spurious rewards reproduce most of it. Treat "reward up" as a training diagnostic only.

3. **How much data for GENERALIZATION (held-out worlds/tools/families)?** A **different and much larger** budget, and the lever is **diversity (kinds), not raw count**. Within-family: thousands-to-tens-of-thousands of distinct seeds per family. Across-family: generalization is a near-**step function** in the number of distinct *structures* (~300-1000 in the cleanest study). This is the owner's actual goal and needs a separate budget + a four-axis held-out split.

These are **two distinct regimes** and a third failure mode (memorizing-the-generator); conflating them is the central error to avoid. Details, numbers, citations, and confidence flags below.

---

## 1. Procedural feasibility — how many generators + how much within-family diversity

**Confidence: HIGH on the qualitative design; MEDIUM on the specific count targets (no clean literature minimum exists for agentic multi-turn tool-use RLVR — see caveats).**

Generator *count alone is not enough* — within-family instance diversity is the co-gating factor (Procgen: a few hundred seeds/family overfits regardless of generator quality; [1812.02341](https://arxiv.org/abs/1812.02341)). The two axes:

- **Across-family / structural diversity** is the binding constraint and generalization is a near-step function in it. **Only-IF** ([2410.04717](https://arxiv.org/abs/2410.04717), *VERIFIED*): below **I_min ≈ 300** distinct instructions a model never generalizes regardless of examples-each; above **I_max ≈ 1000** it generalizes even with few examples each, at fixed budget I·S = 1e6. *Caveat (verifier): this is string-rewrite SFT and its mechanism is semantic/cross-domain diversity; a sub-task needed ~400 not 300. It is an order-of-magnitude design target, not a measured threshold for this agentic setting.*
- **RLVE** ([2511.07317](https://arxiv.org/abs/2511.07317), *VERIFIED*): held-out-environment OOD improved **monotonically** as env count grew C1→C4→C16→C256, **no ceiling found**, and env-scaling beat 3× more compute on the same data (**+3.37% vs +0.49%**).
- **InternBootcamp** ([2508.08636](https://arxiv.org/abs/2508.08636), *VERIFIED*): an "Emergent Moment" phase transition at the **512-task** mixture (~300 steps), and 512-task training beats few-task training *even on the narrow tasks*.

**Concrete targets for THIS project (Qwen3-1.7B, GRPO):**

- **TASK FAMILIES / GENERATORS: target 10-30 now, architected to scale toward ~50-100.** Rationale: a "structure" = family × confound-type × chain-depth × tool-subset, so **10-30 families × ~5-10 confound variants × 2-4 chain depths × 3-5 tool subsets already yields 300-1000+ distinct structures** — i.e. you can clear the Only-IF step-function band from a modest family count. *Don't chase 256+ envs like RLVE for a feasibility project.* Family roadmap from the existing physics core: gravity-direct, gravity-via-mass+radius, gravity-via-pendulum, diameter-via-horizon-dip (shipped, `grpo_data.py:135-139`), spin-via-Foucault (forked), atmospheric-drag inference, density/composition (chemistry), thermal-expansion, orbital-period→mass (astronomy), plus the **epistemic curveballs** (centrifuge-vs-planet "what-kind-of-world", is-there-drag, is-a-tool-lying). Curveballs are the **highest-value OOD** per **DELTA-Code** ([2509.21016](https://arxiv.org/abs/2509.21016)): "transformative shifts requiring new invariants" are the hardest and most diagnostic.
- **WITHIN-FAMILY DIVERSITY: 2,000-10,000 distinct seeds per family minimum.** Procgen quantifies this (CoinRun train-test gap: ~32.7% @100 levels → ~23.2% @1000 → ~11.8% @4000 → ~1.7% @16000; *gap closes near ~10-16k levels*). *Caveat (verifier): the exact gap decimals could not be independently re-confirmed from the PDF; the qualitative curve — overfit at hundreds, gap closes ~16k — is well-established ([1812.02341](https://arxiv.org/abs/1812.02341)). Treat the decimals as research-synthesis-reported, verify against the table before any publication.* The code generates seeds on the fly, so this is free — keep `n_train` large, seeds unique, and vary ALL of {latent, noise (already log-uniform), precision (`vary_precision`), tool subset (`vary_tools`), prompt (`vary_prompt`), + new confound/chain-depth knobs}.

---

## 2. Data for GAINS (in-distribution) — the cheap, weak regime

**Confidence: HIGH that the floor is very low; HIGH (verifier-corrected) that most of it is sharpening/format, not learning. All headline numbers are Qwen-(Math-)specific — see the prominent caveat.**

- **1-shot RLVR** ([2504.20571](https://arxiv.org/abs/2504.20571)): a single example took Qwen2.5-Math-1.5B on MATH500 from **36.0% → 73.6% raw**. **⚠️ VERIFIER CORRECTION — do not use the raw +37.6pp as the headline:** the paper itself attributes only **+8.6pp to non-format reasoning gains** (verbatim: "8.6% improvement beyond format correction"); ~29pp of the jump is the base model simply learning to emit a parseable `\boxed{}`. 6-benchmark avg: **+7.0pp** non-format. *This correction is on-message — it strengthens the point that cheap in-distribution gains are mostly sharpening/format, not learned skill.*
- **LIMR** ([2502.11886](https://arxiv.org/abs/2502.11886), *VERIFIED*): **1,389 selected prompts == full 8,523**, and beat a same-size *random* subset (AIME24 **32.5% vs 25.8%**). Selection/diversity > volume.
- **Learning from Less** ([2604.18381](https://arxiv.org/abs/2604.18381), *VERIFIED*; Qwen3-4B + GRPO on counting/graph/spatial): gains often **peak at 100-200 examples then DECLINE** (counting: ~50% @100ex falls to 40% @500 = overfit; spatial-easy peaks 56.7% @200, falls to 53.1% @500).

**Practical ceiling for the GAINS regime on this project:** a few hundred to ~1.4k distinct training seeds per family at ~250-400 GRPO steps. This *matches the current alien-lab runs* (VERIFIED against docs): diameter run hits **physics 0.476 / reward 0.869 / n≈6** in ~250 steps at lr=2e-6 ([06-07](docs/results/2026-06-07-grpo.md)); beta=0.0075@lr=4e-6 reaches **physics 0.60 → 0.70 at 350 steps**, n→12, stable ([06-08](docs/results/2026-06-08-grpo.md)).

**⚠️ LOAD-BEARING CAVEAT — most of these gains may be elicitation, not learned physics:**
- *Random/spurious rewards* recovered **+21.4pp of the +29.1pp** ground-truth gain on Qwen2.5-Math-7B but **FAILED on Llama3/OLMo2** ([2506.10947](https://arxiv.org/abs/2506.10947), *VERIFIED*). Since this project trains Qwen3-1.7B, a "gain" could be eliciting pretraining gravity priors (g≈9.8 is in every textbook), not learned experimental skill.
- **All the dramatic small-data numbers above (1-shot, LIMR, spurious, RG transfer) are nearly all Qwen2.5(-Math)-centric** (verifier emphasis). **Transplanting any specific percentage to Qwen3-1.7B on physics is unjustified — only the qualitative design rules transfer.** Run a non-Qwen control (recommendation iii below).

---

## 3. Data for GENERALIZATION — the real target, diversity-driven

**Confidence: HIGH on the qualitative "diversity > count" rule (multiple independent confirmations); MEDIUM on portability of the specific thresholds to agentic multi-turn RLVR.**

- **DIVE** ([2603.11076](https://arxiv.org/abs/2603.11076), *VERIFIED*; agentic tool-use, most analogous to this env's tool subsets): diversity-scaling beats quantity-scaling at **4× less data**, **+22 avg pts** across 9 OOD benchmarks (+68% over the strongest 8B baseline, 373 tools, Qwen3-8B). **⚠️ VERIFIER QUALIFICATION:** DIVE is **48k SFT + 3.2k RL (SFT-dominant data synthesis), NOT pure RLVR** — the diversity>quantity finding transfers as a *design principle*, but the data-scale numbers are not directly an RLVR-prompt-count anchor.
- **G-Vendi** ([2505.20161](https://arxiv.org/abs/2505.20161)): gradient-diversity correlates **Spearman ρ ≈ 0.9** with OOD performance.
- **Curriculum:** mixed-complexity gives up to **5× sample efficiency** vs easy-only ([2604.18381](https://arxiv.org/abs/2604.18381)); RG's adaptive curriculum (advance when >70% acc over 20 steps) beat fixed difficulty in *every* env (spell-backwards +40.67pp, [2505.24760](https://arxiv.org/abs/2505.24760)). **But easy worlds must be FADED OUT** or they re-introduce overfitting (E2H [2506.06632](https://arxiv.org/abs/2506.06632); scaling-laws-by-difficulty [2508.19069](https://arxiv.org/abs/2508.19069) shows held-out-*hard* accuracy drops monotonically as easy synthetic data grows).
- **Far upper bound — ProRL** ([2505.24864](https://arxiv.org/abs/2505.24864), *VERIFIED*): genuine boundary *expansion* needed **136K diverse prompts / 5 domains / 2000+ steps / ~16K H100-hrs** + KL penalty + periodic reference-policy hard resets (gains math +15.7 / code +14.4 / STEM +25.9 / IF +22.0 / **logic +54.8%** vs DeepSeek-R1-Distill-Qwen-1.5B — *cite this one table consistently per verifier; the synthesis carried two slightly inconsistent versions*). **Overkill for a 1.7B feasibility project**, but it's the existence proof that expansion (not just sharpening) is possible with diversity + duration + stabilization.

**For this project's GENERALIZATION push:** train on the FULL family mixture jointly (RLVE/InternBootcamp both show joint-mixture > narrow even on narrow tasks), **800-2000 steps** (RG used ~800; ProRL needed 2000+ for expansion), with the worlds-per-step batching fix already landed (4 worlds/step). Curriculum: advance when >70% acc over ~20 steps (RG) or expand difficulty bound when acc@hard > 0.9 (RLVE τ=0.9), and **fade out easy worlds late**. Keep beta tuned per the existing finding (diameter sweet spot **beta=0.0075 @ lr=4e-6**; KL is the de-facto entropy regulator and prevents the collapse that kills OOD).

---

## 4. The three-way distinction (the central conceptual point)

| | What it is | Budget | Evidence strength |
|---|---|---|---|
| **(1) In-distribution gains** | reward rises on the *training* distribution | 1 → ~1.4k prompts, few hundred steps | **WEAKEST** — largely sharpening/format; on Qwen even random rewards produce most of it |
| **(2) Generalization** | held-out worlds/tools/families never seen | hundreds of *structures* × thousands of seeds, 800-2000 steps | **the owner's actual goal** |
| **(3) Memorizing-the-generator** | policy fits the generator's surface statistics (noise dist, template, fixed n) | — | looks generalizing on held-out *seeds* of the *same* family; **fails on held-out families/tools/difficulty** |

The sharpest warning on (3): the **LIS/Activity-Scheduling** case ([2510.27044](https://arxiv.org/abs/2510.27044), *VERIFIED*) — RLVR raised answer accuracy via *shallow heuristics*, not reasoning (outputs predictable from shallow features at **R² 0.781 / 0.841**), and answer-only reward *collapsed* intermediate reasoning. For a noisy physics-recovery task this is the dominant risk: the model can hit `\boxed{g}` by fitting reading-noise statistics or regurgitating g≈9.8.

**The project's structural moat (its single strongest, best-supported advantage):** per-seed hidden latents generated *post-pretraining-cutoff* make it **contamination-proof by construction**. Contrast Wu et al. ([2507.10532](https://arxiv.org/abs/2507.10532), *VERIFIED*): Qwen reconstructs **54.6%** of held-out MATH500 from 60% prefixes vs Llama **3.8%**, dropping to **0.0%** on post-cutoff LiveMathBench. Lean on this.

In one line: **gains need ~1k prompts and prove little; generalization needs ~hundreds of structures × thousands of seeds and is the real target; memorization is what a seed-only held-out set fails to detect — you need held-out families, tools, AND difficulty too.**

---

## 5. Recommendation — concrete for this project

**Held-out split design (the load-bearing part — build FOUR nested axes, not one):**
- **(a) Held-out SEEDS** within trained families — *already implemented* (`grpo_data.py:72-73`, EVAL_SEED_START=1_000_000). Detects within-family instance memorization; **necessary but WEAK — passing this alone is the trap.**
- **(b) Held-out DIFFICULTY** — train easy/mid latent+noise+precision, eval hard. *Motivation:* difficulty extrapolation is intrinsically hard — **⚠️ VERIFIER RE-ATTRIBUTION:** the sharp RG drops (code −71.9%, graphs −33.8%, geometry −33.1%, algorithms −25.6%) are the **o3-mini eval-only difficulty cliff** ([2505.24760](https://arxiv.org/html/2505.24760v2)), **NOT** the held-out-difficulty gap of an RL-trained policy. Use them as motivation that difficulty extrapolation is hard, not as a measured RL-policy failure rate.
- **(c) Held-out TOOLS** — eval a tool combination never advertised together in training (DIVE-style).
- **(d) Held-out FAMILIES** — leave 1-3 entire generators (esp. a curveball) out, eval cold. **This is the owner's true OOD goal** and the cross-domain-transfer test (RG saw algorithmic→algebra **+29.1pp**, →geometry +22.3pp, *VERIFIED*).

Report all four separately. Plus three controls:
- **(i) Report pass@k, not just pass@1.** **⚠️ VERIFIER CORRECTION:** the "+25pp pass@1 vs +2pp pass@8" pair is **fabricated-as-specific — NOT in Yue et al.** Cite the real claim: base pass@k overtakes RL at k up to **1024** (math/visual) / **128** (code), with **no universal crossover-k** ([2504.13837](https://arxiv.org/abs/2504.13837)). The instrumentation (log pass@k) is sound; the inference "flat pass@8 ⇒ no capability gain" is itself **contested** ([2511.16231](https://arxiv.org/abs/2511.16231)).
- **(ii) Non-Qwen control** (Llama3.x / OLMo2) on the headline eval — if gains vanish, they were Qwen elicitation artifacts, not learned physics ([2506.10947](https://arxiv.org/abs/2506.10947)).
- **(iii) Scrambled-physics / inverted-constants control** — confirm the model infers from experiments rather than regurgitating Earth priors. The wide **GRAVITY_MIN=0.4g .. GRAVITY_MAX=2.2g** (`grpo_data.py:30-31`, *VERIFIED*) already partly does this — push wider and verify accuracy at the extremes.

**Data targets, summarized:** 10-30 families (architected to ~50-100) × 2,000-10,000 seeds/family; gains regime ~250-400 steps; generalization push joint-mixture 800-2000 steps with faded-easy curriculum; beta=0.0075 @ lr=4e-6 (diameter recipe).

---

## 6. Overfitting / memorization signatures to monitor (for THIS project)

1. **In-distribution reward rises while any held-out axis (seed/difficulty/tool/family) is flat or DECLINES** as GRPO entropy collapses — the canonical memorization signature ([2508.19069](https://arxiv.org/abs/2508.19069): held-out-hard accuracy drops monotonically as easy data grows). *Log eval reward on all four held-out splits every N steps, not just train reward.*
2. **pass@1 climbs but pass@k (k=8/64/256) stays flat** — RLVR compressed search rather than expanding capability ([2504.13837](https://arxiv.org/abs/2504.13837)). *Caveat: this inference is contested ([2511.16231](https://arxiv.org/abs/2511.16231)) — use as a diagnostic flag, not a verdict.*
3. **Final-answer accuracy rises while the EXPERIMENTAL PROCEDURE degenerates** — n_experiments collapses to 1, measurement reward → 0, or the model emits `\boxed{9.8}` without calling tools. This is the LIS shortcut ([2510.27044](https://arxiv.org/abs/2510.27044)). The repo *already logs* n_experiments / call-freq / measurement-reward — alert if **accuracy and n_experiments decouple** (accuracy up, n flat/low). *Note the live tension: beta=0.01@lr4e6 froze the policy at n=1.0 with measurement reward exactly 0.000 ([06-08](docs/results/2026-06-08-grpo.md)) — a frozen-not-learning state that looks stable.*
4. **Shortcut-predictability probe:** regress the model's `\boxed{}` answer on shallow world features (first/only reading, stated noise, template index). High R² (cf. 0.78-0.84 in [2510.27044](https://arxiv.org/abs/2510.27044)) ⇒ pattern-matching the generator, not aggregating noisy physics. Run on held-out seeds.
5. **Gains vanish on a non-Qwen base or shrink under scrambled-physics** ⇒ the "physics" was Qwen pretraining-prior elicitation ([2506.10947](https://arxiv.org/abs/2506.10947)).
6. **Accuracy degrades specifically at the EXTREMES of the latent range** (g near 0.4g / 2.2g; smallest/largest diameters) while strong near Earth values ⇒ the policy anchored on the prior mean. Bin held-out accuracy by latent magnitude.
7. **Entropy runaway as a false positive:** rising reward + entropy 0.5→1.5+ + grad_norm climbing is *instability*, not learning — distinct from memorization but equally invalidates a "gain." This is the documented **beta=0.005 @ lr=4e-6** divergence (entropy 0.55→1.68 in ~12 steps; [06-08](docs/results/2026-06-08-grpo.md)). Already watched via the entropy/grad_norm/kl watchdog.
8. **Large, family-specific train-easy/eval-hard gap** ⇒ the model interpolates within the trained difficulty band but does not extrapolate the procedure. Treat as memorization of the difficulty band.

---

## Contested claims, flagged honestly

- **SHARPENING vs EXPANSION is UNRESOLVED and directly gates the OOD goal.** Yue et al. ([2504.13837](https://arxiv.org/abs/2504.13837), base pass@k overtakes RL = sharpening only) vs ProRL ([2505.24864](https://arxiv.org/abs/2505.24864)) / RL-Grokking ([2509.21016](https://arxiv.org/abs/2509.21016), new transferable algorithms via grokking) reach opposite conclusions. **⚠️ VERIFIER:** do NOT treat "most cheap gains are sharpening, not new capability" as established fact — it is contested. Yue measured *fixed benchmarks the base already covers*; ProRL/Grokking pick tasks where base pass@k=0. For a 1.7B model on *novel* physics (low base-model grip), which regime governs the alien lab is **genuinely unknown until measured** (Yue's own framing is itself disputed, [2510.04028](https://arxiv.org/abs/2510.04028)).
- **pass@k as the arbiter is itself contested** ([2511.16231](https://arxiv.org/abs/2511.16231): pass@k is a diagnostic of exploration, not an objective). Sound to log; not settled as a capability verdict.
- **The cheap-gains numbers are Qwen-specific** (1-shot, LIMR, spurious rewards, RG/DeepScaleR transfer) — only qualitative design rules transfer to Qwen3-1.7B.
- **Only-IF (~300/1000) and Procgen (~10-16k) are not portable thresholds** — string-rewrite SFT and pixel deep-RL respectively. Order-of-magnitude design targets, loud caveat, not laws for agentic multi-turn RLVR.