#!/usr/bin/env python
"""Watchdog for a long unattended training run. Polls the run's stdout log every
POLL seconds and EXITS (which notifies the launching agent) on the first of:
  * the run finished/crashed   ("DONE ec=" present)
  * entropy collapse           (latest entropy < COLLAPSE_ENTROPY)
  * stall/hang                 (no new step for STALL_POLLS polls)
  * milestone reached          (step >= MILESTONE)
On exit it prints a windowed trend (early/mid/recent) so the agent can judge health.
Usage: watch_run.py <output_file> [milestone_step]
"""
import ast, re, sys, time

f = sys.argv[1]
MILESTONE = int(sys.argv[2]) if len(sys.argv) > 2 else 500
POLL = 300
COLLAPSE_ENTROPY = 0.05   # entropy below this => policy collapse (over-confident)
DIVERGE_ENTROPY = 1.5     # entropy above this => policy divergence (runaway; healthy is ~0.3-0.6)
STALL_POLLS = 5  # 5 * 300s = 25 min with no new step => stalled

def rows():
    out = []
    for line in open(f, errors="ignore"):
        m = re.search(r"\{'loss'.*?'epoch'[^}]*\}", line)
        if m:
            try: out.append(ast.literal_eval(m.group(0)))
            except Exception: pass
    return out

def cur_step():
    s = 0
    for m in re.finditer(r"(\d+)/\d+ \[", open(f, errors="ignore").read()):
        s = int(m.group(1))
    return s

last_step, stall = -1, 0
reason = "unknown"
while True:
    time.sleep(POLL)
    txt = open(f, errors="ignore").read()
    if "DONE ec=" in txt:
        reason = "FINISHED/CRASHED"; break
    rs = rows()
    ent = float(rs[-1].get("entropy", 1.0)) if rs else 1.0
    step = cur_step()
    if rs and ent < COLLAPSE_ENTROPY:
        reason = f"ENTROPY COLLAPSE (entropy={ent:.4f})"; break
    if rs and ent > DIVERGE_ENTROPY:
        reason = f"ENTROPY DIVERGENCE (entropy={ent:.4f} > {DIVERGE_ENTROPY})"; break
    stall = stall + 1 if step == last_step else 0
    last_step = step
    if stall >= STALL_POLLS:
        reason = f"STALLED (no new step in {STALL_POLLS*POLL//60} min, step={step})"; break
    if step >= MILESTONE:
        reason = f"MILESTONE step>={MILESTONE}"; break

print(f"=== WATCHDOG WAKE: {reason} ===")
rs = rows()
if rs:
    n = len(rs); t = max(1, n // 3)
    def win(key, a, b):
        v = [float(r.get(key, 0) or 0) for r in rs[a:b]]
        return round(sum(v) / len(v), 3) if v else None
    print(f"logged rows={n} (~step {2*n}); early | mid | recent:")
    for k in ["rewards/physics_reward/mean", "entropy", "grad_norm",
              "tools/call_frequency", "frac_reward_zero_std"]:
        print(f"  {k:34s} {win(k,0,t)} | {win(k,t,2*t)} | {win(k,2*t,n)}")
