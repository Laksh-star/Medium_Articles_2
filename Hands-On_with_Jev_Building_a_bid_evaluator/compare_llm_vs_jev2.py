"""Same four questions, same two proposals: Jev vs a general LLM with a forced JSON schema.

Put this file in the same folder as run_bid_evaluator.py (it imports the texts and rubric from it).

Setup (once):
    pip install requests anthropic
    export TYPESAFE_API_KEY="your-typesafe-key"
    export ANTHROPIC_API_KEY="your-anthropic-key"

Run:
    python compare_llm_vs_jev.py                    # 20 runs per vendor per system
    python compare_llm_vs_jev.py --runs 5           # quick check first
    python compare_llm_vs_jev.py --llm-model claude-sonnet-4-5
    python compare_llm_vs_jev.py --via-claude-code  # NO Anthropic key: uses your logged-in Claude Code
    python compare_llm_vs_jev.py --dry-run          # no keys, fake data, tests the plumbing

A third, deliberately borderline bid (Cirrus) is included: it sits between rubric levels on every question.
--via-claude-code runs `claude -p` (headless) with tools off and a JSON schema, so the model is forced
to return four integers. Latency there is the API time Claude Code reports, not the CLI start-up.

Keys are read from the environment only, never printed or saved.
Writes compare_results.json (safe to share) and prints a summary table.
"""
import argparse, json, os, random, statistics as st, subprocess, sys, time
import requests
import run_bid_evaluator as rb

KEYS = list(rb.WEIGHTS)

# A deliberately borderline third bid: every dimension sits between two levels of the rubric.
CIRRUS = ("Cirrus Networks proposes a 3-year managed cloud hosting contract starting from 36,000 dollars per month. "
          "Storage above 50 terabytes and premium support tiers are billed separately at rates to be agreed at "
          "contract signature. We target 99.95 percent availability, and service credits may apply in certain "
          "circumstances; incident response times vary by severity and are set out in the support schedule. "
          "Cirrus is ISO 27001 certified, has a SOC 2 Type II audit in progress, encrypts data at rest, and is "
          "reviewing in-transit encryption standards with customers. Implementation is estimated at 10 to 12 weeks, "
          "depending on migration scope to be confirmed in a discovery phase, with any delays handled through "
          "the change request process.")
STATES = dict(rb.STATES)
STATES["Cirrus"] = {"vendor_name": "Cirrus Networks", "proposal_text": CIRRUS}


def rubric_prompt(state):
    parts = ["Score this vendor proposal on each question below. For each question pick ONE integer level "
             "from 0 to 4, where the level is the index of the matching criterion (first criterion = 0).\n",
             f"Vendor: {state['vendor_name']}\nProposal: {state['proposal_text']}\n"]
    for k, q in rb.QUESTIONS.items():
        parts.append(f"Question `{k}`: {q['instructions']}")
        parts += [f"  {i}: {c}" for i, c in enumerate(q["criteria"])]
        parts.append("")
    return "\n".join(parts)


TOOL = {"name": "record_scores", "description": "Record the level for each question.",
        "input_schema": {"type": "object", "additionalProperties": False, "required": KEYS,
                         "properties": {k: {"type": "integer", "minimum": 0, "maximum": 4} for k in KEYS}}}


def run_llm(client, model, state, dry):
    if dry:
        time.sleep(0.01)
        return {k: random.choice([1, 2]) for k in KEYS}, 1500.0, None
    t0 = time.perf_counter()
    msg = client.messages.create(model=model, max_tokens=300, tools=[TOOL],
                                 tool_choice={"type": "tool", "name": "record_scores"},
                                 messages=[{"role": "user", "content": rubric_prompt(state)}])
    wall = (time.perf_counter() - t0) * 1000
    for b in msg.content:
        if b.type == "tool_use":
            out = b.input
            if all(isinstance(out.get(k), int) and 0 <= out[k] <= 4 for k in KEYS):
                return out, wall, None
            return None, wall, f"schema violation: {out}"
    return None, wall, "no tool call returned"


def run_llm_cc(model, state):
    schema = {"type": "object", "additionalProperties": False, "required": KEYS,
              "properties": {k: {"type": "integer", "minimum": 0, "maximum": 4} for k in KEYS}}
    cmd = ["claude", "-p", rubric_prompt(state), "--output-format", "json", "--json-schema", json.dumps(schema),
           "--tools", "", "--no-session-persistence", "--system-prompt", "You score vendor proposals against rubrics."]
    if model:
        cmd += ["--model", model]
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        d = json.loads(p.stdout)
    except Exception as e:
        return None, (time.perf_counter() - t0) * 1000, f"claude -p failed: {e}"
    wall = d.get("duration_api_ms") or (time.perf_counter() - t0) * 1000
    out = d.get("structured_output")
    if out is None:
        try: out = json.loads(d.get("result", ""))
        except Exception: return None, wall, f"no structured output: {str(d.get('result'))[:80]}"
    if all(isinstance(out.get(k), int) and 0 <= out[k] <= 4 for k in KEYS):
        return out, wall, None
    return None, wall, f"schema violation: {out}"


def run_jev(key, state, dry):
    if dry:
        return {k: round(random.uniform(1, 2), 2) for k in KEYS}, {k: 0.9 for k in KEYS}, 850.0
    resp, wall = rb.call(key, state)
    a = resp["answers"]
    return {k: a[k]["score"] for k in KEYS}, {k: a[k]["confidence"] for k in KEYS}, wall


def spread(xs):
    return (round(st.pstdev(xs), 3), round(max(xs) - min(xs), 2)) if len(xs) > 1 else (0, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--llm-model", default=os.environ.get("LLM_MODEL", "claude-sonnet-4-5"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--via-claude-code", action="store_true", help="use `claude -p` instead of an Anthropic API key")
    ap.add_argument("--cc-model", default=None, help="model alias for Claude Code, e.g. sonnet (default: its default)")
    a = ap.parse_args()
    tkey, client = os.environ.get("TYPESAFE_API_KEY"), None
    if not a.dry_run:
        if not tkey:
            sys.exit("Set TYPESAFE_API_KEY first (see top of file).")
        if not a.via_claude_code:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                sys.exit("Set ANTHROPIC_API_KEY, or use --via-claude-code.")
            import anthropic
            client = anthropic.Anthropic()

    out = {"llm_model": ("claude-code:" + (a.cc_model or "default")) if a.via_claude_code else a.llm_model, "runs": a.runs, "jev": {}, "llm": {}}
    for vendor, state in STATES.items():
        jev, llm, fails, jw, lw = [], [], [], [], []
        for i in range(a.runs):
            s, c, w = run_jev(tkey, state, a.dry_run); jev.append(s); jw.append(w)
            r, w2, err = (run_llm_cc(a.cc_model, state) if a.via_claude_code and not a.dry_run
                          else run_llm(client, a.llm_model, state, a.dry_run))
            if err: fails.append(err)
            else: llm.append(r); lw.append(w2)
            print(f"{vendor} {i+1}/{a.runs}", end="\r", flush=True)
        out["jev"][vendor] = {"scores": jev, "wall_ms": jw}
        out["llm"][vendor] = {"scores": llm, "wall_ms": lw, "failures": fails}

    print("\n\n%-8s %-22s %-18s %-18s" % ("Vendor", "Question", "Jev mean (sd,rng)", "LLM mean (sd,rng)"))
    for v in STATES:
        for k in KEYS:
            j = [r[k] for r in out["jev"][v]["scores"]]
            l = [r[k] for r in out["llm"][v]["scores"]]
            js, ls = spread(j), spread(l)
            print("%-8s %-22s %-6.2f (%s,%s)   %-6.2f (%s,%s)" % (
                v, k, st.mean(j), js[0], js[1], st.mean(l) if l else float('nan'), ls[0], ls[1]))
    print()
    for name in ("jev", "llm"):
        w = [x for v in STATES for x in out[name][v]["wall_ms"] if x]
        f = sum(len(out[name][v].get("failures", [])) for v in STATES)
        print(f"{name.upper()}: median wall {st.median(w):.0f} ms, p95 {sorted(w)[int(len(w)*0.95)-1]:.0f} ms, "
              f"failures {f}/{a.runs*len(STATES)}")
    fn = "compare_results_dryrun.json" if a.dry_run else "compare_results.json"
    json.dump(out, open(fn, "w"), indent=2)
    print(f"\nSaved {fn} (no keys inside).")


if __name__ == "__main__":
    main()
