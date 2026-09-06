# gui-agent

A small neural policy that drives a GUI in a closed loop, so an LLM
orchestrator is called once per *subtask* instead of once per *action*.

Today, an LLM controlling a computer pays a full round trip — screenshot in,
tool call out — for every click. This replaces that inner loop with a ~0.6-1.1B
vision-language policy running locally at 10-30Hz, exposed to the orchestrator
as a single MCP tool:

```jsonc
execute_low_level_task({
  "instruction": "fill in the checkout form and submit it",
  "form_data": {"email": "j.doe@example.com", "name": "Jane Doe"},
  "success_criteria": "an order confirmation is visible",
  "timeout_s": 120
})
// -> {"status": "done", "summary": "done: policy reported done [7x MOVE_CLICK, 2x TYPE]",
//     "final_screenshot": {...}}
```

The orchestrator sees the request and that result. The two hundred control ticks
in between never enter its context. That is the whole point.

Built on the GUI-native VLA line of work — SeeClick, CogAgent, OS-Atlas, ShowUI,
UI-TARS for coordinate-as-token grounding, OmniParser for screen parsing, and
VPT for the unlabelled-pretrain then instruction-finetune shape.

> **Status:** the full pipeline is implemented and tested end to end (193 tests,
> including a real training step and a closed-loop rollout against a saved
> checkpoint). No policy has been trained — that needs captured data and a GPU.
> See [Where this actually stands](#where-this-actually-stands).

---

## How it works

An action is a short run of ordinary vocabulary tokens, so predicting one is
literally next-token prediction:

```
<MOVE_CLICK> <ROW_14> <COL_9> <OFFSET_37>
```

Coordinates address the ViT's own patch grid, so `<ROW_14> <COL_9>` names a real
patch embedding. Three design choices follow from that and carry most of the
weight:

**A grammar makes the control tick bounded.** Constrained decoding masks the
logits with an action state machine, so after `<MOVE_CLICK>` only row tokens are
legal, then columns, then offsets. Every control action completes in 2-4 forced
steps and the harness can never receive something malformed. Open-ended
generation runs only after `<TYPE_START>` — once per field typed, not 15 times a
second. This is what lets a 0.5-1B LM sit inside a real-time loop.

**Field-fill is selection, not generation.** When a value is supplied in
`form_data`, the model emits `<TYPE_START> <FIELD_0> <TYPE_END>` and the harness
copies `form_data["email"]` verbatim. The model cannot mistype an email address
because it never types it — it points at it. Exactness is structural rather than
something a loss has to coax out of a generator.

**The instruction comes before the image.** A causal KV cache is reusable only
up to the first token that changes, and the screen changes every frame, so
anything cacheable has to precede the image:

```
[ instruction ][ form_data ]   [ image patches ][ action history ]   -> action
\-------- static prefix -------/ \------- recomputed every tick -----/
```

`PolicySession` encodes the instruction once per subtask instead of fifteen
times a second.

| Component | Choice | Params |
|---|---|---|
| Vision | ViT, patch 32 @ 768px, all 576 patch tokens kept | ~88M |
| Bridge | 2-layer MLP (or Perceiver resampler, 576 -> 144 tokens) | 3.5M / 16M |
| Decoder | Pretrained LM + LoRA, base frozen | 0.5-1B |
| Action vocab | 250 tokens (default action space) | — |

---

## Quick start

```bash
pip install -e '.[all]'
```

### 0. Not comfortable with a command line?

Double-click the setup file for your OS in [`install/`](install/) instead —
it does everything below for you, and stops to ask before it starts
recording anything. See [`install/README.md`](install/README.md).

### 1. Capture

```bash
python -m gui_agent.capture.cli doctor    # can this machine record safely?
python -m gui_agent.capture.cli keygen    # store in your OS keychain
python -m gui_agent.capture.cli record
```

`doctor` refuses to start unless the app blocklist is enforceable and segments
can be encrypted. **Read [docs/privacy.md](docs/privacy.md) first** — this
records everything you do, and the defences only work if you configure them for
your machine.

```bash
python -m gui_agent.capture.cli promote   # redaction sweep, then into the pool
python -m gui_agent.capture.cli status
```

### 2. Build datasets

```bash
python -m gui_agent.data.cli relabel --pool data_pool     # LLM hindsight instructions
python -m gui_agent.data.cli stage1  --out data/stage1.jsonl
python -m gui_agent.data.cli stage2  --out data/stage2.jsonl --balance
```

`stage2` prints coverage warnings before you spend a GPU-week on a dataset with
four distinct form fields in it.

### 3. Train

```bash
python -m gui_agent.train.cli stage1 --data data/stage1.jsonl
python -m gui_agent.train.cli stage2 --data data/stage2.jsonl --init runs/stage1/best
python -m gui_agent.train.cli benchmark --policy runs/stage2/best --target-hz 15
```

### 4. Serve

```bash
python -m gui_agent.harness.server --policy runs/stage2/best
```

Refuses to start outside a VM or container unless you say otherwise. The policy
dispatches un-reviewed input to a real desktop.

### 5. Close the loop (ongoing)

```bash
python -m gui_agent.train.cli collect --policy runs/stage2/best --tasks examples/tasks.jsonl
python -m gui_agent.train.cli stage3  --data data/stage3.jsonl --init runs/stage2/best \
                                      --mix data/stage2.jsonl
```

Stage 3 is not a milestone with an end date. It is where GUI agents go from
working in demos to being reliable, and it costs calendar time.

---

## Layout

```
gui_agent/
  actions.py            action vocabulary, grammar state machine, pixel codec
  config.py             every config object; ActionSpaceConfig is load-bearing
  capture/              Stage 0: recording daemon, privacy layer, segmentation
  data/                 trajectory encoding, stage 1/2 datasets, relabelling
  model/                ViT, projector, tokenizer extension, policy, decoding
  train/                three stages, metrics, eval, latency benchmark
  harness/              safety guard, dispatch, control loop, MCP tool
docs/design.md          why the code is shaped this way
docs/privacy.md         handling guide for the capture pool
eval_sets/              click accuracy, field fill, end-to-end tasks
```

---

## Safety

The orchestrating LLM never reviews individual actions, so the guardrails are at
the dispatch layer and cannot be bypassed by a policy that has learned something
unexpected:

- sandbox required (VM, container, or an explicit opt-out);
- foreground app blocklist — terminals, system settings, keychains;
- an unidentifiable foreground window is **refused**, since the blocklist
  cannot be enforced against a window we cannot see;
- optional screen-region restriction;
- destructive text patterns and escape-hatch key chords blocked outright;
- rate limit and per-subtask action budget;
- every dispatched and blocked action logged for post-hoc review.

Stop conditions: `DONE` (verified independently), `ESCALATE`, no visual change
over K ticks, the same action repeated, repeated safety blocks, timeout.

---

## Where this actually stands

Implemented, tested, and working:

- the action space, grammar and constrained decoder;
- the capture daemon with its privacy layer, and the promotion pipeline;
- trajectory encoding and both dataset builders, with the field-fill split;
- the policy, verified end to end — weighted loss, gradient flow, cached
  closed-loop inference, checkpoint save/load;
- the harness, with every stop condition exercised;
- the MCP tool;
- the training stages, metrics and latency benchmark.

Not done, and not fakeable from here:

- **No trained policy.** Everything above runs; none of it has seen real data.
- **Base LM not chosen.** The default is a placeholder. Plan section 8 is right
  that this should be settled by benchmarking quantised latency on the target
  hardware — `train.cli benchmark` exists for exactly that, and should be run
  before committing to the training pipeline.
- **Eval sets are templates.** Two rows each, referencing screenshots that are
  not committed. The schema and the guidance are there; the data is yours to
  collect.
- **Platform backends are uneven.** Linux/X11 is the most complete. macOS and
  Windows paths are written against the documented APIs but have not been run on
  those platforms — the accessibility calls in particular deserve a careful look
  before you trust `capture.cli doctor` on them.
- **Wayland is largely unsupported**, by design of the compositor rather than an
  oversight here. `doctor` will tell you.

The two risks most likely to bite, both flagged in the plan and neither solved
by code:

- **Field-fill coverage.** The routing decision is only as good as the variety
  of forms in stage-2 data. Narrow coverage shows up as confident, wrong typing
  on unfamiliar layouts, and it will not appear in the loss curve.
- **Escalation calibration.** Under-escalating causes silent failures;
  over-escalating defeats the point of the system. It is trained in both
  directions and measured with precision and recall separately, but it needs an
  eval slice containing genuinely impossible tasks to mean anything.

---

## Development

```bash
python -m pytest tests/ -q     # 193 tests, ~5s on CPU, no downloads
ruff check gui_agent tests
```

Model tests build a tiny real policy — 2-layer ViT, 2-layer Llama, a byte-BPE
tokenizer trained in the fixture — rather than mocking. Nothing is downloaded,
and the real shapes, cache handling and grammar masking are exercised.

If you change the action vocabulary's **order**, existing checkpoints become
silently wrong: token position determines which embedding row each action
trained. The vocabulary signature stored in every checkpoint exists to catch
that, and it should stay append-only.
