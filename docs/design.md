# Design notes

Why the code is shaped the way it is. The implementation plan says *what* to
build; this records the decisions made while building it, especially the ones
where the obvious approach is wrong.

---

## 1. Actions are ordinary tokens

An action is a short run of tokens from the model's own vocabulary, so
predicting one is literally next-token prediction. There is no coordinate
regression head, no separate action decoder, no post-processing step that turns
a hidden state into a click.

```
<MOVE_CLICK> <ROW_14> <COL_9> <OFFSET_37>
```

Coordinates address the ViT's **own patch grid**, following SeeClick and
CogAgent. `<ROW_14> <COL_9>` names patch 14*24+9, which is a real embedding the
model attends over. That is why `model/vit.py` keeps every patch token and never
pools to a CLS vector: pooling would destroy the structure the coordinate
vocabulary depends on.

### Click resolution is a config decision with real consequences

Precision is `screen_width / (grid_cols * offset_grid)`.

| `offset_grid` | extra tokens | precision on a 1920px screen |
|---|---|---|
| 3 | 9 | ~27px |
| 4 | 16 | ~20px |
| **8 (default)** | **64** | **10px** |
| 16 | 256 | 5px |

The plan suggests a small offset vocabulary; the default here is 8 because 27px
is wider than many real UI targets — a toolbar icon is often 16-24px — and the
cost of a finer grid is 55 extra vocabulary entries, which is nothing. Raise it
further if `click_accuracy`'s `median_miss_px` comes back close to
`quantization_error_px`; that comparison is in the report precisely so this
decision can be made from evidence.

### Both absolute and relative pointer actions

Plan section 8 flags this as a decision to make early, and the answer is both.
"Click the submit button" wants absolute grounding; "drag this slider a bit
further" wants a relative delta, because the target has no fixed screen
position. `MOVE_CLICK` and friends cover the first, `MOVE_REL` with bucketed
deltas covers the second, and drags combine them.

---

## 2. The grammar is what makes real time possible

`ActionGrammar` is a state machine over action tokens, and constrained decoding
masks the logits with it at every step. This does three jobs at once:

1. **Bounded decode length.** After `<MOVE_CLICK>` only row tokens are legal,
   then only column tokens, then only offsets. The action completes in exactly
   four steps. There is no sampling until an EOS that may not come.
2. **No malformed actions, ever.** The harness never parses free text into an
   action, so there is no "the model emitted something weird" branch to get
   wrong.
3. **The two speeds.** A control tick is 2-4 tokens. Open-ended generation runs
   only after `<TYPE_START>`, which happens once per field being typed rather
   than 15 times a second. This is the whole reason a 0.5-1B LM can sit in a
   real-time loop.

---

## 3. Field-fill is routing, not generation

Plan section 2.3 says field-fill should be trained as copy/selection rather than
generation. This implementation takes that literally.

After `<TYPE_START>`, the model may emit `<FIELD_k>`, which selects the k-th key
of the supplied `form_data`. The harness then types `form_data[k]` **verbatim**.

```
<TYPE_START> <FIELD_0> <TYPE_END>     ->  types form_data["email"] exactly
<TYPE_START> ...text... <TYPE_END>    ->  types what the model generated
```

The consequence is that field-fill exactness is *structural*. The model cannot
mistype an email address, because it never types it — it points at it. A loss
weight can only make drift less likely; this makes it impossible.

The routing decision is still learned, and that is where the remaining risk
lives. `field_fill_exactness` reports `routing_errors` separately: free-composing
the right string still counts as exact, but it means the guarantee is not being
used, and on an unfamiliar form it would not have been.

---

## 4. Context ordering is forced by the KV cache

```
[ instruction ][ form_data ]   [ image patches ][ action history ]   -> action
\-------- static prefix -------/ \------- recomputed every tick -----/
```

The instruction comes **first**, before the image. This looks backwards next to
most VLMs, which put the image first, and it is not a style choice.

A causal KV cache is reusable only up to the first token that changes. The screen
changes every frame. So anything cacheable must precede the image, or it is
re-encoded 15 times a second for no reason. `PolicySession` encodes the
instruction once per subtask and trims the cache back to the prefix between
ticks.

That trim is verified rather than assumed. `crop()` changed meaning across
transformers versions (a target length, then a negative count of tokens to
drop), and a cache left one token too long misaligns every position after it —
producing plausible, confidently wrong actions with no error anywhere.

`data/dataset.py::_render_prefix` and `GuiPolicy.build_prefix` must produce
identical text. A test asserts it. Prompt skew between training and serving
shows up only as degraded rollouts, which is a miserable thing to debug.

---

## 5. Only the new embeddings train

Base LM weights are frozen; LoRA adapters, the projector, the ViT and the ~250
new action-token embedding rows train.

The new rows need special handling. `peft`'s `modules_to_save=["embed_tokens"]`
would make the *entire* embedding matrix trainable — hundreds of millions of
parameters, every one a chance to drift the language priors the LM was brought
in for. Instead the weight stays trainable and a gradient hook zeroes every row
outside the action block, so the optimiser can only move rows that started from
noise. A test asserts gradients below the block are exactly zero.

Those rows also get their own optimiser group at a higher learning rate and no
weight decay. They start from noise, so they need to move faster than adapters
sitting on already-meaningful weights; and decaying an embedding toward zero is
forgetting what the token means, not regularisation.

### Checkpoints carry a vocabulary signature

The action tokens' *order* determines which embedding row each one trained.
Load a checkpoint against a reordered vocabulary and the embeddings are silently
remapped: the model clicks in the wrong place, and nothing raises. So every
checkpoint stores a signature of its vocabulary and every load checks it.

---

## 6. Stop conditions the plan's pseudocode does not have

Two extra ones, both found by running the loop:

- **`WAIT` does not count toward "stuck".** A wait is *supposed* to leave the
  screen unchanged. Counting it escalates out of every page load — the exact
  situation waiting exists for.
- **Repeated identical actions escalate.** A policy hammering one control on a
  screen with a spinner or a blinking caret never hashes equal, so the visual
  change detector alone never fires. This is the most common way a policy gets
  stuck in practice.

Frame comparison is a perceptual hash, not equality, so a blinking text caret
does not read as progress.

And the harness escalates after repeated safety blocks. If the guard keeps
refusing, the policy is trying to do something it must not, and grinding away at
it is worse than handing back.

---

## 7. `DONE` is an opinion, not a fact

A policy that has drifted emits `DONE` confidently at the wrong moment — that is
what drift looks like from the inside. So the harness checks `success_criteria`
independently before reporting `done`, and downgrades to `escalated` when the
check fails.

A verifier that *errors* reports "not met", not "met". Otherwise a broken API key
turns every run green, which is the worst possible failure mode for a metric
whose job is catching false success.

---

## 8. Escalation is trained in both directions

Plan section 8 calls escalation calibration a genuine ML problem. The
implementation reflects that in the data:

- A rollout that ground on until it timed out **should** have handed back
  earlier — its last ticks become positive `ESCALATE` examples.
- A rollout that succeeded should **not** have escalated anywhere — every one of
  its ticks is negative evidence.

Training only the positive direction produces a policy that escalates
constantly, which defeats the purpose of the whole system. That is why
successful rollouts are included in the stage-3 dataset by default.

Correspondingly, `escalation_metrics` reports `silent_failures` and
`spurious_escalations` separately. A single accuracy number hides the trade
completely: never escalating and always escalating can score identically.

---

## 9. Uncorrected failures are dropped, not cloned

In stage 3, a tick from a failed rollout that nobody corrected is **not** a
training example. It is not evidence of anything — training on it clones the
behaviour that just failed.

Only two things become targets: an explicit correction, or an `ESCALATE` label
where the rollout was already past saving. Successful rollouts contribute
normally, because those are on-policy states with known-good actions, which is
exactly the distribution DAgger exists to cover.

Stage-2 data is mixed back in during stage 3. Training only on corrections is a
fast route to forgetting everything that already worked.

---

## 10. Guardrails are at dispatch, and cannot be skipped

The orchestrating LLM sees a request and a result, never the individual actions.
So `Dispatcher.send` consults the guard itself rather than trusting callers to,
and every check is a property of the action and the screen — never of the
model's confidence. A policy that has generalised badly is precisely the case
where its own judgement is worth least.

The guard fails closed on an unidentifiable foreground window: without knowing
which app has focus, the app blocklist cannot be enforced, and a mis-click into
a terminal is exactly what it exists to prevent.

`Verdict.__bool__` returns its `allowed` flag, which reads nicely as
`if verdict:` and bit this code twice — a blocked verdict is *falsy*, so
`if result.verdict` takes the wrong branch precisely when something was blocked.
Both call sites now test `is not None`. If you add a third, do the same.

---

## 11. What to measure, and in what order

1. **Field-fill exactness** — should be 1.0. Not a quality metric.
2. **Click accuracy** — against element bounding boxes, with the miss
   distribution compared to the grid's quantisation error.
3. **Escalation precision and recall** — separately, on a slice containing
   deliberately impossible tasks.
4. **Task success** — end to end, verifier-confirmed.
5. **Latency p50/p95** — on the target hardware, quantised, split into vision /
   prefill / decode because each has a different fix.

Latency comes last on purpose (plan 7.2 step 7): optimising a model that is not
yet accurate tells you how fast you can be wrong.
