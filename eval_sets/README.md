# Evaluation sets

Three suites, matching the metrics in the plan. Each is a `eval.jsonl` file in
its own directory alongside the screenshots it references.

Define these **before** training. A metric added afterwards tends to be one the
model already happens to pass.

Run them with:

```bash
python -m gui_agent.train.cli eval --policy runs/stage2/best          # offline suites
python -m gui_agent.train.cli eval --policy runs/stage2/best --live   # adds end-to-end
```

---

## `click_accuracy/eval.jsonl`

Does a click land on the element it was meant to?

```json
{
  "screenshot": "click_accuracy/frames/gmail_compose.png",
  "instruction": "click the compose button",
  "bbox": [24, 138, 148, 182],
  "element_id": "compose",
  "screen_w": 1920,
  "screen_h": 1080
}
```

`bbox` is `[left, top, right, bottom]` in screen pixels. Prefer it over
`point` + `tolerance_px`: a 40px miss is fine on a large button and fatal on a
toolbar icon, and only the bounding box knows which.

The report includes the grid's own quantisation error. If the median miss is
close to it, the coordinate resolution is the bottleneck and raising
`ActionSpaceConfig.offset_grid` will help more than more training will.

**Cover small targets deliberately.** A suite of large buttons will report high
accuracy from a model that cannot hit a 16px icon.

---

## `field_fill/eval.jsonl`

Does a supplied value get copied exactly?

```json
{
  "screenshot": "field_fill/frames/checkout_email.png",
  "instruction": "fill in the email field",
  "form_data": {"email": "j.doe@example.com", "name": "Jane Doe"},
  "expected_value": "j.doe@example.com"
}
```

Exactness should be 1.0. Anything less is a bug, not a quality shortfall — the
`<FIELD_k>` routing makes drift structurally impossible, so a miss means the
model routed to free-compose instead. `routing_errors` in the report counts
exactly that, including cases where free-compose happened to produce the right
string.

**Vary the form layouts.** Narrow coverage here shows up in production as
confident, wrong typing on unfamiliar forms — the model learns "this kind of
box takes the email" rather than "use the supplied value".

---

## `end_to_end_tasks/eval.jsonl`

Does the whole thing work? Run live through the harness.

```json
{
  "task_id": "gmail_send_reply",
  "instruction": "reply to the top email saying the invoice is attached",
  "form_data": {},
  "success_criteria": "the reply has been sent and the thread shows the new message",
  "timeout_s": 90,
  "should_escalate": false
}
```

`should_escalate` is the escalation slice. Include tasks that **cannot** be
completed — a button that is not there, a page that fails to load, a login the
policy has no credentials for. A policy that never escalates scores well on
every other metric while failing silently, and only these rows catch it.

Roughly a third of this suite should be impossible on purpose. Report
escalation precision and recall separately from task success; averaging them
hides the trade the plan warns about, where over-escalating defeats the point
of the system and under-escalating makes failures invisible.
