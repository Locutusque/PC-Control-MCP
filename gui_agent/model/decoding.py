"""Two-speed constrained decoding (plan section 3.3).

The control tick must complete in a few milliseconds, and an unconstrained LM
asked for "an action" can wander off into prose.  Both problems have the same
fix: mask the logits with the action grammar, so at every step only legal
continuations are reachable.  A ``<MOVE_CLICK>`` is then followed by a row, a
column and an offset by construction, the action completes in exactly 4 tokens,
and the harness never has to parse something malformed.

The two speeds:

======================  ============================  ==================
path                    trigger                       decode length
======================  ============================  ==================
control tick            every frame                   forced 2-4 tokens
type event              ``<TYPE_START>`` was emitted   open until ``<TYPE_END>``
======================  ============================  ==================

This is what makes reusing a 0.5-1B LM compatible with a real-time loop: the
expensive open-ended generation runs once per field being typed, not 15 times a
second.

Field-fill never generates at all.  ``<TYPE_START> <FIELD_k> <TYPE_END>`` is
three tokens, and the harness copies ``form_data[k]`` verbatim -- exactness is
structural, not something sampling has to get right.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch

from ..actions import Action, ActionCodec, ActionParseError, DecodeState, TokenClass
from .tokenizer_ext import ActionTokenizer

log = logging.getLogger(__name__)

__all__ = ["DecodeResult", "ConstrainedActionDecoder", "SamplingConfig"]


@dataclass(frozen=True)
class SamplingConfig:
    """Control ticks are greedy; only free-compose text samples.

    A sampled click is a click in the wrong place, and at 15Hz the policy gets
    many chances to be wrong.  Typed prose is the one place variety helps.
    """

    greedy_control: bool = True
    text_temperature: float = 0.7
    text_top_p: float = 0.9


@dataclass
class DecodeResult:
    action: Action | None
    token_ids: list[int]
    atoms: list[str]
    n_steps: int
    truncated: bool = False
    error: str | None = None
    # True when the open-ended type path ran, so callers can attribute latency.
    used_type_path: bool = False

    @property
    def ok(self) -> bool:
        return self.action is not None


class ConstrainedActionDecoder:
    """Grammar-masked decoding of exactly one action."""

    def __init__(
        self,
        tokenizer: ActionTokenizer,
        codec: ActionCodec | None = None,
        sampling: SamplingConfig | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.codec = codec or ActionCodec(tokenizer.action_space)
        self.grammar = self.codec.grammar
        self.sampling = sampling or SamplingConfig()
        self.vocab_size = len(tokenizer)
        self._masks: dict[DecodeState, torch.Tensor] = {}
        self._text_mask: torch.Tensor | None = None

    # -- masks ------------------------------------------------------------
    def _text_ids_mask(self) -> torch.Tensor:
        """Every id free-compose text may use."""
        if self._text_mask is None:
            mask = torch.ones(self.vocab_size, dtype=torch.bool)
            start, end = self.tokenizer.text_ids_excluded()
            mask[start:end] = False  # action tokens are never free text
            for special in (self.tokenizer.pad_token_id, self.tokenizer.eos_token_id):
                if special is not None and 0 <= special < self.vocab_size:
                    mask[special] = False
            self._text_mask = mask
        return self._text_mask

    def mask_for(self, state: DecodeState) -> torch.Tensor:
        """Boolean vocabulary mask of legal continuations in ``state``."""
        cached = self._masks.get(state)
        if cached is not None:
            return cached

        classes = self.grammar.allowed_classes(state)
        mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        for cls in classes:
            if cls is TokenClass.TEXT:
                mask |= self._text_ids_mask()
            else:
                start, end = self.tokenizer.class_span(cls)
                mask[start:end] = True
        if not mask.any():
            raise ActionParseError(f"no legal continuation in state {state.value}")
        self._masks[state] = mask
        return mask

    def allowed_ids(self, state: DecodeState) -> list[int]:
        return self.mask_for(state).nonzero(as_tuple=True)[0].tolist()

    # -- selection --------------------------------------------------------
    def _select(self, logits: torch.Tensor, state: DecodeState) -> int:
        """Pick one token id under the grammar mask."""
        if logits.dim() > 1:
            logits = logits[-1] if logits.shape[0] > 1 else logits.reshape(-1)
        logits = logits[: self.vocab_size].float()

        mask = self.mask_for(state).to(logits.device)
        logits = logits.masked_fill(~mask, float("-inf"))

        free_text = state is DecodeState.IN_FREE_TEXT or state is DecodeState.AWAIT_TYPE_BODY
        if self.sampling.greedy_control and not free_text:
            return int(logits.argmax().item())
        return int(_sample(logits, self.sampling.text_temperature, self.sampling.text_top_p))

    # -- the loop ---------------------------------------------------------
    def decode(
        self,
        step: Callable[[list[int]], torch.Tensor],
        form_fields: Sequence[str] | None = None,
        max_control_steps: int | None = None,
        max_type_steps: int | None = None,
    ) -> DecodeResult:
        """Decode one action.

        ``step`` takes the ids generated so far this action (empty on the first
        call) and returns next-token logits.  Keeping the model interaction
        behind that callable is what lets the policy own KV-cache management
        while the grammar stays a pure function of the token stream.
        """
        max_control = max_control_steps or self.grammar.max_control_atoms()
        max_type = max_type_steps or self.codec.config.max_type_tokens

        state = self.grammar.start()
        chain: list[DecodeState] = []
        ids: list[int] = []
        used_type_path = False
        budget = max_control

        while not self.grammar.is_complete(state):
            if len(ids) >= budget:
                return DecodeResult(
                    None, ids, self.tokenizer.decode_atoms(ids), len(ids),
                    truncated=True, used_type_path=used_type_path,
                    error=f"hit the {budget}-token budget in state {state.value}",
                )
            try:
                token_id = self._select(step(ids), state)
            except ActionParseError as exc:
                return DecodeResult(None, ids, self.tokenizer.decode_atoms(ids), len(ids),
                                    used_type_path=used_type_path, error=str(exc))
            ids.append(token_id)

            atom = self.tokenizer._id_to_atom.get(token_id)
            if atom == "<TYPE_START>":
                # Switch to the slow path: open-ended generation, one field at
                # a time, instead of the real-time control budget.
                used_type_path = True
                budget = max_type
            try:
                state, chain = self.grammar.step(state, atom if atom is not None else "", chain)
            except ActionParseError as exc:  # pragma: no cover - the mask prevents this
                return DecodeResult(None, ids, self.tokenizer.decode_atoms(ids), len(ids),
                                    used_type_path=used_type_path, error=str(exc))

        atoms = self.tokenizer.decode_atoms(ids)
        try:
            action = self.codec.decode(atoms, form_fields)
        except ActionParseError as exc:
            return DecodeResult(None, ids, atoms, len(ids),
                                used_type_path=used_type_path, error=str(exc))
        return DecodeResult(action, ids, atoms, len(ids), used_type_path=used_type_path)


def _sample(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    if temperature <= 0:
        return int(logits.argmax().item())
    probs = torch.softmax(logits / temperature, dim=-1)
    if 0 < top_p < 1:
        ordered, index = torch.sort(probs, descending=True)
        cumulative = ordered.cumsum(dim=-1)
        # Keep the first token that crosses the threshold, so top_p never
        # empties the candidate set.
        cut = (cumulative - ordered) > top_p
        ordered[cut] = 0.0
        total = ordered.sum()
        if total <= 0:  # pragma: no cover - only if every prob underflowed
            return int(logits.argmax().item())
        choice = torch.multinomial(ordered / total, 1)
        return int(index[choice].item())
    return int(torch.multinomial(probs, 1).item())
