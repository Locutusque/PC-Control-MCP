"""Extending the base LM tokenizer with the action vocabulary (plan 2.1/3.2).

The action tokens are added as *special* tokens in exactly the order
:class:`~gui_agent.actions.ActionVocab` defines, so their ids form one
contiguous block at the end of the vocabulary.  Three things depend on that:

* the freshly-initialised embedding rows the trainer optimises separately are a
  contiguous slice;
* constrained decoding can mask a whole token class with a slice rather than a
  gather;
* a checkpoint can be validated against the vocabulary it was trained with --
  :meth:`ActionTokenizer.signature` is stored alongside the weights, because a
  reordered vocabulary produces a model that clicks in the wrong place and
  raises nothing.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from ..actions import ActionVocab, TokenClass
from ..config import ActionSpaceConfig

log = logging.getLogger(__name__)

__all__ = ["ActionTokenizer", "TokenizerMismatch"]


# Atoms of this shape are meant to be action tokens; anything matching it that
# is not in the vocabulary is a bug, not free-compose text.
_ACTION_SHAPED = re.compile(r"^<[A-Z][A-Z0-9_+-]*>$")


class TokenizerMismatch(RuntimeError):
    """A checkpoint's action vocabulary does not match the one in use."""


@dataclass
class _Span:
    start: int
    end: int  # exclusive

    def ids(self) -> list[int]:
        return list(range(self.start, self.end))


class ActionTokenizer:
    """A base tokenizer plus the action-token block."""

    def __init__(self, base_tokenizer, action_space: ActionSpaceConfig | None = None) -> None:
        self.base = base_tokenizer
        self.action_space = action_space or ActionSpaceConfig()
        self.vocab = ActionVocab(self.action_space)

        self.base_vocab_size = len(base_tokenizer)

        # A tokenizer saved from a trained policy already carries the whole
        # block, and reloading a checkpoint must go down that path rather than
        # trying to add the tokens a second time. Adding only *some* of them,
        # though, means the ids are no longer one contiguous appended block,
        # which everything downstream assumes -- that stays an error.
        existing = base_tokenizer.convert_tokens_to_ids(list(self.vocab.tokens))
        unknown = base_tokenizer.unk_token_id
        present = sum(
            1 for i in existing if i is not None and (unknown is None or i != unknown)
        )
        self.reloaded = present == len(self.vocab)

        if not self.reloaded:
            if present:
                raise TokenizerMismatch(
                    f"{present} of {len(self.vocab)} action tokens are already present in "
                    "the base vocabulary. Appending the rest would leave the block "
                    "non-contiguous. Use a fresh tokenizer, or one saved from a "
                    "checkpoint built with this exact action space."
                )
            added = base_tokenizer.add_special_tokens(
                {"additional_special_tokens": list(self.vocab.tokens)}
            )
            if added != len(self.vocab):
                raise TokenizerMismatch(
                    f"expected to add {len(self.vocab)} action tokens but the tokenizer "
                    f"accepted {added}"
                )

        ids = base_tokenizer.convert_tokens_to_ids(list(self.vocab.tokens))
        if len(set(ids)) != len(ids) or any(i is None for i in ids):
            raise TokenizerMismatch("action tokens did not map to distinct ids")
        if ids != list(range(min(ids), min(ids) + len(ids))):
            raise TokenizerMismatch(
                "action token ids are not contiguous; constrained decoding and the "
                "new-embedding parameter group both assume a single contiguous block"
            )

        self.action_id_start = min(ids)
        self.action_id_end = self.action_id_start + len(ids)
        self._atom_to_id = dict(zip(self.vocab.tokens, ids, strict=True))
        self._id_to_atom = {i: t for t, i in self._atom_to_id.items()}

        # One contiguous span per class, in ActionVocab order.
        self._spans: dict[TokenClass, _Span] = {}
        cursor = self.action_id_start
        for cls in (
            TokenClass.ACTION, TokenClass.TYPE_END, TokenClass.ROW, TokenClass.COL,
            TokenClass.OFFSET, TokenClass.DX, TokenClass.DY, TokenClass.SCROLL_DX,
            TokenClass.SCROLL_DY, TokenClass.KEY, TokenClass.WAIT, TokenClass.FIELD,
        ):
            n = len(self.vocab.of_class(cls))
            self._spans[cls] = _Span(cursor, cursor + n)
            cursor += n

    # -- construction -----------------------------------------------------
    @classmethod
    def from_pretrained(cls, model_name: str, action_space: ActionSpaceConfig | None = None,
                        **kwargs) -> ActionTokenizer:
        from transformers import AutoTokenizer

        base = AutoTokenizer.from_pretrained(model_name, **kwargs)
        if base.pad_token is None:
            # Padding is masked out of the loss anyway; reusing EOS avoids
            # adding a token that would break the contiguity check above.
            base.pad_token = base.eos_token
        return cls(base, action_space)

    def save_pretrained(self, path) -> None:
        self.base.save_pretrained(path)

    # -- properties -------------------------------------------------------
    def __len__(self) -> int:
        return len(self.base)

    @property
    def n_action_tokens(self) -> int:
        return len(self.vocab)

    @property
    def pad_token_id(self) -> int:
        return self.base.pad_token_id

    @property
    def eos_token_id(self) -> int:
        return self.base.eos_token_id

    def signature(self) -> str:
        """Fingerprint tying a checkpoint to this exact vocabulary."""
        return f"{self.vocab.signature()}@{self.action_id_start}"

    def check_signature(self, expected: str | None) -> None:
        if expected and expected != self.signature():
            raise TokenizerMismatch(
                f"checkpoint was trained with action vocabulary {expected!r} but the "
                f"current configuration produces {self.signature()!r}. The action-token "
                "embeddings would be silently misaligned; rebuild the tokenizer from the "
                "checkpoint's action_space.json."
            )

    # -- ids --------------------------------------------------------------
    def is_action_id(self, token_id: int) -> bool:
        return self.action_id_start <= token_id < self.action_id_end

    def atom_id(self, atom: str) -> int:
        try:
            return self._atom_to_id[atom]
        except KeyError:
            raise KeyError(f"{atom!r} is not an action token") from None

    def class_ids(self, cls: TokenClass) -> list[int]:
        """Every token id in a class -- the mask used by constrained decoding."""
        span = self._spans.get(cls)
        return span.ids() if span else []

    def class_span(self, cls: TokenClass) -> tuple[int, int]:
        span = self._spans[cls]
        return span.start, span.end

    def text_ids_excluded(self) -> tuple[int, int]:
        """The action-token span, which free-compose text must never enter."""
        return self.action_id_start, self.action_id_end

    # -- atoms <-> ids ----------------------------------------------------
    def encode_atoms(self, atoms, add_eos: bool = False) -> list[int]:
        """Atom list -> token ids.

        An atom present in the action vocabulary becomes its single fixed id;
        anything else is free-compose text and goes through the base tokenizer.
        """
        ids: list[int] = []
        for atom in atoms:
            fixed = self._atom_to_id.get(atom)
            if fixed is not None:
                ids.append(fixed)
                continue
            if _ACTION_SHAPED.match(atom):
                # An out-of-range coordinate or a typo would otherwise be
                # tokenized as ordinary text and become a garbage training
                # target, with nothing to indicate anything went wrong.
                raise KeyError(
                    f"{atom!r} looks like an action token but is not in this action "
                    f"space ({self.action_space.grid_rows}x{self.action_space.grid_cols} "
                    f"grid, offset_grid={self.action_space.offset_grid})"
                )
            ids.extend(self.base.encode(atom, add_special_tokens=False))
        if add_eos and self.eos_token_id is not None:
            ids.append(self.eos_token_id)
        return ids

    def decode_atoms(self, ids) -> list[str]:
        """Token ids -> atoms, re-joining runs of text into single chunks."""
        atoms: list[str] = []
        buffer: list[int] = []

        def flush() -> None:
            if buffer:
                atoms.append(self.base.decode(buffer, skip_special_tokens=True))
                buffer.clear()

        for token_id in ids:
            atom = self._id_to_atom.get(int(token_id))
            if atom is None:
                buffer.append(int(token_id))
            else:
                flush()
                atoms.append(atom)
        flush()
        return atoms

    # -- context ----------------------------------------------------------
    def encode_text(self, text: str, max_length: int | None = None) -> list[int]:
        ids = self.base.encode(text, add_special_tokens=False)
        return ids[:max_length] if max_length else ids

    def resize_embeddings(self, model) -> None:
        """Grow the LM's embedding matrix and lm_head to fit the new tokens."""
        current = model.get_input_embeddings().weight.shape[0]
        if current >= len(self.base):
            return
        # mean_resizing seeds new rows near the embedding mean rather than at
        # random, which keeps the first training steps from being dominated by
        # noise in the ~250 action tokens.
        try:
            model.resize_token_embeddings(len(self.base), mean_resizing=True)
        except TypeError:  # older transformers
            model.resize_token_embeddings(len(self.base))
        log.info("resized embeddings %d -> %d", current, len(self.base))

    def new_token_slice(self) -> slice:
        """The rows of the embedding matrix that train from scratch."""
        return slice(self.action_id_start, self.action_id_end)
