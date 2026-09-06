"""The discrete action vocabulary and its grammar (plan section 2).

Every GUI action is a short run of ordinary vocabulary tokens, so predicting an
action *is* next-token prediction -- there is no separate regression head, and
coordinates address the ViT's own patch grid (the SeeClick / CogAgent
formulation).

Three things live here and must stay in agreement, because they are the seam
where the model meets the operating system:

* :class:`ActionVocab` -- the ordered token list.  Order fixes embedding row
  indices, so it is append-only across checkpoints.
* :class:`ActionGrammar` -- the state machine that says which token classes may
  legally follow.  Constrained decoding uses it to force a valid 2-4 token
  action every control tick; the parser uses it to read one back.
* :class:`ActionCodec` -- conversion between an :class:`Action` and both its
  token form and concrete screen pixels.

Token *atoms*
-------------
A serialised action is a list of atoms.  An atom is either a special action
token (a string present in :class:`ActionVocab`) or an arbitrary text chunk
carrying free-compose typing.  Modules that own a real tokenizer
(``model/tokenizer_ext.py``) map the former to their fixed ids and run the
latter through the base LM tokenizer; nothing here needs a tokenizer, which
keeps the action space testable on its own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import Enum
from typing import Iterable, Iterator, Sequence

from .config import ActionSpaceConfig

__all__ = [
    "ActionType",
    "TokenClass",
    "Action",
    "ActionVocab",
    "ActionGrammar",
    "ActionCodec",
    "ActionParseError",
    "DecodeState",
]


class ActionParseError(ValueError):
    """Raised when a token stream does not form a legal action."""


class ActionType(str, Enum):
    MOVE_CLICK = "MOVE_CLICK"
    DOUBLE_CLICK = "DOUBLE_CLICK"
    RIGHT_CLICK = "RIGHT_CLICK"
    MOVE_REL = "MOVE_REL"
    DRAG_START = "DRAG_START"
    DRAG_END = "DRAG_END"
    SCROLL = "SCROLL"
    KEY = "KEY"
    TYPE = "TYPE"  # serialises as <TYPE_START> ... <TYPE_END>
    WAIT = "WAIT"
    DONE = "DONE"
    ESCALATE = "ESCALATE"

    @property
    def is_absolute_pointer(self) -> bool:
        return self in _ABSOLUTE_POINTER

    @property
    def is_terminal(self) -> bool:
        """Terminal actions end the subtask rather than driving the screen."""
        return self in (ActionType.DONE, ActionType.ESCALATE)


_ABSOLUTE_POINTER = frozenset(
    {
        ActionType.MOVE_CLICK,
        ActionType.DOUBLE_CLICK,
        ActionType.RIGHT_CLICK,
        ActionType.DRAG_START,
        ActionType.DRAG_END,
    }
)


class TokenClass(str, Enum):
    """Groups of tokens the grammar can permit as a set."""

    ACTION = "action"
    ROW = "row"
    COL = "col"
    OFFSET = "offset"
    DX = "dx"
    DY = "dy"
    SCROLL_DX = "scroll_dx"
    SCROLL_DY = "scroll_dy"
    KEY = "key"
    WAIT = "wait"
    FIELD = "field"
    TEXT = "text"  # any base-LM token: free-compose typing only
    TYPE_END = "type_end"


class DecodeState(Enum):
    """Position within one action."""

    AWAIT_ACTION = "await_action"
    AWAIT_ROW = "await_row"
    AWAIT_COL = "await_col"
    AWAIT_OFFSET = "await_offset"
    AWAIT_DX = "await_dx"
    AWAIT_DY = "await_dy"
    AWAIT_SCROLL_DX = "await_scroll_dx"
    AWAIT_SCROLL_DY = "await_scroll_dy"
    AWAIT_KEY = "await_key"
    AWAIT_WAIT = "await_wait"
    AWAIT_TYPE_BODY = "await_type_body"  # right after <TYPE_START>
    IN_FREE_TEXT = "in_free_text"
    COMPLETE = "complete"


# --------------------------------------------------------------------------
# Action
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Action:
    """One decoded action.

    Coordinate fields are grid indices, not pixels; use
    :meth:`ActionCodec.to_pixels` to resolve them against a real screen.
    """

    type: ActionType
    row: int | None = None
    col: int | None = None
    offset: int | None = None
    dx: int | None = None
    dy: int | None = None
    scroll_dx: int | None = None
    scroll_dy: int | None = None
    key: str | None = None
    wait_ms: int | None = None
    # Exactly one of these is set for a TYPE action.  ``field`` means
    # field-fill: the harness copies form_data[field] verbatim, so the value
    # cannot drift.  ``text`` means free-compose: the LM generated it.
    field: str | None = None
    text: str | None = None

    def __post_init__(self) -> None:
        t = self.type
        if t.is_absolute_pointer:
            _require(self, "row", "col", "offset")
        elif t is ActionType.MOVE_REL:
            _require(self, "dx", "dy")
        elif t is ActionType.SCROLL:
            _require(self, "scroll_dx", "scroll_dy")
        elif t is ActionType.KEY:
            _require(self, "key")
        elif t is ActionType.WAIT:
            _require(self, "wait_ms")
        elif t is ActionType.TYPE:
            if (self.field is None) == (self.text is None):
                raise ActionParseError(
                    "TYPE needs exactly one of field= (field-fill) or text= (free-compose)"
                )

    @property
    def is_field_fill(self) -> bool:
        return self.type is ActionType.TYPE and self.field is not None

    @property
    def is_free_compose(self) -> bool:
        return self.type is ActionType.TYPE and self.text is not None

    def resolve_text(self, form_data: dict | None) -> str:
        """The literal string this TYPE action should send to the OS."""
        if self.type is not ActionType.TYPE:
            raise ActionParseError(f"{self.type} is not a TYPE action")
        if self.text is not None:
            return self.text
        if not form_data or self.field not in form_data:
            raise ActionParseError(
                f"field-fill referenced {self.field!r}, absent from form_data"
            )
        return str(form_data[self.field])

    def summary(self) -> str:
        """One-line human-readable form, for audit logs and rollout traces."""
        if self.type.is_absolute_pointer:
            return f"{self.type.value}(r{self.row},c{self.col},o{self.offset})"
        if self.type is ActionType.MOVE_REL:
            return f"MOVE_REL({self.dx:+d},{self.dy:+d})"
        if self.type is ActionType.SCROLL:
            return f"SCROLL({self.scroll_dx:+d},{self.scroll_dy:+d})"
        if self.type is ActionType.KEY:
            return f"KEY({self.key})"
        if self.type is ActionType.WAIT:
            return f"WAIT({self.wait_ms}ms)"
        if self.type is ActionType.TYPE:
            if self.field is not None:
                return f"TYPE(field={self.field})"
            preview = (self.text or "")[:32]
            return f"TYPE({preview!r}{'...' if len(self.text or '') > 32 else ''})"
        return self.type.value


def _require(action: Action, *fields: str) -> None:
    missing = [f for f in fields if getattr(action, f) is None]
    if missing:
        raise ActionParseError(
            f"{action.type.value} requires {', '.join(fields)}; missing {', '.join(missing)}"
        )


# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------


def _fmt_signed(value: int) -> str:
    return f"+{value}" if value > 0 else str(value)


class ActionVocab:
    """The ordered block of special tokens added to the base tokenizer.

    Ordering is deterministic given an :class:`ActionSpaceConfig` and must be
    treated as append-only: token position determines which embedding row a
    checkpoint trained, so reordering silently remaps learned behaviour.
    """

    def __init__(self, config: ActionSpaceConfig | None = None) -> None:
        self.config = config or ActionSpaceConfig()
        cfg = self.config

        by_class: dict[TokenClass, list[str]] = {}

        action_tokens = [
            "<MOVE_CLICK>", "<DOUBLE_CLICK>", "<RIGHT_CLICK>", "<MOVE_REL>",
            "<DRAG_START>", "<DRAG_END>", "<SCROLL>", "<KEY>", "<TYPE_START>",
            "<WAIT>", "<DONE>", "<ESCALATE>",
        ]
        by_class[TokenClass.ACTION] = action_tokens
        by_class[TokenClass.TYPE_END] = ["<TYPE_END>"]
        by_class[TokenClass.ROW] = [f"<ROW_{i}>" for i in range(cfg.grid_rows)]
        by_class[TokenClass.COL] = [f"<COL_{i}>" for i in range(cfg.grid_cols)]
        by_class[TokenClass.OFFSET] = [f"<OFFSET_{i}>" for i in range(cfg.offset_grid**2)]
        by_class[TokenClass.DX] = [f"<DX_{_fmt_signed(d)}>" for d in cfg.delta_buckets]
        by_class[TokenClass.DY] = [f"<DY_{_fmt_signed(d)}>" for d in cfg.delta_buckets]
        srange = range(-cfg.scroll_range, cfg.scroll_range + 1)
        by_class[TokenClass.SCROLL_DX] = [f"<SCROLL_DX_{_fmt_signed(d)}>" for d in srange]
        by_class[TokenClass.SCROLL_DY] = [f"<SCROLL_DY_{_fmt_signed(d)}>" for d in srange]
        by_class[TokenClass.KEY] = [f"<KEY_{k}>" for k in cfg.keys]
        by_class[TokenClass.WAIT] = [_wait_token(ms) for ms in cfg.wait_ms]
        by_class[TokenClass.FIELD] = [f"<FIELD_{i}>" for i in range(cfg.max_form_fields)]

        # TokenClass.TEXT is intentionally absent: free-compose text uses the
        # base LM's own vocabulary, not this block.
        self._by_class = {k: tuple(v) for k, v in by_class.items()}

        ordered: list[str] = []
        for cls in (
            TokenClass.ACTION, TokenClass.TYPE_END, TokenClass.ROW, TokenClass.COL,
            TokenClass.OFFSET, TokenClass.DX, TokenClass.DY, TokenClass.SCROLL_DX,
            TokenClass.SCROLL_DY, TokenClass.KEY, TokenClass.WAIT, TokenClass.FIELD,
        ):
            ordered.extend(self._by_class[cls])

        if len(set(ordered)) != len(ordered):
            dupes = sorted({t for t in ordered if ordered.count(t) > 1})
            raise ValueError(f"action vocabulary contains duplicates: {dupes}")

        self.tokens: tuple[str, ...] = tuple(ordered)
        self._index = {tok: i for i, tok in enumerate(self.tokens)}
        self._class_of = {
            tok: cls for cls, toks in self._by_class.items() for tok in toks
        }

    # -- lookups ----------------------------------------------------------
    def __len__(self) -> int:
        return len(self.tokens)

    def __contains__(self, token: object) -> bool:
        return token in self._index

    def index(self, token: str) -> int:
        try:
            return self._index[token]
        except KeyError:
            raise ActionParseError(f"{token!r} is not an action token") from None

    def of_class(self, cls: TokenClass) -> tuple[str, ...]:
        return self._by_class.get(cls, ())

    def class_of(self, token: str) -> TokenClass:
        try:
            return self._class_of[token]
        except KeyError:
            raise ActionParseError(f"{token!r} is not an action token") from None

    def action_token(self, action_type: ActionType) -> str:
        return "<TYPE_START>" if action_type is ActionType.TYPE else f"<{action_type.value}>"

    def action_type_of(self, token: str) -> ActionType:
        if token == "<TYPE_START>":
            return ActionType.TYPE
        if not (token.startswith("<") and token.endswith(">")):
            raise ActionParseError(f"{token!r} is not an action-type token")
        try:
            return ActionType(token[1:-1])
        except ValueError:
            raise ActionParseError(f"{token!r} is not an action-type token") from None

    def signature(self) -> str:
        """Stable fingerprint of the vocabulary, stored with checkpoints."""
        import hashlib

        digest = hashlib.sha256("\n".join(self.tokens).encode()).hexdigest()
        return f"{len(self.tokens)}:{digest[:16]}"


def _wait_token(ms: int) -> str:
    return f"<WAIT_{ms // 1000}S>" if ms >= 1000 and ms % 1000 == 0 else f"<WAIT_{ms}MS>"


# --------------------------------------------------------------------------
# Grammar
# --------------------------------------------------------------------------

# Which state each action type enters after its opening token, and the chain of
# argument states that follows.
_ARG_CHAIN: dict[ActionType, tuple[DecodeState, ...]] = {
    ActionType.MOVE_CLICK: (DecodeState.AWAIT_ROW, DecodeState.AWAIT_COL, DecodeState.AWAIT_OFFSET),
    ActionType.DOUBLE_CLICK: (DecodeState.AWAIT_ROW, DecodeState.AWAIT_COL, DecodeState.AWAIT_OFFSET),
    ActionType.RIGHT_CLICK: (DecodeState.AWAIT_ROW, DecodeState.AWAIT_COL, DecodeState.AWAIT_OFFSET),
    ActionType.DRAG_START: (DecodeState.AWAIT_ROW, DecodeState.AWAIT_COL, DecodeState.AWAIT_OFFSET),
    ActionType.DRAG_END: (DecodeState.AWAIT_ROW, DecodeState.AWAIT_COL, DecodeState.AWAIT_OFFSET),
    ActionType.MOVE_REL: (DecodeState.AWAIT_DX, DecodeState.AWAIT_DY),
    ActionType.SCROLL: (DecodeState.AWAIT_SCROLL_DX, DecodeState.AWAIT_SCROLL_DY),
    ActionType.KEY: (DecodeState.AWAIT_KEY,),
    ActionType.WAIT: (DecodeState.AWAIT_WAIT,),
    ActionType.TYPE: (DecodeState.AWAIT_TYPE_BODY,),
    ActionType.DONE: (),
    ActionType.ESCALATE: (),
}

_STATE_CLASS: dict[DecodeState, TokenClass] = {
    DecodeState.AWAIT_ACTION: TokenClass.ACTION,
    DecodeState.AWAIT_ROW: TokenClass.ROW,
    DecodeState.AWAIT_COL: TokenClass.COL,
    DecodeState.AWAIT_OFFSET: TokenClass.OFFSET,
    DecodeState.AWAIT_DX: TokenClass.DX,
    DecodeState.AWAIT_DY: TokenClass.DY,
    DecodeState.AWAIT_SCROLL_DX: TokenClass.SCROLL_DX,
    DecodeState.AWAIT_SCROLL_DY: TokenClass.SCROLL_DY,
    DecodeState.AWAIT_KEY: TokenClass.KEY,
    DecodeState.AWAIT_WAIT: TokenClass.WAIT,
}


class ActionGrammar:
    """State machine over action atoms.

    Constrained decoding (``model/decoding.py``) asks :meth:`allowed_classes`
    for the legal continuations and masks every other logit, which is what lets
    the control tick force a complete, well-formed action in 2-4 steps instead
    of sampling freely and hoping.  It also guarantees the harness never has to
    handle a malformed action.
    """

    def __init__(self, vocab: ActionVocab | None = None) -> None:
        self.vocab = vocab or ActionVocab()

    def start(self) -> DecodeState:
        return DecodeState.AWAIT_ACTION

    def allowed_classes(self, state: DecodeState, pending: ActionType | None = None) -> frozenset[TokenClass]:
        """Token classes that may legally appear next."""
        if state is DecodeState.COMPLETE:
            return frozenset()
        if state is DecodeState.AWAIT_TYPE_BODY:
            # Right after <TYPE_START> the model routes: pick a supplied form
            # field (copy) or start generating (compose).  An immediate
            # <TYPE_END> types nothing and is disallowed.
            return frozenset({TokenClass.FIELD, TokenClass.TEXT})
        if state is DecodeState.IN_FREE_TEXT:
            return frozenset({TokenClass.TEXT, TokenClass.TYPE_END})
        cls = _STATE_CLASS.get(state)
        if cls is None:  # pragma: no cover - every non-terminal state is mapped
            raise ActionParseError(f"no continuation defined for {state}")
        return frozenset({cls})

    def allowed_tokens(self, state: DecodeState) -> tuple[str, ...]:
        """Concrete action tokens allowed next (excludes free TEXT)."""
        out: list[str] = []
        for cls in self.allowed_classes(state):
            if cls is TokenClass.TEXT:
                continue
            out.extend(self.vocab.of_class(cls))
        return tuple(out)

    def step(self, state: DecodeState, atom: str, chain: list[DecodeState] | None = None) -> tuple[DecodeState, list[DecodeState]]:
        """Advance one atom.  ``chain`` holds the remaining argument states."""
        chain = list(chain or [])
        allowed = self.allowed_classes(state)

        if atom not in self.vocab:
            # Any non-special atom is free-compose text.
            if TokenClass.TEXT not in allowed:
                raise ActionParseError(f"unexpected text {atom!r} in state {state.value}")
            return DecodeState.IN_FREE_TEXT, chain

        cls = self.vocab.class_of(atom)
        if cls not in allowed:
            raise ActionParseError(
                f"{atom} ({cls.value}) is not legal in state {state.value}; "
                f"expected one of {sorted(c.value for c in allowed)}"
            )

        if cls is TokenClass.ACTION:
            action_type = self.vocab.action_type_of(atom)
            chain = list(_ARG_CHAIN[action_type])
        elif cls is TokenClass.FIELD:
            # Field-fill is a single token, then the span must close.
            chain = []
            return DecodeState.IN_FREE_TEXT, chain
        elif cls is TokenClass.TYPE_END:
            chain = []

        if chain:
            return chain.pop(0), chain
        return DecodeState.COMPLETE, chain

    def is_complete(self, state: DecodeState) -> bool:
        return state is DecodeState.COMPLETE

    def max_control_atoms(self) -> int:
        """Longest non-TYPE action, in atoms -- the forced decode budget."""
        return 1 + max(len(chain) for t, chain in _ARG_CHAIN.items() if t is not ActionType.TYPE)


# --------------------------------------------------------------------------
# Codec
# --------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"<[A-Z][A-Z0-9_+-]*>")


class ActionCodec:
    """Action <-> tokens, and grid coordinates <-> screen pixels."""

    def __init__(self, config: ActionSpaceConfig | None = None) -> None:
        self.config = config or ActionSpaceConfig()
        self.vocab = ActionVocab(self.config)
        self.grammar = ActionGrammar(self.vocab)

    # -- geometry ---------------------------------------------------------
    def to_pixels(self, action: Action, screen_w: int, screen_h: int) -> tuple[int, int]:
        """Grid cell + sub-cell offset -> the pixel at that sub-cell's centre."""
        if not action.type.is_absolute_pointer:
            raise ActionParseError(f"{action.type.value} has no absolute coordinates")
        return self.grid_to_pixels(action.row, action.col, action.offset, screen_w, screen_h)

    def grid_to_pixels(self, row: int, col: int, offset: int, screen_w: int, screen_h: int) -> tuple[int, int]:
        cfg = self.config
        g = cfg.offset_grid
        _check_range("row", row, cfg.grid_rows)
        _check_range("col", col, cfg.grid_cols)
        _check_range("offset", offset, g * g)
        cell_w = screen_w / cfg.grid_cols
        cell_h = screen_h / cfg.grid_rows
        sub_row, sub_col = divmod(offset, g)
        x = (col + (sub_col + 0.5) / g) * cell_w
        y = (row + (sub_row + 0.5) / g) * cell_h
        # Clamp so rounding at the far edge cannot land one pixel off-screen.
        return (
            min(int(round(x)), screen_w - 1),
            min(int(round(y)), screen_h - 1),
        )

    def pixels_to_grid(self, x: float, y: float, screen_w: int, screen_h: int) -> tuple[int, int, int]:
        """Inverse of :meth:`grid_to_pixels`, used to label captured clicks."""
        cfg = self.config
        g = cfg.offset_grid
        u = _clamp(x / (screen_w / cfg.grid_cols), 0.0, cfg.grid_cols - 1e-9)
        v = _clamp(y / (screen_h / cfg.grid_rows), 0.0, cfg.grid_rows - 1e-9)
        col, frac_x = int(u), u - int(u)
        row, frac_y = int(v), v - int(v)
        sub_col = min(int(frac_x * g), g - 1)
        sub_row = min(int(frac_y * g), g - 1)
        return row, col, sub_row * g + sub_col

    def quantization_error_px(self, screen_w: int, screen_h: int) -> tuple[float, float]:
        """Worst-case click error introduced by the grid, in pixels."""
        cfg = self.config
        return (
            screen_w / (cfg.grid_cols * cfg.offset_grid) / 2,
            screen_h / (cfg.grid_rows * cfg.offset_grid) / 2,
        )

    # -- value bucketing --------------------------------------------------
    def bucket_delta(self, value: float) -> int:
        return min(self.config.delta_buckets, key=lambda b: (abs(b - value), abs(b)))

    def bucket_scroll(self, value: float) -> int:
        r = self.config.scroll_range
        return int(_clamp(round(value), -r, r))

    def bucket_wait(self, ms: float) -> int:
        return min(self.config.wait_ms, key=lambda w: abs(w - ms))

    # -- serialisation ----------------------------------------------------
    def encode(self, action: Action, form_fields: Sequence[str] | None = None) -> list[str]:
        """Action -> list of atoms."""
        v = self.vocab
        t = action.type
        atoms = [v.action_token(t)]

        if t.is_absolute_pointer:
            atoms += [f"<ROW_{action.row}>", f"<COL_{action.col}>", f"<OFFSET_{action.offset}>"]
        elif t is ActionType.MOVE_REL:
            atoms += [f"<DX_{_fmt_signed(action.dx)}>", f"<DY_{_fmt_signed(action.dy)}>"]
        elif t is ActionType.SCROLL:
            atoms += [
                f"<SCROLL_DX_{_fmt_signed(action.scroll_dx)}>",
                f"<SCROLL_DY_{_fmt_signed(action.scroll_dy)}>",
            ]
        elif t is ActionType.KEY:
            atoms.append(f"<KEY_{action.key}>")
        elif t is ActionType.WAIT:
            atoms.append(_wait_token(action.wait_ms))
        elif t is ActionType.TYPE:
            if action.field is not None:
                fields = list(form_fields or [])
                if action.field not in fields:
                    raise ActionParseError(
                        f"field-fill target {action.field!r} not in form_fields {fields}"
                    )
                idx = fields.index(action.field)
                if idx >= self.config.max_form_fields:
                    raise ActionParseError(
                        f"form field index {idx} exceeds max_form_fields "
                        f"{self.config.max_form_fields}"
                    )
                atoms.append(f"<FIELD_{idx}>")
            else:
                atoms.append(action.text)
            atoms.append("<TYPE_END>")

        for atom in atoms:
            if atom in self.vocab or t is ActionType.TYPE:
                continue
            raise ActionParseError(f"encoder produced unknown token {atom!r}")
        return atoms

    def decode(self, atoms: Sequence[str], form_fields: Sequence[str] | None = None) -> Action:
        """Exactly one action's worth of atoms -> :class:`Action`."""
        actions = list(self.decode_stream(atoms, form_fields))
        if len(actions) != 1:
            raise ActionParseError(f"expected exactly one action, parsed {len(actions)}")
        return actions[0]

    def decode_stream(
        self, atoms: Iterable[str], form_fields: Sequence[str] | None = None
    ) -> Iterator[Action]:
        """Parse a run of atoms into consecutive actions."""
        fields = list(form_fields or [])
        state = self.grammar.start()
        chain: list[DecodeState] = []
        buf: list[str] = []

        for atom in atoms:
            buf.append(atom)
            state, chain = self.grammar.step(state, atom, chain)
            if state is DecodeState.IN_FREE_TEXT:
                continue  # keep consuming until <TYPE_END>
            if self.grammar.is_complete(state):
                yield self._build(buf, fields)
                buf, chain = [], []
                state = self.grammar.start()

        if buf:
            raise ActionParseError(
                f"trailing incomplete action: {' '.join(map(str, buf))} (state {state.value})"
            )

    def _build(self, atoms: list[str], fields: list[str]) -> Action:
        head = atoms[0]
        atype = self.vocab.action_type_of(head)
        args = atoms[1:]

        if atype.is_absolute_pointer:
            return Action(
                type=atype,
                row=_num(args[0], "ROW_"),
                col=_num(args[1], "COL_"),
                offset=_num(args[2], "OFFSET_"),
            )
        if atype is ActionType.MOVE_REL:
            return Action(type=atype, dx=_num(args[0], "DX_"), dy=_num(args[1], "DY_"))
        if atype is ActionType.SCROLL:
            return Action(
                type=atype,
                scroll_dx=_num(args[0], "SCROLL_DX_"),
                scroll_dy=_num(args[1], "SCROLL_DY_"),
            )
        if atype is ActionType.KEY:
            return Action(type=atype, key=args[0][len("<KEY_") : -1])
        if atype is ActionType.WAIT:
            return Action(type=atype, wait_ms=_parse_wait(args[0]))
        if atype is ActionType.TYPE:
            body = args[:-1]  # drop <TYPE_END>
            if len(body) == 1 and body[0] in self.vocab and body[0].startswith("<FIELD_"):
                idx = _num(body[0], "FIELD_")
                if idx >= len(fields):
                    raise ActionParseError(
                        f"<FIELD_{idx}> selected but only {len(fields)} form fields supplied"
                    )
                return Action(type=atype, field=fields[idx])
            return Action(type=atype, text="".join(body))
        return Action(type=atype)

    # -- convenience ------------------------------------------------------
    def click(self, x: float, y: float, screen_w: int, screen_h: int,
              type: ActionType = ActionType.MOVE_CLICK) -> Action:
        """Build an absolute-pointer action from raw pixels."""
        row, col, offset = self.pixels_to_grid(x, y, screen_w, screen_h)
        return Action(type=type, row=row, col=col, offset=offset)

    def render(self, atoms: Sequence[str]) -> str:
        return " ".join(a if a in self.vocab else repr(a) for a in atoms)


def _num(token: str, prefix: str) -> int:
    if not token.startswith(f"<{prefix}") or not token.endswith(">"):
        raise ActionParseError(f"expected <{prefix}N>, got {token!r}")
    return int(token[len(prefix) + 1 : -1])


def _parse_wait(token: str) -> int:
    if token.endswith("MS>"):
        return int(token[len("<WAIT_") : -3])
    if token.endswith("S>"):
        return int(token[len("<WAIT_") : -2]) * 1000
    raise ActionParseError(f"unparseable wait token {token!r}")


def _check_range(name: str, value: int, limit: int) -> None:
    if not isinstance(value, int) or not 0 <= value < limit:
        raise ActionParseError(f"{name}={value} out of range [0, {limit})")


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
