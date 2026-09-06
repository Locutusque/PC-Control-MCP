"""Configuration objects for every stage of the GUI control agent.

Everything that determines the *shape of the action vocabulary* lives in
:class:`ActionSpaceConfig`.  That object is serialised into every checkpoint,
because the new action tokens get freshly-initialised embeddings whose row
index is derived purely from the vocabulary ordering -- silently changing a
field here would remap embeddings and produce a model that clicks in the wrong
place with no error message.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _asdict(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _asdict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [_asdict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _asdict(v) for k, v in obj.items()}
    return obj


class _ConfigMixin:
    """JSON round-tripping shared by every config dataclass."""

    def to_dict(self) -> dict:
        return _asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        kwargs = {}
        types = {f.name: f for f in dataclasses.fields(cls)}
        for name, value in data.items():
            if name not in types:
                continue  # forward compatible: ignore unknown keys
            ftype = types[name].type
            # Nested config dataclasses are reconstructed recursively.
            resolved = _CONFIG_TYPES.get(ftype if isinstance(ftype, str) else getattr(ftype, "__name__", ""))
            if resolved is not None and isinstance(value, dict):
                kwargs[name] = resolved.from_dict(value)
            elif isinstance(value, list):
                kwargs[name] = tuple(tuple(v) if isinstance(v, list) else v for v in value)
            else:
                kwargs[name] = value
        return cls(**kwargs)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path):
        return cls.from_dict(json.loads(Path(path).read_text()))


# --------------------------------------------------------------------------
# Vision
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class VisionConfig(_ConfigMixin):
    """Small ViT over a resized screenshot (plan section 3.1).

    Defaults land at ~86M parameters, inside the 50-90M budget of section 3.5.
    The patch grid doubles as the coordinate vocabulary, so ``image_size`` and
    ``patch_size`` also fix the click resolution -- see
    :class:`ActionSpaceConfig`.
    """

    image_size: int = 768
    patch_size: int = 32
    dim: int = 768
    depth: int = 12
    heads: int = 12
    mlp_ratio: float = 4.0
    channels: int = 3
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.image_size % self.patch_size:
            raise ValueError(
                f"image_size {self.image_size} must be divisible by patch_size {self.patch_size}"
            )
        if self.dim % self.heads:
            raise ValueError(f"dim {self.dim} must be divisible by heads {self.heads}")

    @property
    def grid_side(self) -> int:
        return self.image_size // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.grid_side**2


# --------------------------------------------------------------------------
# Action space
# --------------------------------------------------------------------------

# Fixed OS key vocabulary (plan section 2.1).  Order is load-bearing: it fixes
# token ids.  Append new keys at the end, never insert in the middle.
DEFAULT_KEYS: tuple[str, ...] = (
    # Editing / navigation
    "ENTER", "TAB", "ESC", "BACKSPACE", "DELETE", "SPACE",
    "UP", "DOWN", "LEFT", "RIGHT", "HOME", "END", "PAGE_UP", "PAGE_DOWN",
    # Modifiers (emitted as chords, e.g. <KEY> <KEY_CTRL_C>)
    "CTRL", "ALT", "SHIFT", "META",
    # Common chords
    "CTRL_A", "CTRL_C", "CTRL_V", "CTRL_X", "CTRL_Z", "CTRL_Y",
    "CTRL_S", "CTRL_F", "CTRL_W", "CTRL_T", "CTRL_L", "CTRL_R",
    "ALT_TAB", "SHIFT_TAB",
    # Function keys
    "F1", "F2", "F3", "F4", "F5", "F6",
    "F7", "F8", "F9", "F10", "F11", "F12",
)

# Symmetric, roughly log-spaced buckets for relative cursor motion.  Section 8
# of the plan calls out that absolute grounding alone cannot express "drag this
# slider a bit further"; MOVE_REL/DRAG use these.
DEFAULT_DELTA_BUCKETS: tuple[int, ...] = (
    -256, -128, -64, -32, -16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16, 32, 64, 128, 256,
)

DEFAULT_WAIT_MS: tuple[int, ...] = (100, 250, 500, 1000, 2000)


@dataclass(frozen=True)
class ActionSpaceConfig(_ConfigMixin):
    """Discrete action vocabulary (plan section 2.1).

    ``grid_rows``/``grid_cols`` must match the ViT patch grid so that a
    ``<ROW_r> <COL_c>`` pair addresses exactly one patch embedding; use
    :meth:`for_vision` to derive them.

    ``offset_grid`` subdivides each patch into ``offset_grid**2`` sub-cells for
    the ``<OFFSET_k>`` refinement token.  Click precision on a screen of width
    ``W`` is ``W / (grid_cols * offset_grid)`` pixels.  With the defaults that
    is 1920 / (24 * 8) = 10px, which is tight enough to hit a small toolbar
    icon; dropping ``offset_grid`` to 3 costs only 55 vocabulary entries but
    blows the error radius out to ~27px, which is wider than many UI targets.
    """

    grid_rows: int = 24
    grid_cols: int = 24
    offset_grid: int = 8
    scroll_range: int = 5
    max_form_fields: int = 16
    max_type_tokens: int = 96
    keys: tuple[str, ...] = DEFAULT_KEYS
    delta_buckets: tuple[int, ...] = DEFAULT_DELTA_BUCKETS
    wait_ms: tuple[int, ...] = DEFAULT_WAIT_MS

    def __post_init__(self) -> None:
        for name in ("grid_rows", "grid_cols", "offset_grid", "max_form_fields"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        if self.scroll_range < 1:
            raise ValueError("scroll_range must be >= 1")
        if len(set(self.keys)) != len(self.keys):
            raise ValueError("duplicate entries in keys")
        if 0 not in self.delta_buckets:
            raise ValueError("delta_buckets must contain 0 (a no-op on one axis)")
        if len(set(self.delta_buckets)) != len(self.delta_buckets):
            raise ValueError("duplicate entries in delta_buckets")

    @classmethod
    def for_vision(cls, vision: VisionConfig, **overrides) -> ActionSpaceConfig:
        """Build an action space whose coordinate grid matches ``vision``."""
        return cls(grid_rows=vision.grid_side, grid_cols=vision.grid_side, **overrides)

    def matches_vision(self, vision: VisionConfig) -> bool:
        return self.grid_rows == vision.grid_side and self.grid_cols == vision.grid_side


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ProjectorConfig(_ConfigMixin):
    """Vision -> LM bridge (plan section 3.1)."""

    kind: str = "mlp"  # "mlp" | "perceiver"
    hidden_dim: int = 2048
    num_latents: int = 144  # perceiver only: compresses 576 patches -> 144 tokens
    num_heads: int = 8
    depth: int = 2
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in ("mlp", "perceiver"):
            raise ValueError(f"unknown projector kind {self.kind!r}")


@dataclass(frozen=True)
class DecoderConfig(_ConfigMixin):
    """Pretrained LM decoder + LoRA (plan section 3.2).

    ``base_model`` should be settled by benchmarking quantised latency on the
    target hardware (plan section 8), not by parameter count on paper -- see
    ``train/benchmark_latency.py``.
    """

    base_model: str = "HuggingFaceTB/SmolLM2-360M-Instruct"
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    )
    # Base weights stay frozen; only LoRA, the projector and the freshly added
    # action-token embeddings train.  Full fine-tuning on a narrow GUI corpus
    # overwrites the language priors we brought the LM in for (plan 3.2).
    freeze_base: bool = True
    train_new_embeddings: bool = True
    torch_dtype: str = "bfloat16"
    attn_implementation: str | None = None
    load_in_4bit: bool = False
    load_in_8bit: bool = False


@dataclass(frozen=True)
class PolicyConfig(_ConfigMixin):
    vision: VisionConfig = field(default_factory=VisionConfig)
    action_space: ActionSpaceConfig = field(default_factory=ActionSpaceConfig)
    projector: ProjectorConfig = field(default_factory=ProjectorConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    # How many past actions are replayed into the context each tick.
    max_history: int = 8
    max_instruction_tokens: int = 128
    max_form_data_tokens: int = 256

    def __post_init__(self) -> None:
        if not self.action_space.matches_vision(self.vision):
            raise ValueError(
                "action_space coordinate grid "
                f"({self.action_space.grid_rows}x{self.action_space.grid_cols}) does not "
                f"match the ViT patch grid ({self.vision.grid_side}x{self.vision.grid_side}); "
                "use ActionSpaceConfig.for_vision(vision) to keep them aligned"
            )


# --------------------------------------------------------------------------
# Capture
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PrivacyConfig(_ConfigMixin):
    """Section 4.1.3 -- these defaults are deliberately conservative.

    The daemon records everything on screen, so the safety layer lives in the
    hot path rather than in a post-hoc filter somebody might forget to run.
    """

    # Recording pauses entirely while one of these owns the foreground window.
    app_blocklist: tuple[str, ...] = (
        "1password", "bitwarden", "keepass", "lastpass", "dashlane", "keychain access",
        "gnome-keyring", "seahorse", "authy", "gpg", "ssh-agent",
    )
    domain_blocklist: tuple[str, ...] = (
        "bank", "chase.com", "wellsfargo.com", "paypal.com", "stripe.com",
        "healthcare.gov", "mychart", "irs.gov", "vault", "accounts.google.com/signin",
    )
    # Window titles matching these substrings pause capture too.
    title_blocklist: tuple[str, ...] = (
        "incognito", "private browsing", "inprivate", "password", "sign in", "log in",
    )
    # Never record characters typed into a field the accessibility tree marks
    # secure.  We keep only a count so the trajectory stays structurally intact.
    redact_password_fields: bool = True
    # Async OCR+regex sweep that must pass before raw segments are promoted
    # into the training pool.
    ocr_redaction: bool = True
    redaction_patterns: tuple[tuple[str, str], ...] = (
        ("credit_card", r"\b(?:\d[ -]*?){13,16}\b"),
        ("ssn", r"\b\d{3}-\d{2}-\d{4}\b"),
        ("us_phone", r"\b\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}\b"),
        ("email", r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
        ("iban", r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b"),
        ("api_key", r"\b(?:sk|pk|ghp|gho|xox[abps])[-_][A-Za-z0-9]{16,}\b"),
    )
    # Encrypt segments at rest; local-only by default (plan 4.1.3).
    encrypt_at_rest: bool = True
    allow_network_upload: bool = False
    # Delete raw segments this many days after they are processed into
    # training examples, so the sensitive archive does not grow without bound.
    raw_retention_days: int = 14


@dataclass(frozen=True)
class CaptureConfig(_ConfigMixin):
    """Background recording daemon (plan section 4.1)."""

    root: str = "raw_capture"
    pool_root: str = "data_pool"
    # Matches the target control-tick rate so frame/action alignment downstream
    # is a simple index, not an interpolation problem.
    fps: int = 15
    # Pause after this many seconds without input; idle screen time is pure
    # storage waste for this dataset.
    idle_timeout_s: float = 8.0
    # Cut a trajectory when input stops for this long, or on an app switch.
    segment_idle_gap_s: float = 4.0
    segment_max_duration_s: float = 300.0
    segment_on_app_switch: bool = True
    # Downscale before encoding: the model only ever sees image_size anyway.
    capture_scale: float = 0.5
    video_codec: str = "ffv1"  # "ffv1" (lossless) | "libx264" (near-lossless)
    video_crf: int = 18
    # Always show a tray indicator while recording (plan 4.1.1) -- this is not
    # meant to be a silent process.
    tray_indicator: bool = True
    pause_hotkey: str = "<ctrl>+<alt>+p"
    label_hotkey: str = "<ctrl>+<alt>+l"
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SafetyConfig(_ConfigMixin):
    """Guardrails at the dispatch layer (plan section 6.4).

    The orchestrating LLM never reviews individual actions, so these checks are
    the only thing standing between a mis-generalising policy and the real
    filesystem.  They are enforced on the way *out* of the model, regardless of
    what it predicted.
    """

    # Actions may only land inside this rect (screen pixels).  None = full screen.
    allowed_region: tuple[int, int, int, int] | None = None
    # Refuse to act while one of these owns the foreground window.
    blocked_apps: tuple[str, ...] = (
        "terminal", "iterm", "cmd.exe", "powershell", "windows terminal", "konsole",
        "gnome-terminal", "xterm", "system preferences", "system settings",
        "control panel", "regedit", "disk utility", "keychain access",
    )
    # Key chords that are never dispatched, whatever the policy emits.
    blocked_keys: tuple[str, ...] = (
        "CTRL_W",      # closes the tab the harness is driving
        "ALT_TAB",     # escapes the scoped window
        "META",        # opens the OS launcher
        "F11",         # fullscreen toggles break coordinate grounding
    )
    # Typed text matching these is dropped before it reaches the OS.
    blocked_text_patterns: tuple[str, ...] = (
        r"\brm\s+-rf\b",
        r"\bdel\s+/[sqf]\b",
        r"\bformat\s+[a-z]:",
        r"\bsudo\b",
        r"\bmkfs\b",
        r"\bdd\s+if=",
        r":\s*\(\)\s*\{.*\}\s*;\s*:",  # fork bomb
    )
    max_actions_per_second: float = 30.0
    max_actions_per_subtask: int = 600
    # Log every dispatched action for post-hoc review, which matters most
    # during stage-3 rollout collection.
    audit_log: str | None = "runs/dispatch_audit.jsonl"
    # Refuse to run at all unless the process looks sandboxed (VM, scoped
    # browser profile, or an explicit opt-out).
    require_sandbox: bool = True
    dry_run: bool = False


@dataclass(frozen=True)
class HarnessConfig(_ConfigMixin):
    """Closed-loop control harness (plan section 6.2)."""

    target_hz: float = 15.0
    default_timeout_s: float = 60.0
    # Escalate after this many consecutive frames with no visual change.
    stuck_frames: int = 12
    # Perceptual-hash distance below which two frames count as "unchanged".
    frame_hash_threshold: int = 2
    max_history: int = 8
    # Ask the verifier whether success_criteria is met before returning "done".
    verify_success: bool = True
    safety: SafetyConfig = field(default_factory=SafetyConfig)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainConfig(_ConfigMixin):
    stage: str = "stage1"
    output_dir: str = "runs/stage1"
    data_root: str = "data_pool"
    epochs: int = 1
    max_steps: int = -1
    batch_size: int = 8
    grad_accum: int = 4
    lr: float = 2e-4
    # New action-token embeddings start from noise and need a hotter LR than
    # the LoRA adapters sitting on top of pretrained weights.
    embedding_lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    seed: int = 0
    num_workers: int = 4
    log_every: int = 20
    eval_every: int = 500
    save_every: int = 1000
    # Extra CE weight on field-fill target spans: copying a supplied value
    # exactly is a correctness requirement, not a quality one (plan 2.3).
    copy_loss_weight: float = 4.0
    # Stage 3 only: weight on the ESCALATE calibration term.
    escalate_loss_weight: float = 1.0
    resume_from: str | None = None


_CONFIG_TYPES: dict[str, type] = {
    "VisionConfig": VisionConfig,
    "ActionSpaceConfig": ActionSpaceConfig,
    "ProjectorConfig": ProjectorConfig,
    "DecoderConfig": DecoderConfig,
    "PolicyConfig": PolicyConfig,
    "PrivacyConfig": PrivacyConfig,
    "CaptureConfig": CaptureConfig,
    "SafetyConfig": SafetyConfig,
    "HarnessConfig": HarnessConfig,
    "TrainConfig": TrainConfig,
}
