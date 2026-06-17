import math

import torch
import torch.nn as nn
import torch.nn.functional as F


MODEL_TYPES = ("rnn", "lstm")
NORM_MODES = ("none", "mean", "layer")


# ----------------------------------------------------------------------------
# Straight-through estimators
# ----------------------------------------------------------------------------

class _STESign(torch.autograd.Function):
    """Forward: +-1 (zero -> +1). Backward: hardtanh STE on the latent."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output * (x.abs() <= 1).to(grad_output.dtype)


def ste_sign(x):
    return _STESign.apply(x)


def ste_step(x):
    """Binary {0, 1} step with STE: step(x) = (sign(x) + 1) / 2."""
    return (ste_sign(x) + 1.0) * 0.5


class _STERound(torch.autograd.Function):
    """Forward: round to nearest int. Backward: identity."""

    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def ste_round(x):
    return _STERound.apply(x)


def to_pm1_int8(t):
    """Map any tensor to int8 {-1, +1} (zero -> +1)."""
    return torch.where(t >= 0, torch.ones_like(t), -torch.ones_like(t)).to(torch.int8)


# ----------------------------------------------------------------------------
# Pre-activation normalization (the "adaptive threshold")
# ----------------------------------------------------------------------------

def normalize_preact(z, mode, scale, bias=None, eps=1e-5):
    """
    Normalize an integer-valued popcount pre-activation z: [..., d] before sign().

    mode:
      "none"  -> z / scale                  (fixed threshold at 0)
      "mean"  -> (z - mean(z)) / scale       (adaptive threshold = mean of the
                                              popcounts; only the mean is needed
                                              at inference -> XNOR friendly)
      "layer" -> (z - mean(z)) / std(z)      (layer norm; the /std is a training-
                                              time gradient scaler that vanishes
                                              under sign() at inference when
                                              bias is None)

    bias: optional integer threshold, subtracted in the normalized units. Off by
    default; keeping it None means the inference fold is a pure threshold needing
    only the mean (mean/none) and never the std.
    """
    if mode == "layer":
        mu = z.mean(dim=-1, keepdim=True)
        var = z.var(dim=-1, unbiased=False, keepdim=True)
        out = (z - mu) / torch.sqrt(var + eps)
    elif mode == "mean":
        mu = z.mean(dim=-1, keepdim=True)
        out = (z - mu) / scale
    elif mode == "none":
        out = z / scale
    else:
        raise ValueError(f"unknown norm mode {mode!r}")

    if bias is not None:
        out = out - bias
    return out


# ----------------------------------------------------------------------------
# Binary vanilla RNN  (carry/read state design)
# ----------------------------------------------------------------------------

class BRNN(nn.Module):
    """
    Quantization-aware binary RNN.

    State layout (this is the original 'carry/read' design):
        state = [ carry | read ]          dim = carry_dim + embed_dim
    Each step, the whole state is pushed through `num_ff` square +-1 layers with
    sign activations. The 'read' tail of the result produces the logits and is
    then overwritten by the next token's embedding; the 'carry' head is the
    recurrent memory.

    All matmuls are +-1 x +-1 (XNOR-popcount). Latent real-valued parameters live
    in `*_lat`; forward passes use their quantized (sign / round) views.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        embed_dim: int = 128,
        carry_dim: int = 896,
        num_ff: int = 2,
        use_thresholds: bool = False,
        norm_mode: str = "mean",
    ):
        super().__init__()
        if vocab_size < 1:
            raise ValueError("vocab_size must be >= 1")
        if embed_dim < 1:
            raise ValueError("embed_dim must be >= 1")
        if carry_dim < 0:
            raise ValueError("carry_dim must be >= 0")
        if num_ff < 1:
            raise ValueError("num_ff must be >= 1")
        if norm_mode not in NORM_MODES:
            raise ValueError(f"norm_mode must be one of {NORM_MODES}")

        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)
        self.carry_dim = int(carry_dim)
        self.num_ff = int(num_ff)
        self.use_thresholds = bool(use_thresholds)
        self.norm_mode = norm_mode

        self.state_dim = self.carry_dim + self.embed_dim
        self.act_ste_scale = math.sqrt(self.state_dim)
        self.logit_scale = 1.0 / math.sqrt(self.embed_dim)

        self.initial_lat = nn.Parameter(torch.empty(self.state_dim).uniform_(-1, 1))
        self.embed_lat = nn.Parameter(torch.empty(self.vocab_size, self.embed_dim).uniform_(-1, 1))
        self.ff_lat = nn.Parameter(
            torch.empty(self.num_ff, self.state_dim, self.state_dim).uniform_(-1, 1)
        )
        self.head_lat = nn.Parameter(torch.empty(self.embed_dim, self.vocab_size).uniform_(-1, 1))

        if self.use_thresholds:
            self.ff_thresh_lat = nn.Parameter(torch.zeros(self.num_ff, self.state_dim))
        else:
            self.register_parameter("ff_thresh_lat", None)

    @property
    def config(self):
        return {
            "model_type": "rnn",
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "carry_dim": self.carry_dim,
            "num_ff": self.num_ff,
            "use_thresholds": self.use_thresholds,
            "norm_mode": self.norm_mode,
        }

    # ---- quantized views ----
    def q_initial(self): return ste_sign(self.initial_lat)
    def q_embed(self):   return ste_sign(self.embed_lat)
    def q_ff(self):      return ste_sign(self.ff_lat)
    def q_head(self):    return ste_sign(self.head_lat)
    def q_thresh(self):  return ste_round(self.ff_thresh_lat) if self.use_thresholds else None

    # ---- parameter groupings ----
    def sign_parameters(self):
        return [self.initial_lat, self.embed_lat, self.ff_lat, self.head_lat]

    def int_parameters(self):
        return [self.ff_thresh_lat] if self.use_thresholds else []

    def aux_parameters(self):
        return []

    @torch.no_grad()
    def clip_latents_(self):
        for p in self.sign_parameters():
            p.data.clamp_(-1, 1)

    # ---- core ----
    def _step(self, state, ff, thresh, head):
        for i in range(self.num_ff):
            pre = state @ ff[i]                                  # XNOR-popcount
            bias = thresh[i] if thresh is not None else None
            state = ste_sign(normalize_preact(pre, self.norm_mode, self.act_ste_scale, bias))
        carry = state[:, :self.carry_dim]
        read = state[:, self.carry_dim:]
        logits = (read @ head) * self.logit_scale               # XNOR-popcount
        return logits, carry

    def forward(self, tokens):
        if tokens.ndim == 1:
            tokens = tokens.view(1, -1)
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [B, T] or [T]")

        B, T = tokens.shape
        initial, embed = self.q_initial(), self.q_embed()
        ff, thresh, head = self.q_ff(), self.q_thresh(), self.q_head()

        state = initial.unsqueeze(0).expand(B, self.state_dim).contiguous()
        total = state.new_zeros(())
        for t in range(T):
            logits, carry = self._step(state, ff, thresh, head)
            total = total + F.cross_entropy(logits, tokens[:, t])
            state = torch.cat([carry, embed[tokens[:, t]]], dim=1)
        return total / T

    @torch.no_grad()
    def generate(self, prompt_tokens=None, num_tokens=0, temperature=1.0):
        device = self.initial_lat.device
        prompt_tokens = _as_prompt(prompt_tokens, device)
        initial, embed = self.q_initial(), self.q_embed()
        ff, thresh, head = self.q_ff(), self.q_thresh(), self.q_head()

        state = initial.unsqueeze(0).contiguous()
        for tok in prompt_tokens.tolist():
            _, carry = self._step(state, ff, thresh, head)
            state = torch.cat([carry, embed[tok].unsqueeze(0)], dim=1)

        out = torch.empty(num_tokens, dtype=torch.long, device=device)
        for i in range(num_tokens):
            logits, carry = self._step(state, ff, thresh, head)
            tok = _sample(logits[0], temperature)
            out[i] = tok
            state = torch.cat([carry, embed[tok].unsqueeze(0)], dim=1)
        return out

    @torch.no_grad()
    def export_quantized(self):
        out = {
            "config": self.config,
            "initial": to_pm1_int8(self.initial_lat).cpu(),
            "embed": to_pm1_int8(self.embed_lat).cpu(),
            "ff": to_pm1_int8(self.ff_lat).cpu(),
            "head": to_pm1_int8(self.head_lat).cpu(),
        }
        if self.use_thresholds:
            out["ff_thresh"] = torch.round(self.ff_thresh_lat).to(torch.int32).cpu()
        return out


# ----------------------------------------------------------------------------
# Binary LSTM  (standard topology)
# ----------------------------------------------------------------------------

class BLSTM(nn.Module):
    """
    Quantization-aware binary LSTM, standard topology.

    Hidden state h (+-1) and an integer cell c. The token embedding enters only
    through the gate matrices; the head reads the full hidden state.

        gate_input  = concat(h_prev, embed(token))     dim = hidden_dim + embed_dim
        z_k         = W_k @ gate_input                  (XNOR-popcount), k in {i,f,o,g}
        i,f,o       = step(norm(z_i,z_f,z_o))           {0,1}
        g           = sign(norm(z_g))                   {-1,+1}
        c           = f * c + i * g                     integer accumulator
        h           = o * sign(c)                       +-1
        logits      = (h @ head) * logit_scale          (XNOR-popcount)

    Every matmul is +-1 x +-1; all per-step extras (mean threshold, gate masks,
    cell accumulate) are O(hidden_dim) elementwise.
    """

    NUM_GATES = 4  # 0=input, 1=forget, 2=output, 3=candidate

    def __init__(
        self,
        *,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 1024,
        use_thresholds: bool = False,
        norm_mode: str = "mean",
    ):
        super().__init__()
        if vocab_size < 1:
            raise ValueError("vocab_size must be >= 1")
        if embed_dim < 1:
            raise ValueError("embed_dim must be >= 1")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1")
        if norm_mode not in NORM_MODES:
            raise ValueError(f"norm_mode must be one of {NORM_MODES}")

        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.use_thresholds = bool(use_thresholds)
        self.norm_mode = norm_mode

        self.gate_input_dim = self.hidden_dim + self.embed_dim
        self.act_ste_scale = math.sqrt(self.gate_input_dim)
        self.logit_scale = 1.0 / math.sqrt(self.hidden_dim)

        self.initial_h_lat = nn.Parameter(torch.empty(self.hidden_dim).uniform_(-1, 1))
        self.embed_lat = nn.Parameter(torch.empty(self.vocab_size, self.embed_dim).uniform_(-1, 1))
        self.gate_lat = nn.Parameter(
            torch.empty(self.NUM_GATES, self.gate_input_dim, self.hidden_dim).uniform_(-1, 1)
        )
        self.head_lat = nn.Parameter(torch.empty(self.hidden_dim, self.vocab_size).uniform_(-1, 1))

        # Training-only: widens the sign(c) STE window so gradient flows through
        # the long-term cell. Positive scale -> forward sign unchanged.
        self.cell_log_scale = nn.Parameter(torch.tensor(math.log(4.0)))

        if self.use_thresholds:
            self.gate_bias_lat = nn.Parameter(torch.zeros(self.NUM_GATES, self.hidden_dim))
        else:
            self.register_parameter("gate_bias_lat", None)

    @property
    def config(self):
        return {
            "model_type": "lstm",
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "use_thresholds": self.use_thresholds,
            "norm_mode": self.norm_mode,
        }

    # ---- quantized views ----
    def q_initial_h(self): return ste_sign(self.initial_h_lat)
    def q_embed(self):     return ste_sign(self.embed_lat)
    def q_gate(self):      return ste_sign(self.gate_lat)
    def q_head(self):      return ste_sign(self.head_lat)
    def q_bias(self):      return ste_round(self.gate_bias_lat) if self.use_thresholds else None

    # ---- parameter groupings ----
    def sign_parameters(self):
        return [self.initial_h_lat, self.embed_lat, self.gate_lat, self.head_lat]

    def int_parameters(self):
        return [self.gate_bias_lat] if self.use_thresholds else []

    def aux_parameters(self):
        return [self.cell_log_scale]

    @torch.no_grad()
    def clip_latents_(self):
        for p in self.sign_parameters():
            p.data.clamp_(-1, 1)

    # ---- core ----
    def _logits(self, h, head):
        return (h @ head) * self.logit_scale                    # XNOR-popcount

    def _cell_step(self, h, c, e, gate_w, bias):
        gate_input = torch.cat([h, e], dim=1)                   # [B, gate_input_dim]
        pre = torch.einsum("bi,kid->bkd", gate_input, gate_w)   # [B, 4, hidden] XNOR

        def gate(k, fn):
            b = bias[k] if bias is not None else None
            return fn(normalize_preact(pre[:, k], self.norm_mode, self.act_ste_scale, b))

        i = gate(0, ste_step)
        f = gate(1, ste_step)
        o = gate(2, ste_step)
        g = gate(3, ste_sign)

        c_new = f * c + i * g
        h_new = o * ste_sign(c_new / torch.exp(self.cell_log_scale))
        return h_new, c_new

    def forward(self, tokens):
        if tokens.ndim == 1:
            tokens = tokens.view(1, -1)
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [B, T] or [T]")

        B, T = tokens.shape
        initial_h, embed = self.q_initial_h(), self.q_embed()
        gate_w, bias, head = self.q_gate(), self.q_bias(), self.q_head()

        h = initial_h.unsqueeze(0).expand(B, self.hidden_dim).contiguous()
        c = h.new_zeros(B, self.hidden_dim)
        total = h.new_zeros(())
        for t in range(T):
            total = total + F.cross_entropy(self._logits(h, head), tokens[:, t])
            h, c = self._cell_step(h, c, embed[tokens[:, t]], gate_w, bias)
        return total / T

    @torch.no_grad()
    def generate(self, prompt_tokens=None, num_tokens=0, temperature=1.0):
        device = self.initial_h_lat.device
        prompt_tokens = _as_prompt(prompt_tokens, device)
        initial_h, embed = self.q_initial_h(), self.q_embed()
        gate_w, bias, head = self.q_gate(), self.q_bias(), self.q_head()

        h = initial_h.unsqueeze(0).contiguous()
        c = h.new_zeros(1, self.hidden_dim)
        for tok in prompt_tokens.tolist():
            h, c = self._cell_step(h, c, embed[tok].unsqueeze(0), gate_w, bias)

        out = torch.empty(num_tokens, dtype=torch.long, device=device)
        for i in range(num_tokens):
            tok = _sample(self._logits(h, head)[0], temperature)
            out[i] = tok
            h, c = self._cell_step(h, c, embed[tok].unsqueeze(0), gate_w, bias)
        return out

    @torch.no_grad()
    def export_quantized(self):
        out = {
            "config": self.config,
            "initial_h": to_pm1_int8(self.initial_h_lat).cpu(),
            "embed": to_pm1_int8(self.embed_lat).cpu(),
            "gate": to_pm1_int8(self.gate_lat).cpu(),
            "head": to_pm1_int8(self.head_lat).cpu(),
        }
        if self.use_thresholds:
            out["gate_bias"] = torch.round(self.gate_bias_lat).to(torch.int32).cpu()
        return out


# ----------------------------------------------------------------------------
# Shared helpers / factory
# ----------------------------------------------------------------------------

def _as_prompt(prompt_tokens, device):
    if prompt_tokens is None:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.as_tensor(prompt_tokens, dtype=torch.long, device=device).flatten()


def _sample(logits, temperature):
    if temperature < 0:
        raise ValueError("temperature must be >= 0")
    if temperature == 0:
        return int(torch.argmax(logits).item())
    probs = torch.softmax(logits / float(temperature), dim=0)
    return int(torch.multinomial(probs, num_samples=1).item())


def build_model(config):
    """Construct a model from its config dict (as stored in checkpoints)."""
    cfg = dict(config)
    model_type = cfg.pop("model_type")
    if model_type == "rnn":
        return BRNN(**cfg)
    if model_type == "lstm":
        return BLSTM(**cfg)
    raise ValueError(f"Unknown model_type {model_type!r}; expected one of {MODEL_TYPES}")
