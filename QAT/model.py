import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Fixed sizes
# ----------------------------------------------------------------------------

READ_DIM = 128
VOCAB = 128
DEFAULT_CARRY_DIM = 896 # 128 + 896 = 1024, for A500
DEFAULT_NUM_FF = 2

# Logit scaling for read @ head; preserved from the original kernel model.
LOGIT_SCALE = 1.0 / 16.0


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


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------

class BRNN(nn.Module):
    """
    QAT version of the binary RNN.

    Latent (continuous) parameters are held in `*_lat` tensors.
    Forward passes always use the quantised view:
        +-1 weights via ste_sign
        integer thresholds via ste_round

    After training, call `export_quantized()` to get discrete int8 / int32
    tensors in the layout used by the original ReferenceBRNN buffers.
    """

    def __init__(
        self,
        num_ff: int = DEFAULT_NUM_FF,
        carry_dim: int = DEFAULT_CARRY_DIM,
    ):
        super().__init__()
        if num_ff < 0:
            raise ValueError("num_ff must be >= 0")
        if carry_dim < 0:
            raise ValueError("carry_dim must be >= 0")

        self.num_ff    = int(num_ff)
        self.carry_dim = int(carry_dim)
        self.read_dim  = READ_DIM
        self.d_model   = self.carry_dim + self.read_dim

        # Sign() is scale-invariant in the forward, so we divide the
        # pre-activation by sqrt(d_model) before sign() purely so the hardtanh
        # STE's gate width matches the typical pre-activation scale.
        self.act_ste_scale = math.sqrt(self.d_model)

        # Latent (real-valued) sign weights, init ~ Uniform(-1, 1).
        self.initial_lat = nn.Parameter(torch.empty(self.d_model).uniform_(-1, 1))
        self.embed_lat   = nn.Parameter(torch.empty(VOCAB, self.read_dim).uniform_(-1, 1))
        self.ff_lat      = nn.Parameter(
            torch.empty(self.num_ff, self.d_model, self.d_model).uniform_(-1, 1)
        )
        self.head_lat    = nn.Parameter(torch.empty(self.read_dim, VOCAB).uniform_(-1, 1))

        # Latent thresholds, rounded in forward via STE.
        self.ff_thresh_lat = nn.Parameter(torch.zeros(self.num_ff, self.d_model))

    # ------------------------------------------------------------------
    # Quantised views (with STE for backward)
    # ------------------------------------------------------------------

    def q_initial(self): return ste_sign(self.initial_lat)
    def q_embed(self):   return ste_sign(self.embed_lat)
    def q_ff(self):      return ste_sign(self.ff_lat)
    def q_head(self):    return ste_sign(self.head_lat)
    def q_thresh(self):  return ste_round(self.ff_thresh_lat)

    @torch.no_grad()
    def clip_latents_(self):
        """Standard BinaryConnect-style clip on sign latents."""
        self.initial_lat.data.clamp_(-1, 1)
        self.embed_lat.data.clamp_(-1, 1)
        self.ff_lat.data.clamp_(-1, 1)
        self.head_lat.data.clamp_(-1, 1)
        # ff_thresh_lat is *not* clipped: thresholds may be any integer.

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def _stack(self, x, ff, thr, head):
        """
        x:   [B, d_model]   +-1 floats
        ff:  [num_ff, d_model, d_model]   +-1 floats
        thr: [num_ff, d_model]            int-valued floats
        head:[read_dim, VOCAB]            +-1 floats
        """
        for i in range(self.num_ff):
            preact = x @ ff[i]
            x = ste_sign((preact - thr[i]) / self.act_ste_scale)
        carry = x[:, :self.carry_dim]
        read  = x[:, self.carry_dim:]
        logits = (read @ head) * LOGIT_SCALE
        return logits, carry

    def forward(self, tokens):
        """
        tokens: [B, T] long. Returns mean cross-entropy across (B, T) in nats.
        """
        if tokens.ndim == 1:
            tokens = tokens.view(1, -1)
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [B, T] or [T]")

        B, T = tokens.shape

        initial = self.q_initial()
        embed   = self.q_embed()
        ff      = self.q_ff()
        thresh  = self.q_thresh()
        head    = self.q_head()

        x = initial.unsqueeze(0).expand(B, self.d_model).contiguous()

        total = x.new_zeros(())
        for t in range(T):
            logits, carry = self._stack(x, ff, thresh, head)
            total = total + F.cross_entropy(logits, tokens[:, t])
            x = torch.cat([carry, embed[tokens[:, t]]], dim=1)

        return total / T

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens=None,
        num_tokens: int = 0,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Greedy / temperature-sampled generation under the quantised model.

        prompt_tokens: 1-D long tensor / list / None. State is primed by
                       feeding these in order before generation.
        num_tokens:    number of tokens to sample after the prompt.
        temperature:   0 -> greedy argmax. >0 -> sample from softmax(logits/T).

        Returns a 1-D long tensor of length `num_tokens` on the model's device.
        """
        if temperature < 0:
            raise ValueError("temperature must be >= 0")
        if num_tokens < 0:
            raise ValueError("num_tokens must be >= 0")

        device = self.initial_lat.device

        if prompt_tokens is None:
            prompt_tokens = torch.empty(0, dtype=torch.long, device=device)
        else:
            prompt_tokens = torch.as_tensor(
                prompt_tokens, dtype=torch.long, device=device
            ).flatten()

        # Quantise weights once for the whole rollout.
        initial = self.q_initial()
        embed   = self.q_embed()
        ff      = self.q_ff()
        thresh  = self.q_thresh()
        head    = self.q_head()

        x = initial.unsqueeze(0).contiguous()  # [1, d_model]

        # Prime with the prompt.
        for tok in prompt_tokens.tolist():
            _, carry = self._stack(x, ff, thresh, head)
            x = torch.cat([carry, embed[tok].unsqueeze(0)], dim=1)

        out = torch.empty(num_tokens, dtype=torch.long, device=device)

        for i in range(num_tokens):
            logits, carry = self._stack(x, ff, thresh, head)
            logits = logits[0]  # [VOCAB]

            if temperature == 0:
                tok = int(torch.argmax(logits).item())
            else:
                probs = torch.softmax(logits / float(temperature), dim=0)
                tok = int(torch.multinomial(probs, num_samples=1).item())

            out[i] = tok
            x = torch.cat([carry, embed[tok].unsqueeze(0)], dim=1)

        return out

    # ------------------------------------------------------------------
    # Discrete export
    # ------------------------------------------------------------------

    @torch.no_grad()
    def export_quantized(self):
        """
        Returns a dict of plain discrete tensors:
            initial   [d_model]                   int8   in {-1,+1}
            embed     [VOCAB, read_dim]           int8   in {-1,+1}
            ff        [num_ff, d_model, d_model]  int8   in {-1,+1}
            head      [read_dim, VOCAB]           int8   in {-1,+1}
            ff_thresh [num_ff, d_model]           int32  (any integer)
        plus geometry metadata.
        """
        def pm1(t):
            return torch.where(t >= 0, torch.ones_like(t), -torch.ones_like(t)).to(torch.int8)

        return {
            "initial":   pm1(self.initial_lat).cpu(),
            "embed":     pm1(self.embed_lat).cpu(),
            "ff":        pm1(self.ff_lat).cpu(),
            "head":      pm1(self.head_lat).cpu(),
            "ff_thresh": torch.round(self.ff_thresh_lat).to(torch.int32).cpu(),
            "num_ff":    self.num_ff,
            "carry_dim": self.carry_dim,
            "read_dim":  self.read_dim,
        }


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def infer_carry_dim_from_state_dict(state_dict) -> int:
    """Recover carry_dim from a saved BRNN latent state_dict."""
    if "initial_lat" in state_dict:
        return int(state_dict["initial_lat"].shape[0]) - READ_DIM
    if "ff_lat" in state_dict and state_dict["ff_lat"].ndim == 3:
        return int(state_dict["ff_lat"].shape[1]) - READ_DIM
    return DEFAULT_CARRY_DIM
