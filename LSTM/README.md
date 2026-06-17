# Binarized RNN / LSTM for retro hardware

(Note: Put this together with claude-opus-4.8, but haven't matched vanilla quality yet.)

Character-level language models trained with quantization-aware training (QAT)
so that inference reduces to **XNOR + popcount**. Every weight matrix and every
recurrent state is `±1`, so each matmul is `popcount(XNOR(state, weight))` — the
operation a 6502/68000 can do quickly. All per-step extras (an adaptive
threshold, the LSTM gate masks, the integer cell) are `O(width)` elementwise.

## Architectures

Two models share the same training/inference tooling (`--model {rnn,lstm}`):

- **`rnn`** — the original vanilla design. The state is partitioned
  `[carry | read]` of width `state_dim = carry_dim + embed_dim`. Each step the
  whole state passes through `num_ff` square `±1` layers with sign activations;
  the `read` tail produces the logits and is then overwritten by the next
  token's embedding, while `carry` is the recurrent memory.

- **`lstm`** — a standard-topology binary LSTM. Hidden state `h` is `±1`; the
  cell `c` is a small per-element **integer accumulator** (cheap adds, no
  matmul). The token embedding enters only through the gate matrices, and the
  head reads the full hidden state:

  ```
  gate_input = concat(h_prev, embed(token))     # dim = hidden_dim + embed_dim
  z_k        = W_k @ gate_input                  # XNOR-popcount, k in {i,f,o,g}
  i,f,o      = step(norm(z))                      # {0,1}
  g          = sign(norm(z_g))                    # {-1,+1}
  c          = f * c + i * g
  h          = o * sign(c)
  logits     = (h @ head) * logit_scale           # XNOR-popcount
  ```

## Binarization (simplest workable)

- Weights are `±1` via a sign straight-through estimator (BinaryConnect-style:
  real latent weights, `sign()` in the forward, hardtanh STE in the backward,
  latents clipped to `[-1, 1]`).
- LSTM gates are a hard `{0,1}` step; the candidate is `sign`. The cell is left
  as an integer accumulator — a binary cell cannot hold state, which defeats the
  point of an LSTM.
- A small training-only `cell_log_scale` widens the `sign(c)` STE window so
  gradients flow through long-term memory. It's a positive scalar, so the
  forward sign is unchanged and it is dropped at inference.

## Normalization / adaptive threshold (`--norm`)

Quantized recurrent nets suffer from exploding BPTT gradients. The fix
(following Hou et al., *Normalization Helps Training of Quantized LSTM*,
NeurIPS 2019) is to normalize the gate/layer pre-activations:

- `none` — `sign(z)`: fixed threshold at 0 (original behaviour).
- `mean` *(default, recommended)* — `sign(z - mean(z))`. The mean of the
  popcounts becomes a per-step **adaptive threshold**. At inference this needs
  only the mean of the popcounts you already computed — fully XNOR-compatible.
- `layer` — full layer norm `(z - mean)/std`. The `/std` is purely a
  training-time gradient scaler; with thresholds off it cancels under `sign()`,
  so inference still folds to a mean-based threshold.

With biases/thresholds off (the default), the inference-time threshold is just
`mean` (or, if you prefer exact balance, the median) of the popcounts — no
stored thresholds and no division.

## Thresholds / biases (`--use-thresholds`)

Optional integer thresholds (RNN) / gate biases (LSTM). **Off by default.**
Turning them on makes the inference fold require `std` as well, so leave them
off for the leanest retro deployment unless they clearly help.

## Files

- `model.py` — STE ops, `normalize_preact`, `BRNN`, `BLSTM`, `build_model`.
- `train.py` — data loading, training loop, checkpointing, CSV logging.
- `infer.py` — load a checkpoint and sample text.
- `plot_progress.py` — plot loss vs. time with a regression fit.

Vocabulary is derived from the distinct bytes in the training file; the head and
embedding are sized to it automatically. Checkpoints store a `config` dict and
rebuild from it (no backwards compatibility with older checkpoints).

## Usage

Train the vanilla RNN with the adaptive mean-threshold:

```bash
python train.py --model rnn --norm mean --num-ff 2 --carry-dim 896 \
  --file ./training.txt --checkpoint-path ./rnn.pt --csv-path ./rnn.csv
```

Train the binary LSTM:

```bash
python train.py --model lstm --norm mean --hidden-dim 1024 \
  --file ./training.txt --checkpoint-path ./lstm.pt --csv-path ./lstm.csv
```

Ablations:

```bash
python train.py --model rnn  --norm none                      # original behaviour
python train.py --model lstm --norm layer                     # textbook layer norm
python train.py --model lstm --norm mean --hidden-dim 384 \
  --batch-size 2048 --seq-len 128                              # fast iteration
```

Resume, plot, sample:

```bash
python train.py --resume --checkpoint-path ./lstm.pt --file ./training.txt
python plot_progress.py ./lstm.csv
python infer.py --checkpoint-path ./lstm.pt --prompt "The " --num-tokens 400 --temperature 0.7
```

## Key options

| Option | Meaning |
| --- | --- |
| `--model {rnn,lstm}` | architecture |
| `--norm {none,mean,layer}` | pre-activation normalization (default `mean`) |
| `--embed-dim` | token embedding width (both models) |
| `--carry-dim` / `--num-ff` | RNN carry width / number of layers |
| `--hidden-dim` | LSTM hidden width |
| `--use-thresholds` | enable integer thresholds/biases (default off) |
| `--batch-size`, `--seq-len`, `--lr`, `--steps` | training schedule |
| `--bf16`, `--device`, `--seed`, `--resume` | runtime |
