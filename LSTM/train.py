import argparse
import csv
import math
import os
import random
import time
from datetime import datetime
from typing import Dict, Tuple

import torch

from model import build_model, MODEL_TYPES, NORM_MODES


# ============================================================
# Data
# ============================================================

# def load_byte_training_file(path: str) -> Tuple[torch.Tensor, Dict[int, int], Dict[int, int]]:
#     """Read a byte file and build a vocabulary from the distinct bytes present."""
#     with open(path, "rb") as f:
#         raw = f.read()
#     if len(raw) == 0:
#         raise ValueError(f"{path} is empty")

#     distinct = sorted(set(raw))
#     byte_to_id = {b: i for i, b in enumerate(distinct)}
#     id_to_byte = {i: b for b, i in byte_to_id.items()}

#     encoded = torch.tensor([byte_to_id[b] for b in raw], dtype=torch.long)
#     return encoded, byte_to_id, id_to_byte


# reverting to fixed ascii, TODO: make fixing to ascii an option rather than this hardcoded monkeypatch

def load_byte_training_file(
    path: str,
) -> Tuple[torch.Tensor, Dict[int, int], Dict[int, int]]:
    with open(path, "rb") as f:
        raw = f.read()

    if len(raw) == 0:
        raise ValueError(f"{path} is empty")

    # Fixed ASCII vocabulary (ids 0..127).
    distinct = list(range(128))

    byte_to_id = {b: i for i, b in enumerate(distinct)}
    id_to_byte = {i: b for b, i in byte_to_id.items()}

    missing = [b for b in set(raw) if b not in byte_to_id]
    if missing:
        raise ValueError(
            f"File contains byte values outside the ASCII vocabulary: {sorted(missing)[:8]}..."
        )

    encoded = torch.tensor([byte_to_id[b] for b in raw], dtype=torch.long)
    return encoded, byte_to_id, id_to_byte

def make_random_batch(encoded_cpu, batch_size, seq_len, device) -> torch.Tensor:
    n = encoded_cpu.numel()
    if n <= seq_len:
        raise ValueError(
            f"Training file has only {n} tokens; need more than seq_len={seq_len}."
        )
    starts = torch.randint(0, n - seq_len, (batch_size,))
    offsets = torch.arange(seq_len)
    indices = starts.unsqueeze(1) + offsets.unsqueeze(0)
    return encoded_cpu[indices].to(device=device, dtype=torch.long, non_blocking=True)


# ============================================================
# Device
# ============================================================

def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        if device.index is not None:
            torch.cuda.set_device(device)
        device = torch.device("cuda", torch.cuda.current_device())
    return device


# ============================================================
# Checkpoints
# ============================================================

def save_checkpoint(path, model, optimizer, byte_to_id, id_to_byte, step) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "step": int(step),
        "config": model.config,
        "latent_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "quantized": model.export_quantized(),
        "optimizer_state_dict": optimizer.state_dict(),
        "byte_to_id": byte_to_id,
        "id_to_byte": id_to_byte,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python_random_state": random.getstate(),
    }
    torch.save(payload, path)
    now = datetime.now()
    print(f"\n>>> Checkpoint saved to {path} at step {step}")
    print(f">>> Time: {now.strftime('%Y-%m-%d %H:%M:%S')} ({int(now.timestamp())})")


def torch_load_checkpoint(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_checkpoint(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    print(f"Loading checkpoint: {path}")
    ckpt = torch_load_checkpoint(path)

    model = build_model(ckpt["config"]).to(device)
    model.load_state_dict({k: v.to(device) for k, v in ckpt["latent_state_dict"].items()})

    step = int(ckpt.get("step", 0))
    byte_to_id = ckpt.get("byte_to_id", {})
    id_to_byte = ckpt.get("id_to_byte", {})
    optim_state = ckpt.get("optimizer_state_dict", None)

    if "torch_rng_state" in ckpt:
        torch.set_rng_state(ckpt["torch_rng_state"])
    if ckpt.get("cuda_rng_state_all") and torch.cuda.is_available():
        states = ckpt["cuda_rng_state_all"]
        if len(states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(states)
    if "python_random_state" in ckpt:
        random.setstate(ckpt["python_random_state"])

    print(f"Resumed at step {step}: {model.config}")
    return model, step, byte_to_id, id_to_byte, optim_state


# ============================================================
# Logging
# ============================================================

def append_csv_row(csv_path, step, loss_nats) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["step", "loss_nats_per_token", "loss_bits_per_token", "unixtime"])
        writer.writerow([step, loss_nats, loss_nats / math.log(2.0), int(datetime.now().timestamp())])


def print_status(step, loss_nats, steps_per_sec) -> None:
    print(
        f"Step {step}. Loss: {loss_nats / math.log(2.0):.6f} bits/token. "
        f"({steps_per_sec:.1f} step/s)",
        flush=True,
    )


# ============================================================
# Main
# ============================================================

def build_config_from_args(args, vocab_size):
    if args.model == "rnn":
        return {
            "model_type": "rnn",
            "vocab_size": vocab_size,
            "embed_dim": args.embed_dim,
            "carry_dim": args.carry_dim,
            "num_ff": args.num_ff,
            "use_thresholds": args.use_thresholds,
            "norm_mode": args.norm,
        }
    return {
        "model_type": "lstm",
        "vocab_size": vocab_size,
        "embed_dim": args.embed_dim,
        "hidden_dim": args.hidden_dim,
        "use_thresholds": args.use_thresholds,
        "norm_mode": args.norm,
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--file", type=str, default="./training.txt")

    parser.add_argument("--model", type=str, default="rnn", choices=MODEL_TYPES)
    parser.add_argument("--norm", type=str, default="mean", choices=NORM_MODES,
                        help="pre-activation normalization ('mean' recommended; "
                             "'none' = fixed threshold at 0)")
    parser.add_argument("--embed-dim", type=int, default=128,
                        help="token embedding width (both models)")
    parser.add_argument("--carry-dim", type=int, default=896,
                        help="(rnn) recurrent carry width; state_dim = carry_dim + embed_dim")
    parser.add_argument("--num-ff", type=int, default=2,
                        help="(rnn) number of feed-forward layers")
    parser.add_argument("--hidden-dim", type=int, default=1024,
                        help="(lstm) hidden state width")
    parser.add_argument("--use-thresholds", action="store_true",
                        help="enable integer thresholds (rnn) / gate biases (lstm); off by default")

    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seq-len", type=int, default=256)

    parser.add_argument("--lr", type=float, default=1e-3, help="LR for sign-quantized weights")
    parser.add_argument("--thresh-lr", type=float, default=1e-2,
                        help="LR for integer thresholds/biases and aux scalars")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--bf16", action="store_true", help="bfloat16 autocast (CUDA only)")
    parser.add_argument("--no-tf32", action="store_true", help="disable TF32 matmul")

    parser.add_argument("--steps", type=int, default=0, help="0 = train forever")

    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--csv-every", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--checkpoint-path", type=str, default="./checkpoint.pt")
    parser.add_argument("--csv-path", type=str, default="./data.csv")

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.seq_len < 2:
        raise ValueError("--seq-len must be >= 2")

    device = resolve_device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if not args.no_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if args.bf16 and device.type != "cuda":
        raise RuntimeError("--bf16 requires CUDA")
    amp_enabled = bool(args.bf16)

    print(f"device: {device}")
    print(f"Training data: {args.file}")
    encoded_cpu, byte_to_id, id_to_byte = load_byte_training_file(args.file)
    vocab_size = len(byte_to_id)
    print(f"Loaded {encoded_cpu.numel()} tokens; vocab size {vocab_size}.")

    if args.resume:
        model, step, ckpt_b2i, ckpt_i2b, optim_state = load_checkpoint(args.checkpoint_path, device)
        if ckpt_b2i and ckpt_b2i != byte_to_id:
            raise ValueError("Checkpoint vocabulary does not match the current training file.")
        byte_to_id, id_to_byte = ckpt_b2i or byte_to_id, ckpt_i2b or id_to_byte
    else:
        model = build_model(build_config_from_args(args, vocab_size)).to(device)
        step = 0
        optim_state = None

    groups = [{"params": model.sign_parameters(), "lr": args.lr, "weight_decay": args.weight_decay}]
    extra = model.int_parameters() + model.aux_parameters()
    if extra:
        groups.append({"params": extra, "lr": args.thresh_lr, "weight_decay": 0.0})

    optimizer = torch.optim.AdamW(groups)
    if optim_state is not None:
        optimizer.load_state_dict(optim_state)

    print(f"model config: {model.config}")
    print(f"batch shape: ({args.batch_size}, {args.seq_len}); bf16: {amp_enabled}; start step: {step}")

    model.train()
    t0 = time.time()
    log_t0, log_step0 = t0, step

    while True:
        if args.steps > 0 and step >= args.steps:
            break

        batch = make_random_batch(encoded_cpu, args.batch_size, args.seq_len, device)

        if amp_enabled:
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(batch)
        else:
            loss = model(batch)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        model.clip_latents_()

        step += 1
        loss_val = float(loss.item())

        if args.csv_every > 0 and step % args.csv_every == 0:
            append_csv_row(args.csv_path, step, loss_val)

        if args.print_every > 0 and step % args.print_every == 0:
            now = time.time()
            print_status(step, loss_val, (step - log_step0) / max(now - log_t0, 1e-6))
            log_t0, log_step0 = now, step

        if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
            save_checkpoint(args.checkpoint_path, model, optimizer, byte_to_id, id_to_byte, step)

    save_checkpoint(args.checkpoint_path, model, optimizer, byte_to_id, id_to_byte, step)


if __name__ == "__main__":
    main()
