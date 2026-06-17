import argparse
from typing import Dict

import torch

from model import build_model


def torch_load_checkpoint(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def normalize_mapping_keys(d: Dict) -> Dict[int, int]:
    return {int(k): int(v) for k, v in d.items()}


def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch_load_checkpoint(checkpoint_path)
    if "config" not in ckpt or "latent_state_dict" not in ckpt:
        raise ValueError(f"{checkpoint_path} is not a train.py checkpoint.")

    model = build_model(ckpt["config"]).to(device)
    model.load_state_dict({k: v.to(device) for k, v in ckpt["latent_state_dict"].items()})
    model.eval()

    byte_to_id = normalize_mapping_keys(ckpt.get("byte_to_id", {}))
    id_to_byte = normalize_mapping_keys(ckpt.get("id_to_byte", {}))
    return model, byte_to_id, id_to_byte


def encode_prompt(prompt, byte_to_id, device) -> torch.Tensor:
    ids = []
    for b in prompt.encode("utf-8"):
        if b not in byte_to_id:
            raise ValueError(f"Prompt byte {b} is not in the checkpoint vocabulary.")
        ids.append(byte_to_id[b])
    return torch.tensor(ids, dtype=torch.long, device=device)


def decode_tokens(tokens, id_to_byte) -> str:
    raw = bytearray()
    for token in tokens:
        token = int(token)
        raw.append(int(id_to_byte[token])) if token in id_to_byte else raw.extend(b"?")
    return bytes(raw).decode("utf-8", errors="replace")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=str, default="./checkpoint.pt")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--num-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="0 = greedy argmax; >0 = sample at this temperature")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.num_tokens < 0:
        raise ValueError("--num-tokens must be >= 0")
    if args.temperature < 0:
        raise ValueError("--temperature must be >= 0")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    model, byte_to_id, id_to_byte = load_model(args.checkpoint_path, device)
    print(f"(config: {model.config})")

    prompt_tokens = encode_prompt(args.prompt, byte_to_id, device)
    generated = model.generate(
        prompt_tokens=prompt_tokens,
        num_tokens=args.num_tokens,
        temperature=args.temperature,
    )
    print(args.prompt + decode_tokens(generated.tolist(), id_to_byte))


if __name__ == "__main__":
    main()
