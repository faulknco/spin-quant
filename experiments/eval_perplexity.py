"""
Evaluate perplexity of a (possibly quantized) GPT-2 model on WikiText-2.

Usage:
    python experiments/eval_perplexity.py --scheme scalar --K 256 --block_dim 16
    python experiments/eval_perplexity.py --scheme on     --K_dir 256 --n_levels 16
    python experiments/eval_perplexity.py --scheme rg     --K_coarse 64 --K_residual 64
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transformers.pytorch_utils import Conv1D
from src.layers import CodebookLinear, ONLinear, RGLinear


def get_mlp_layers(model):
    """
    Return (name, parent_module, attr_name) for all MLP linear layers in GPT-2.

    GPT-2 in HuggingFace uses Conv1D (weight shape [in, out]) not nn.Linear
    ([out, in]), so we detect both.
    """
    targets = []
    for block_idx, block in enumerate(model.transformer.h):
        for attr in ["c_fc", "c_proj"]:
            layer = getattr(block.mlp, attr, None)
            if layer is not None and isinstance(layer, (torch.nn.Linear, Conv1D)):
                targets.append((f"h[{block_idx}].mlp.{attr}", block.mlp, attr))
    return targets


def conv1d_to_linear(layer) -> torch.nn.Linear:
    """
    Convert a HuggingFace Conv1D to nn.Linear.
    Conv1D stores weight as [in_features, out_features]; Linear stores [out, in].
    """
    if isinstance(layer, torch.nn.Linear):
        return layer
    in_f, out_f = layer.weight.shape
    lin = torch.nn.Linear(in_f, out_f, bias=layer.bias is not None)
    lin.weight = torch.nn.Parameter(layer.weight.T.contiguous())
    if layer.bias is not None:
        lin.bias = torch.nn.Parameter(layer.bias.clone())
    return lin


def quantize_model(model, args):
    targets = get_mlp_layers(model)
    print(f"Quantizing {len(targets)} layers with scheme={args.scheme} ...")
    for name, parent, attr in targets:
        linear = conv1d_to_linear(getattr(parent, attr))
        if args.scheme == "scalar":
            q_layer = CodebookLinear.from_linear(linear, block_dim=args.block_dim, K=args.K)
        elif args.scheme == "on":
            q_layer = ONLinear.from_linear(linear, block_dim=args.block_dim,
                                           K_dir=args.K_dir, n_levels=args.n_levels)
        elif args.scheme == "rg":
            q_layer = RGLinear.from_linear(linear, block_dim=args.block_dim,
                                           K_coarse=args.K_coarse, K_residual=args.K_residual)
        else:
            raise ValueError(f"Unknown scheme: {args.scheme}")
        setattr(parent, attr, q_layer)
        print(f"  quantized {name}")
    return model


def eval_perplexity(model, tokenizer, texts, max_length=128, device="cpu"):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = enc["input_ids"].to(device)
            out = model(input_ids, labels=input_ids)
            n_toks = input_ids.numel()
            total_loss += out.loss.item() * n_toks
            total_tokens += n_toks
    avg_nll = total_loss / total_tokens
    return torch.exp(torch.tensor(avg_nll)).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--scheme", choices=["baseline", "scalar", "on", "rg"], default="baseline")
    parser.add_argument("--block_dim", type=int, default=16)
    parser.add_argument("--K", type=int, default=256, help="codebook size (scalar scheme)")
    parser.add_argument("--K_dir", type=int, default=256, help="direction codebook size (on scheme)")
    parser.add_argument("--n_levels", type=int, default=16, help="norm levels (on scheme)")
    parser.add_argument("--K_coarse", type=int, default=64, help="coarse codebook size (rg scheme)")
    parser.add_argument("--K_residual", type=int, default=64, help="residual codebook size (rg scheme)")
    parser.add_argument("--n_texts", type=int, default=200)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = model.to(device)

    # baseline perplexity
    print("Loading WikiText-2 ...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in dataset["text"] if len(t.strip()) > 50][: args.n_texts]

    print("Evaluating baseline ...")
    ppl_base = eval_perplexity(model, tokenizer, texts, device=device)
    print(f"Baseline perplexity: {ppl_base:.2f}")

    if args.scheme != "baseline":
        model = quantize_model(model, args)
        print("Evaluating quantized model ...")
        ppl_q = eval_perplexity(model, tokenizer, texts, device=device)
        print(f"Quantized perplexity ({args.scheme}): {ppl_q:.2f}")
        print(f"Delta PPL: +{ppl_q - ppl_base:.2f}")


if __name__ == "__main__":
    main()
