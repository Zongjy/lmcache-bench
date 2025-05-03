import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    p = argparse.ArgumentParser("Visualize Attention Heatmap")
    p.add_argument('--model_name',   type=str,   default='meta-llama/Meta-Llama-3.1-8B-Instruct')
    p.add_argument('--seq_len',      type=int,   default=4096)
    p.add_argument('--num_samples',  type=int,   default=1)
    p.add_argument('--layer_idx',    type=int,   default=0, help="Which layer to plot (0-based)")
    p.add_argument('--save_dir',     type=str,   default='./visualization')
    p.add_argument('--seed',         type=int,   default=227)
    return p.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_long_samples(tokenizer, seq_len, num_samples, seed):
    ds = load_dataset("allenai/paloma", "c4_en", split="test") \
            .shuffle(seed) \
            .select(range(100))
    samples = []
    for text in ds["text"]:
        toks = tokenizer(text,
                         truncation=True,
                         max_length=seq_len,
                         return_tensors='pt').input_ids[0]
        if toks.size(0) == seq_len:
            samples.append(toks)
            if len(samples) == num_samples:
                break
    return torch.stack(samples).to('cuda')

@torch.inference_mode()
def get_attentions(model_name, inputs):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager"
    )
    model.eval()
    out = model(input_ids=inputs, output_attentions=True)
    return out.attentions

def plot_attention_heatmap(attns, layer_idx, seq_len, save_path=None):
    """
    attns: list of tensors, each shape (batch, heads, seq_len, seq_len)
    layer_idx: which layer to visualize
    seq_len: sequence length
    save_path: if provided, will save the figure
    """
    # select layer, batch=0, average over heads
    A = attns[layer_idx][0].mean(0).cpu().numpy()  # (seq_len, seq_len)
    # normalize per query (row)
    row_max = A.max(axis=1, keepdims=True) + 1e-12
    A_norm = A / row_max

    plt.figure(figsize=(8, 6))
    im = plt.imshow(A_norm, origin='lower', aspect='auto')
    plt.colorbar(im, label='Normalized Attention Score')
    plt.xlabel('Key Position (k)', fontsize=12)
    plt.ylabel('Query Position (q)', fontsize=12)
    plt.title(f'Layer {layer_idx+1} Attention Heatmap', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # tokenizer setup
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # load inputs and get attentions
    samples = load_long_samples(tokenizer, args.seq_len, args.num_samples, args.seed)
    attns   = get_attentions(args.model_name, samples)

    # plot heatmap for selected layer
    save_path = os.path.join(
        args.save_dir,
        f"attention_heatmap_layer{args.layer_idx+1}.pdf"
    )
    plot_attention_heatmap(attns, args.layer_idx, args.seq_len, save_path)
    print(f"Saved heatmap to {save_path}")

if __name__ == '__main__':
    main()