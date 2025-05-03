import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from matplotlib import font_manager

font_path = '/home/liyi/lmcache/SimHei.ttf'
cn_font = font_manager.FontProperties(fname=font_path) 

def parse_args():
    p = argparse.ArgumentParser("Visualize Attention Heatmap")
    p.add_argument('--model_name',   type=str,   default='meta-llama/Meta-Llama-3.1-8B-Instruct')
    p.add_argument('--seq_len',      type=int,   default=256)
    p.add_argument('--num_samples',  type=int,   default=1)
    p.add_argument('--layer_idx',    type=int,   default=0, help="Which layer to plot (0-based)")
    p.add_argument('--save_dir',     type=str,   default='/home/liyi/lmcache/visualization')
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
    A = attns[layer_idx][0].float().mean(0).cpu().numpy()  # (seq_len, seq_len)
    # normalize per query (row)
    row_max = A.max(axis=1, keepdims=True) + 1e-12
    A_norm = A / row_max

    A_tril = np.tril(A_norm)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
            A_tril, 
            cmap='PuBu',
            aspect='auto',
            origin='upper',
        )  
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('注意力分数（标准化）', fontsize=12, fontproperties=cn_font)
    ax.set_xlabel('键向量位置（key Position）', fontsize=14, fontproperties=cn_font)
    ax.set_ylabel('查询位置（Query Position）', fontsize=14, fontproperties=cn_font)
    ax.set_title('注意力热图', fontsize=16, fontproperties=cn_font)
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