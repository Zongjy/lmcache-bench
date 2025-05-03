import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from matplotlib import font_manager

plt.style.use('default')
font_path = '/home/liyi/lmcache/SimHei.ttf'
cn_font = font_manager.FontProperties(fname=font_path) 

def parse_args():
    p = argparse.ArgumentParser("Analyze attention sparsity")
    p.add_argument('--model_name', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct')
    p.add_argument('--seq_len',    type=int, default=4096)
    p.add_argument('--num_samples',type=int, default=1)
    p.add_argument('--save_dir',   type=str, default='/home/liyi/lmcache/visualization')
    p.add_argument('--seed',       type=int, default=227)
    return p.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(seed)

def load_long_samples(tokenizer, seq_len, num_samples, seed):
    ds = load_dataset("allenai/paloma", "c4_en", split="test").shuffle(seed).select(range(100))
    samples = []
    for text in ds["text"]:
        ids = tokenizer(text, truncation=True, max_length=seq_len, return_tensors='pt').input_ids[0]
        if ids.size(0) == seq_len:
            samples.append(ids)
            if len(samples) == num_samples:
                break
    return torch.stack(samples).to('cuda')

@torch.inference_mode()
def get_attentions(model_name, inputs):
    model = AutoModelForCausalLM.from_pretrained(model_name,
        torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="eager")
    model.eval()
    return model(input_ids=inputs, output_attentions=True).attentions

def build_sparsity_df(attns, model_name, seq_len):
    data = []
    for layer_idx, layer in enumerate(attns):
        attn = layer[0, :, -1, :].mean(0)       # last token, mean over heads
        total = attn.sum().item()
        sorted_attn = torch.sort(attn, descending=True).values
        cum_pct = torch.cumsum(sorted_attn, 0) / total * 100
        for i, pct in enumerate(cum_pct):
            data.append({
                'Layer': layer_idx,
                'TokenFrac': (i+1)/seq_len,
                'CumAttn%': float(pct),
            })
    return pd.DataFrame(data)

def plot_sparsity(df, model_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    layers = [0, 5, 11, 15, 23, 27, 31]
    markers = ['o', 's', '^', 'v', 'D', '*']
    prop_cycler = plt.rcParams['axes.prop_cycle']
    colors = [d['color'] for d in prop_cycler]
    crossing_xs = []

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, L in enumerate(layers):
        sub = df[df['Layer']==L]
        color = colors[i % len(colors)]

        # x_raw = sub['TokenFrac'].values
        # y_raw = sub['CumAttn%'].values

        eps_x  = np.array([0.0, 1e-4, 2e-4, 4e-4, 6e-4, 9e-4, 1e-3, 2e-3, 3e-3, 4e-3,])
        eps_y  = np.zeros_like(eps_x)
        x_raw = np.concatenate((eps_x, sub['TokenFrac'].values))
        y_raw = np.concatenate((eps_y, sub['CumAttn%'].values))
        order = np.argsort(x_raw)
        x_raw = x_raw[order]
        y_raw = y_raw[order]

        from scipy.ndimage import gaussian_filter1d
        y_smooth = gaussian_filter1d(y_raw, sigma=20)

        from scipy.interpolate import UnivariateSpline
        f = UnivariateSpline(x_raw, y_smooth, s=0)
        x = np.linspace(0, 1, 1000)
        y = np.clip(f(x), 0, 100)

        ax.plot(x, y,
                color=color,
                linewidth=2,
                alpha=0.8,
                label=f'第 {L+1} 层')
        
        mask = y >= 90
        if mask.any():
            xi = x[mask][0]
            yi = y[mask][0]
            crossing_xs.append(xi)
            ax.scatter([xi], [yi],
                       marker=markers[i % len(markers)],
                       s=100,
                       color=color,
                       zorder=5)


    ax.hlines(90, 0, 1, colors='gray', linestyles='--', label='累计注意力分数 90%')

    if crossing_xs:
        x_max = min(1.0, max(crossing_xs) * 1.1)
    else:
        x_max = 1.0
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 105)

    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x*100)}"))
    
    ax.set_xlabel('选择词元数占比 (%)', fontsize=16, fontproperties=cn_font)
    ax.set_ylabel('累计注意力分数占比 (%)', fontsize=16, fontproperties=cn_font)
    ax.set_title('不同层稀疏度分析', fontsize=20, pad=8, fontproperties=cn_font)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.legend(loc='lower right', fontsize=14, frameon=False, ncol=2, prop=cn_font)

    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.9)
    # ax.set_axisbelow(True)  # 网格线在曲线之下

    plt.tight_layout()
    print(f"Saving attention sparsity plot to {os.path.join(save_dir, 'attention_sparsity.pdf')}")
    plt.savefig(os.path.join(save_dir, 'attention_sparsity.pdf'), dpi=300)
    plt.close()

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    samples = load_long_samples(tokenizer, args.seq_len, args.num_samples, args.seed)
    attns   = get_attentions(args.model_name, samples)
    df      = build_sparsity_df(attns, args.model_name, args.seq_len)

    del attns
    torch.cuda.empty_cache()
    plot_sparsity(df, args.model_name, args.save_dir)

if __name__ == '__main__':
    main()