# scripts/tflayout/13_train_bpe.py

import argparse
import os

import numpy as np
import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


def run_train_bpe(promoter_seq="out/promoter_seq.parquet", outdir="out",
                  vocab_size=4000, min_frequency=10):
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_parquet(promoter_seq)
    print(f"读入 {len(df)} 条启动子序列作为训练语料")

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    # 故意不设 pre_tokenizer：保持"整条输入字符串就是一个词"的默认行为，理由见
    # 文件头说明。不要在这里加 Whitespace()之类的pre_tokenizer。
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["[PAD]", "[UNK]"],  # [PAD]排第一个，训练后期望id=0，
                                              # 跟12_cis_transformer.py默认
                                              # pad_token_id=0对齐
    )
    tokenizer.train_from_iterator(iter(df["seq"]), trainer=trainer, length=len(df))

    vocab = tokenizer.get_vocab()
    print(f"训练完成，实际词表大小 {len(vocab)}"
          f"（目标{vocab_size}，语料不够大/min_frequency偏高时达不到目标值属正常）")

    pad_id = tokenizer.token_to_id("[PAD]")
    print(f"[PAD] 的 id = {pad_id}（应为0，不是的话12_cis_transformer.py的"
          f"pad_token_id参数要改成这个值）")

    multi_char = sorted((t for t in vocab if len(t) > 1 and not t.startswith("[")),
                        key=len, reverse=True)
    print(f"学到的多字符token数: {len(multi_char)}"
          "（这一行如果是0，说明pre_tokenizer的设计理解错了，BPE没学到任何合并，"
          "回头再查，不要往下用）")
    if multi_char:
        print(f"  最长的10个: {multi_char[:10]}")

    tokenizer_path = os.path.join(outdir, "bpe_tokenizer.json")
    tokenizer.save(tokenizer_path)
    print(f"-> {tokenizer_path}")

    # 抽样看压缩率(1500bp原始碱基 -> 多少个token)，帮后面定Dataset/Transformer的
    # padding长度用多少合适
    sample = df["seq"].head(200).tolist()
    lens = np.array([len(tokenizer.encode(s).ids) for s in sample])
    print(f"抽样200条序列的token数：中位{np.median(lens):.0f}，"
          f"90%分位{np.percentile(lens, 90):.0f}，最大{lens.max()}"
          f"（原始每条都是1500bp，压缩比≈1500/中位数≈{1500 / np.median(lens):.1f}倍；"
          "12_cis_transformer.py自检时N=50是随便设的示例值，接真实数据要按这里的"
          "90%分位数来定padding长度）")

    # 顺手把可复现性验证一下：随便挑一条，encode再decode，应该原样回来(允许N/大小写
    # 这类边界字符有细微差异，但ACGT主体应该完全一致)
    probe = df["seq"].iloc[0]
    back = tokenizer.decode(tokenizer.encode(probe).ids)
    print(f"可逆性抽查：encode再decode跟原序列是否一致: {back.replace(' ', '') == probe}"
          "（如果不一致，可能是decode默认会在token之间插空格，需要额外处理，"
          "不影响训练本身，只影响想把token id倒推回序列时要不要去空格）")

    # 把全部序列编码好存下来，Dataset直接读，不用每次都重新分词
    all_ids = [tokenizer.encode(s).ids for s in df["seq"]]
    out_df = pd.DataFrame({"gene_id": df["gene_id"], "token_ids": all_ids})
    out_path = os.path.join(outdir, "promoter_token_ids.parquet")
    out_df.to_parquet(out_path)
    print(f"-> {out_path}（列：gene_id, token_ids；每行一个变长int列表，"
          "DataLoader阶段再按batch动态padding，参考09_torch_dataset.py的"
          "collate_fn写法）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--promoter-seq", default="out/promoter_seq.parquet")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--vocab-size", type=int, default=4000)
    ap.add_argument("--min-frequency", type=int, default=10)
    a = ap.parse_args()
    run_train_bpe(a.promoter_seq, a.outdir, a.vocab_size, a.min_frequency)
