<p align="center">
  <a href="https://arxiv.org/abs/2508.04796">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-Paper-red">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img alt="License" src="https://img.shields.io/badge/License-MIT-green.svg">
  </a>
</p>

Parity-Aware Byte-Pair Encoding: Improving Cross-lingual Fairness in Tokenization
================================== 
This repository provides an implementation of the **Parity-Aware BPE** algorithm.
Paper: ["Parity-Aware Byte-Pair Encoding: Improving Cross-lingual Fairness in Tokenization"](https://arxiv.org/abs/2508.04796) [ACL 2026]


Overview
------------
**Parity-aware BPE** learns a tokenization that ensures parity in token lengths across languages on a multi-parallel development set.
Unlike standard BPE, which optimizes merges based on a single corpus, this approach explicitly considers cross-lingual fairness during the tokenization process.


Installation
------------

You can install this package directly from GitHub:

```bash
pip install git+https://github.com/swiss-ai/parity-aware-bpe.git
```

For development installation:

```bash
git clone https://github.com/swiss-ai/parity-aware-bpe.git
cd parity-aware-bpe
pip install -e .
```


Usage Instructions
------------------

The arguments of `parity-aware-bpe` are as follows:

- `--variant`: Parity-aware BPE variant. Options:
    - base – standard parity-aware BPE (default)
    - window – moving-window balancing version 
- `--input`: Space-separated list of training corpora (one per language).
- `--dev`: Space-separated list of development texts used for parity computation (multi-parallel). The tool assumes that the language of the nth input corpus corresponds to the nth dev corpus (same order as `--input`).
- `--ratio`: Space-separated list of desired compression ratios (floats), relative to pre-tokenized training set length, per input language. Can be used for parity computation (on training data) in lieu of development set.
- `--global-merges`: Optionally, one can perform the first M merge operations based on global frequency statistics (equivalent to standard BPE), and only switch to a parity-optimizing mode after (Hybrid parity-aware BPE). This argument controls how many merge operations are performed based on global statistics.
- `--symbols`: Total number of BPE merges to perform.
- `--output`: Path to the output file where BPE merge rules will be saved (one per line).
- `--total-symbols`: Adjusts the number of merges by subtracting character counts (so `--symbols` approximates total symbols needed).
- `--min-frequency`:  Minimum pair frequency to continue merging (default: `2`).
- `--window-size`: Context window size for the window-balancing variant (default: `100`).
- `--alpha`: Parameter controlling the moving-window balancing behavior (default: `2`).

### Example Usage

```bash
Python3 parity_aware_bpe/parity_aware_learn_bpe.py \
        --symbols {num_operations} \ 
        --variant {"base" or "window"} \
        --input {train_files}  \
        --dev {development_files}  \
        --output {output_file} 
```

Classical BPE
------------------
To run the classical BPE algorithm you can use `learn_bpe.py`:

```bash
Python3 parity_aware_bpe/learn_bpe.py \
        --symbols {num_operations} \ 
        --input {train_files}  \
        --dev {development_files}  \
        --output {output_file} 
```


Generating a Vocabulary
------------------
After learning the merges, you can build a vocabulary file using the `build_vocab_from_merges` function in `HF_tokenizer.py`.
To create a Hugging Face-compatible tokenizer:

```bash
Python3 parity_aware_bpe/HF_tokenizer.py \
        --merges_file_path {merge_file_path} \
        --tokenizer_path {tokenizer_save_folder}

```

Loading the tokenizer
------------------
```python
import os
from transformers import PreTrainedTokenizerFast
from tokenizers.pre_tokenizers import Whitespace, ByteLevel
from tokenizers.models import BPE
from tokenizers import Tokenizer, pre_tokenizers

merge_file = os.path.join(tokenizer_path, "merges.txt")
vocab_file = os.path.join(tokenizer_path, "vocab.json")
tokenizer = Tokenizer(BPE(vocab=vocab_file, merges=merge_file))
pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), ByteLevel(use_regex=False)]) # You need to use the same pre_tokenizer as the one used in BPE training
tokenizer.pre_tokenizer = pre_tokenizer

wrapped_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
```

Intrinsic Evaluation
------------------
For our intrinsic evaluation, we use `Tok##Suite`  to analyze and compare tokenizers across multiple languages and metrics. You can find the evaluation suite [here](https://github.com/cimeister/tokenizer-analysis-suite).

Citation
------------------

If you use this code for your research, please cite our paper:

``` bib
@article{foroutan-meister-et-al-2025-parity-aware-bpe,
  title={Parity-Aware Byte-Pair Encoding: Improving Cross-lingual Fairness in Tokenization},
  author={Foroutan, Negar and Meister, Clara and Paul, Debjit and Niklaus, Joel and Ahmadi, Sina, and Bosselut, Antoine and Sennrich, Rico},
  url={https://arxiv.org/abs/2508.04796},
  booktitle={arXiv},
  year={2025}
}
```
