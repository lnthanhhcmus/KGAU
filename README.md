# KGAU

Official code for **A Negative-Sampling-Free Training Framework for Knowledge Graph Embeddings via Alignment and Uniformity** (Le-Nguyen, Nguyen, and Le; preprint submitted to Elsevier).

Knowledge Graph Embedding (KGE) models are usually trained with negative sampling, which is expensive and only supervises the embedding space indirectly. **KGAU** is a training framework for *decomposable* KGE models (score functions that factor into a relation-conditioned query and a target). It replaces negative sampling with a positive-only geometric objective:

- **Alignment** pulls query and target representations of observed triples together.
- **Uniformity** spreads query, target, and entity embeddings on the unit hypersphere.

The base architecture and its inference score are kept. On WN18RR and FB15k-237, KGAU matches or improves most paired baselines (MRR gains of up to 0.037), while cutting training time per epoch by about 70% and peak GPU memory by about 65% on ComplEx.

**Authors:** [Gia Bao Le-Nguyen](mailto:lngbao22@clc.fitus.edu.vn), [Uyen Ngoc Phuong Nguyen](mailto:nnpuyen22@clc.fitus.edu.vn), [Thanh Le](mailto:lnthanh@fit.hcmus.edu.vn) (University of Science, VNU-HCM).

## Setup

Python 3.8+ (3.10 or 3.11 recommended). Paper runs used an NVIDIA Tesla T4 (16 GB) for the main tables and an RTX 3060 Ti (8 GB) for supplementary tests.

```bash
python -m venv .venv
```

Activate the venv in the **same** shell before installing (the prompt should show `(.venv)`):

```bash
# Linux/macOS
source .venv/bin/activate

# Windows Command Prompt (cmd)
.venv\Scripts\activate.bat

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks the script, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first. Confirm the interpreter is the venv, not Miniconda/system Python:

```bash
python -c "import sys; print(sys.executable)"
# expect: ...\KGDirectAU\.venv\Scripts\python.exe  (Windows)
#     or: .../KGDirectAU/.venv/bin/python          (Linux/macOS)
```

Then install PyTorch for your CUDA driver (see comments in `requirements.txt`), then the rest:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

WN18RR and FB15k-237 raw splits are under `data/`. Preprocess once:

```bash
python data/preprocess.py --dataset wn18rr
python data/preprocess.py --dataset fb15k237
```

This writes JSON splits and id maps to `data/<dataset>/preprocessed/`. Labeled `*_w_label.txt` files (R2D2-style negatives for triple classification) are included.

## Reproducing the paper

Each experiment is one JSON file under `configs/`. The `--model` field selects the architecture; AU variants use `au_loss.py` and `kgau_strategy.py` (no negative sampler). CLI flags override JSON.

```bash
# WN18RR baselines (Table 2, “Original”)
python main.py --config-path configs/TransE_WN18RR_margin_ranking.json
python main.py --config-path configs/DistMult_WN18RR_adversarial_bce.json
python main.py --config-path configs/ComplEx_WN18RR_adversarial_bce.json
python main.py --config-path configs/RotatE_WN18RR_adversarial_bce.json
python main.py --config-path configs/pRotatE_WN18RR_adversarial_bce.json
python main.py --config-path configs/TransERR_WN18RR.json
python main.py --config-path configs/DaBR_WN18RR.json

# WN18RR KGAU (Table 2, “KGAU”)
python main.py --config-path configs/TransE-AU_WN18RR.json
python main.py --config-path configs/DistMult-AU_WN18RR.json
python main.py --config-path configs/ComplEx-AU_WN18RR.json
python main.py --config-path configs/RotatE-AU_WN18RR.json
python main.py --config-path configs/pRotatE-AU_WN18RR.json
python main.py --config-path configs/TransERR-AU_WN18RR.json
python main.py --config-path configs/DaBR-AU_WN18RR_independent_spheres.json

# FB15k-237: same families as configs/*_FB15k237*.json (Table 3)

python main.py @configs/ComplEx-AU_WN18RR.json          # shorthand for --config-path
python main.py --config-path configs/ComplEx-AU_WN18RR.json --task lp
python main.py --config-path configs/ComplEx-AU_WN18RR.json --is-test --eval-model-path logs/<run>/best_model.mdl
```

Runs write `logs/<model-dataset>_<yyyy-mm-dd>_<hh-mm-ss>/` with `run.log`, `results.txt`, `best_model.mdl`, and `last_model.mdl`. The paper reports the checkpoint with the best validation MRR (cosine MRR for distance-based AU models). Test dumps may also include native and Lp-distance scores; **the numbers below are the manuscript Tables 2–3**.

Hyperparameters follow Appendix Tables C.4 (baselines) and C.5 (KGAU). Validation is every 5 epochs. DaBR / DaBR-AU use 2,000 epochs rather than the 20,000 of the original DaBR paper, as in the manuscript.

### Reported test results (manuscript)

Link prediction, filtered ranking. Bold = improvement over the paired original model.

**WN18RR (Table 2)**


| Model       | MRR        | Hits@1     | Hits@3     | Hits@10    | Acc.       |
| ----------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| TransE      | 0.2065     | 0.0147     | 0.3843     | 0.4546     | 0.6924     |
| TransE-AU   | 0.1865     | **0.0434** | 0.2688     | 0.4542     | **0.8789** |
| DistMult    | 0.4212     | 0.3778     | 0.4315     | 0.5126     | 0.8109     |
| DistMult-AU | **0.4378** | **0.3912** | **0.4609** | **0.5242** | **0.8312** |
| ComplEx     | 0.4451     | 0.4033     | 0.4616     | 0.5265     | 0.8465     |
| ComplEx-AU  | **0.4818** | **0.4414** | **0.5014** | **0.5542** | **0.8545** |
| RotatE      | 0.4753     | 0.4335     | 0.4916     | 0.5562     | 0.8398     |
| RotatE-AU   | 0.4647     | 0.4207     | 0.4856     | 0.5487     | **0.8419** |
| pRotatE     | 0.4717     | 0.4325     | 0.4887     | 0.5449     | 0.8591     |
| pRotatE-AU  | 0.4519     | 0.4032     | 0.4747     | **0.5453** | 0.8470     |
| TransERR    | 0.4746     | 0.4397     | 0.4858     | 0.5454     | 0.8387     |
| TransERR-AU | 0.4682     | 0.4171     | **0.4904** | **0.5668** | **0.8735** |
| DaBR        | 0.4083     | 0.3570     | 0.4332     | 0.5027     | 0.7998     |
| DaBR-AU     | **0.4461** | **0.4037** | **0.4652** | **0.5270** | **0.8285** |


**FB15k-237 (Table 3)**


| Model       | MRR        | Hits@1     | Hits@3     | Hits@10    | Acc.       |
| ----------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| TransE      | 0.2886     | 0.1898     | 0.3278     | 0.4805     | 0.5101     |
| TransE-AU   | 0.2865     | **0.2027** | 0.3116     | 0.4551     | **0.6771** |
| DistMult    | 0.2874     | 0.2032     | 0.3135     | 0.4568     | 0.7411     |
| DistMult-AU | **0.3075** | **0.2207** | **0.3349** | **0.4837** | **0.7464** |
| ComplEx     | 0.3217     | 0.2313     | 0.3523     | 0.5045     | 0.7518     |
| ComplEx-AU  | **0.3296** | **0.2378** | **0.3626** | **0.5168** | **0.7522** |
| RotatE      | 0.3401     | 0.2451     | 0.3765     | 0.5312     | 0.7902     |
| RotatE-AU   | 0.2651     | 0.1867     | 0.2879     | 0.4238     | 0.7037     |
| pRotatE     | 0.3044     | 0.2135     | 0.3354     | 0.4879     | 0.7639     |
| pRotatE-AU  | 0.2472     | 0.1719     | 0.2667     | 0.3960     | 0.6890     |
| TransERR    | 0.3649     | 0.2707     | 0.4008     | 0.5554     | 0.7980     |
| TransERR-AU | 0.2525     | 0.1769     | 0.2715     | 0.4051     | 0.7079     |
| DaBR        | 0.2994     | 0.2039     | 0.3281     | 0.4976     | 0.7454     |
| DaBR-AU     | 0.2770     | 0.1951     | 0.3011     | 0.4418     | 0.7294     |


F1 / PR-AUC / ROC-AUC and per-epoch time / peak GPU memory are in the paper. Table 4 of the manuscript is a contextual comparison with unreimplemented KGC systems, not a KGAU pairing.

## Configs

WN18RR filenames; FB15k-237 uses the same wiring under `configs/*_FB15k237*.json`. Embedder is `lookup_embedder.py` unless overridden.


| Model       | Config                                    | Scorer        | Loss                         | Sampler                        | Strategy              |
| ----------- | ----------------------------------------- | ------------- | ---------------------------- | ------------------------------ | --------------------- |
| TransE      | `TransE_WN18RR_margin_ranking.json`       | `transe.py`   | `margin_ranking_loss.py`     | `filtered_1_to_n_sampler.py`   | `negsamp_strategy.py` |
| TransE-AU   | `TransE-AU_WN18RR.json`                   | `transe.py`   | `au_loss.py`                 | —                              | `kgau_strategy.py`    |
| DistMult    | `DistMult_WN18RR_adversarial_bce.json`    | `distmult.py` | `adversarial_bce_loss.py`    | `filtered_1_to_n_sampler.py`   | `negsamp_strategy.py` |
| DistMult-AU | `DistMult-AU_WN18RR.json`                 | `distmult.py` | `au_loss.py`                 | —                              | `kgau_strategy.py`    |
| ComplEx     | `ComplEx_WN18RR_adversarial_bce.json`     | `complex.py`  | `adversarial_bce_loss.py`    | `filtered_1_to_n_sampler.py`   | `negsamp_strategy.py` |
| ComplEx-AU  | `ComplEx-AU_WN18RR.json`                  | `complex.py`  | `au_loss.py`                 | —                              | `kgau_strategy.py`    |
| RotatE      | `RotatE_WN18RR_adversarial_bce.json`      | `rotate.py`   | `adversarial_bce_loss.py`    | `filtered_1_to_n_sampler.py`   | `negsamp_strategy.py` |
| RotatE-AU   | `RotatE-AU_WN18RR.json`                   | `rotate.py`   | `au_loss.py`                 | —                              | `kgau_strategy.py`    |
| pRotatE     | `pRotatE_WN18RR_adversarial_bce.json`     | `protate.py`  | `adversarial_bce_loss.py`    | `filtered_1_to_n_sampler.py`   | `negsamp_strategy.py` |
| pRotatE-AU  | `pRotatE-AU_WN18RR.json`                  | `protate.py`  | `au_loss.py`                 | —                              | `kgau_strategy.py`    |
| TransERR    | `TransERR_WN18RR.json`                    | `transerr.py` | `adversarial_bce_loss.py`    | `filtered_1_to_n_sampler.py`   | `negsamp_strategy.py` |
| TransERR-AU | `TransERR-AU_WN18RR.json`                 | `transerr.py` | `au_loss.py`                 | —                              | `kgau_strategy.py`    |
| DaBR        | `DaBR_WN18RR.json`                        | `dabr.py`     | `pointwise_logistic_loss.py` | `uniform_pointwise_sampler.py` | `negsamp_strategy.py` |
| DaBR-AU     | `DaBR-AU_WN18RR_independent_spheres.json` | `dabr.py`     | `au_loss.py`                 | —                              | `kgau_strategy.py`    |


KGAU does not sample negatives. Extra losses, samplers, and strategies in the tree (including SimKGC) are for custom configs and are **not** paper experiments.

## Repository layout

```
KGDirectAU/
├── main.py
├── configs/                 # JSON recipes + config.py
├── data/                    # WN18RR, FB15k237, preprocess.py
├── models/
│   ├── builder.py           # embedder + scorer + loss + sampler + strategy
│   ├── embedders/
│   ├── losses/au_loss.py
│   ├── strategies/kgau_strategy.py
│   └── <model>.py           # TransE, DistMult, ComplEx, RotatE, …
├── base/                    # trainer, evaluator, KGE bases
├── metrics/                 # ranking + triple classification
└── utils/
```

## Citation

```bibtex
@article{lenguyen2026kgau,
  title   = {A Negative-Sampling-Free Training Framework for Knowledge Graph Embeddings via Alignment and Uniformity},
  author  = {Le-Nguyen, Gia Bao and Nguyen, Uyen Ngoc Phuong and Le, Thanh},
  journal = {Preprint submitted to Elsevier},
  year    = {2026}
}
```

Data: [WN18RR](https://github.com/TimDettmers/ConvE), [FB15k-237](https://web.informatik.uni-mannheim.de/LinkedData/FB15K237/). Triple-classification labels follow R2D2. Alignment and Uniformity follow Wang and Isola (ICML 2020) and DirectAU (KDD 2022).
