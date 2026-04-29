# LCA-ICD: Label-Card Augmented ICD Coding

> **Authoritative label grounding for long-tail ICD-9 automatic coding**

LCA-ICD is a research prototype for automatic ICD-9 coding from clinical discharge summaries. The project focuses on a key problem in ICD coding: many codes are rare, semantically similar, and difficult to distinguish from sibling codes under the same ICD hierarchy. Instead of relying on free-form LLM-generated label expansion, this project uses authoritative coding resources to build traceable **ICD label cards**, then applies hierarchy-aware hard negative learning to improve long-tail and sibling-level discrimination.

This repository is intended for research on MIMIC-style ICD-9 coding and can be used to reproduce baseline experiments, construct official knowledge-augmented label text, train label-aware coders, and evaluate long-tail performance.

---

## Motivation

Automatic ICD coding is a multi-label classification task where each hospital admission can be assigned multiple diagnosis and procedure codes. Strong neural ICD coders can achieve good overall performance, but they still struggle with:

- **Long-tail labels**: many ICD codes appear only a few times in the training set.
- **Sibling-code confusion**: codes under the same ICD parent are often semantically close.
- **Label description sparsity**: ICD long titles are short and may not contain enough matching terms.
- **LLM hallucination risk**: unconstrained LLM expansion of ICD titles may introduce unsupported medical facts.

The central idea of this project is:

> LLMs should not be used as medical fact generators. Instead, they should act as constrained curators that organize official coding knowledge into verifiable, contrastive label representations.

---

## Core Idea

For each ICD-9 code, LCA-ICD builds an **official label card** using traceable external resources, such as:

- ICD-9-CM official code titles and descriptions
- ICD hierarchy information
- HCUP CCS category paths
- UMLS / ICD-related synonyms when available
- sibling codes under the same parent node

A label card is designed to provide richer but controlled label-side information:

```text
[CODE] 428.0
[TITLE] Congestive heart failure, unspecified
[PARENT] 428 Heart failure
[CCS_PATH] Diseases of circulatory system > heart failure
[SYNONYMS] congestive heart failure; CHF; cardiac failure
[SIBLING_CONTRAST] distinguish from left heart failure, systolic heart failure, diastolic heart failure
[SOURCE_TRACE] CMS / HCUP CCS / UMLS
```

The model then learns not only to predict ICD labels, but also to separate true labels from structurally similar hard negatives.

---

## Main Contributions

This project explores the following research directions:

1. **Authoritative label-card construction**
   - Replace free-form LLM ICD-title expansion with source-grounded label cards.
   - Keep each generated phrase traceable to official coding resources.

2. **Constrained LLM curation**
   - Use LLMs to organize, normalize, and compress official knowledge.
   - Avoid unsupported medical knowledge generation.

3. **Hierarchy-aware hard negative learning**
   - Select hard negatives from ICD siblings, CCS neighbors, and semantic label neighbors.
   - Improve discrimination among clinically similar long-tail labels.

4. **Evidence-preserving note representation**
   - Keep the raw discharge summary as the primary input.
   - Optionally add an LLM-generated evidence inventory, but only when extracted phrases can be matched back to the source note.

5. **Long-tail-oriented evaluation**
   - Report overall metrics and separate performance on head, medium-frequency, and tail ICD codes.
   - Analyze sibling false positives and rare-code recall.

---

## Method Overview

The full pipeline contains four stages.

### 1. Data Preparation

Input data should contain one row per hospital admission:

```text
SUBJECT_ID | HADM_ID | TEXT | LABELS
```

where:

- `TEXT` is the discharge summary or raw note text.
- `LABELS` is the set of ICD-9 codes assigned to the admission.
- `HADM_ID` is recommended as the stable admission-level identifier.

> **Important:** This repository does not include MIMIC notes, patient records, or other restricted clinical data.

### 2. Label Knowledge Construction

For each ICD-9 code, the project builds a label-side representation:

```text
code_norm
long_title
cms_title
umls_synonyms
ccs_level_1
ccs_level_2
ccs_level_3
parent_code
sibling_codes
label_card_raw
label_card_llm_constrained
source_trace_json
```

The label card is used as the text representation of each ICD label during retrieval, contrastive learning, or label-aware classification.

### 3. Model Training

The project supports label-aware ICD coding models such as PLM-CA-style and GKI-ICD-style architectures.

The training objective can combine:

```text
L = L_BCE
  + λ1 L_label_contrast
  + λ2 L_sibling_margin
  + λ3 L_evidence_alignment
  + λ4 L_view_consistency
```

where:

- `L_BCE`: standard multi-label binary classification loss.
- `L_label_contrast`: pulls note representations closer to gold label cards.
- `L_sibling_margin`: separates gold labels from sibling hard negatives.
- `L_evidence_alignment`: encourages label predictions to align with source-grounded evidence phrases.
- `L_view_consistency`: keeps predictions from raw-note and evidence-view inputs consistent.

### 4. Evaluation

Recommended metrics include:

- Micro-F1
- Macro-F1
- Precision@k, especially P@5 and P@8
- Recall@k, especially R@20
- Mean Average Precision
- Head / medium / tail label performance
- Rare-code recall
- Sibling false-positive rate

---

## Recommended Repository Structure

```text
LCA-ICD-project/
├── README.md
├── requirements.txt
├── configs/
│   ├── plmca_raw.yaml
│   ├── gki_raw.yaml
│   └── lca_icd.yaml
├── data/
│   └── README.md
├── prompts/
│   ├── evidence_inventory_prompt.txt
│   └── label_card_prompt.txt
├── scripts/
│   ├── preprocess_notes.py
│   ├── build_icd_hierarchy.py
│   ├── build_label_cards.py
│   ├── train.py
│   ├── evaluate.py
│   └── analyze_tail_labels.py
├── slurm/
│   ├── run_plmca_raw.sbatch
│   ├── run_gki_raw.sbatch
│   └── run_lca_icd.sbatch
├── src/
│   ├── data.py
│   ├── models.py
│   ├── losses.py
│   ├── hard_negative.py
│   ├── metrics.py
│   └── utils.py
└── outputs/
    └── README.md
```

Large files such as raw clinical notes, model checkpoints, generated summaries, and intermediate outputs should not be committed to GitHub.

---

## Installation

Create a Python environment:

```bash
conda create -n lca-icd python=3.10 -y
conda activate lca-icd
```

Install dependencies:

```bash
pip install -r requirements.txt
```

A minimal `requirements.txt` may include:

```text
torch
transformers
accelerate
pandas
numpy
scikit-learn
tqdm
pyyaml
openpyxl
```

---

## Quick Start

### 1. Prepare the ICD coding dataset

```bash
python scripts/preprocess_notes.py \
  --input_raw_notes /path/to/NOTEEVENTS.csv \
  --input_labels /path/to/labels.csv \
  --output data/notes_labeled.csv
```

Expected output format:

```text
HADM_ID,TEXT,LABELS
```

### 2. Build ICD hierarchy and sibling maps

```bash
python scripts/build_icd_hierarchy.py \
  --cms_icd9_zip /path/to/icd-9-cm-v32-master-descriptions.zip \
  --ccs_zip /path/to/Multi_Level_CCS_2015.zip \
  --local_vocab /path/to/icd9_vocab.xlsx \
  --output_dir data/icd9_hierarchy_assets
```

### 3. Build source-grounded label cards

```bash
python scripts/build_label_cards.py \
  --icd_vocab /path/to/icd9_vocab.xlsx \
  --hierarchy_dir data/icd9_hierarchy_assets \
  --output data/icd9_label_cards.csv
```

### 4. Train a baseline model

```bash
python scripts/train.py \
  --config configs/plmca_raw.yaml
```

### 5. Train LCA-ICD

```bash
python scripts/train.py \
  --config configs/lca_icd.yaml
```

### 6. Evaluate

```bash
python scripts/evaluate.py \
  --checkpoint outputs/lca_icd/best_model.pt \
  --test_file data/test.csv \
  --label_card_file data/icd9_label_cards.csv
```

---

## Suggested Experiments

| Setting | Purpose |
|---|---|
| PLM-CA + raw note | Strong clean baseline |
| GKI-ICD + raw note | Knowledge-injection baseline |
| PLM-CA + free LLM label expansion | Test unconstrained expansion noise |
| PLM-CA + official label card | Test source-grounded label representation |
| LCA-ICD + sibling hard negatives | Main hierarchy-aware method |
| LCA-ICD + evidence inventory | Final evidence-grounded variant |

---

## Ablation Studies

| Ablation | Question |
|---|---|
| w/o UMLS synonyms | Are synonyms useful? |
| w/o CCS hierarchy | Does coarse hierarchy help? |
| w/o sibling hard negatives | Is sibling contrast the key component? |
| w/o LLM curator | Is constrained LLM organization better than raw field concatenation? |
| free expansion vs constrained label card | Does source grounding reduce noise? |
| summary-only vs raw + evidence | Does summarization lose rare-code evidence? |

---

## Data and Privacy Notice

This repository is designed for research with restricted clinical datasets such as MIMIC-III or MIMIC-IV. The repository should only contain code, configuration files, prompts, and small synthetic examples.

Do **not** commit:

```text
NOTEEVENTS.csv
raw discharge summaries
patient-level clinical notes
full generated summaries from restricted data
MIMIC-derived text files
model checkpoints containing memorized clinical text
API keys or access tokens
```

Use a `.gitignore` file to exclude restricted data and large outputs.

Recommended `.gitignore` entries:

```gitignore
# restricted data
data/raw/
data/processed/
MIMIC/
NOTEEVENTS.csv
*.csv
*.xlsx
*.jsonl

# outputs and checkpoints
outputs/
checkpoints/
runs/
logs/
*.pt
*.pth
*.bin
*.safetensors

# environment and cache
.env
__pycache__/
*.pyc
.ipynb_checkpoints/
```

If small example files are needed, use fully synthetic or de-identified toy examples under `examples/`.

---

## Project Status

This project is currently a research prototype. The current focus is on:

- reproducing clean raw-note baselines;
- building reliable ICD-9 label cards from official resources;
- evaluating constrained label grounding against free-form LLM expansion;
- improving long-tail and sibling-confusable ICD code prediction.

---

## Citation

If you use this project, please cite the relevant baseline ICD coding models, official coding resources, and any dataset licenses required by your experimental setup.

A project citation can be added later:

```bibtex
@misc{lcaicd2026,
  title  = {LCA-ICD: Label-Card Augmented ICD Coding},
  author = {Your Name},
  year   = {2026},
  note   = {Research prototype}
}
```

---

## License

This repository is for academic research purposes. Add an appropriate open-source license before public release. Make sure the selected license is compatible with all third-party code, pretrained models, and datasets used in the project.
