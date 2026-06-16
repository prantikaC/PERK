# PERK: Personal Email Research Knowledge Graph

**PERK** is a personal knowledge graph (PKG) that captures the scientific activities of a
researcher — the tasks they work on, the methods and datasets they use, the papers they
write, the venues they submit to, the meetings they attend, and the people they
collaborate with — **as discussed in their academic emails**. The graph is constructed by
LLM-based extraction over email threads and grounded in the **PERKOnto** ontology.

| Resource | Link |
|---|---|
| **Persistent ontology URI** | <https://w3id.org/perkonto> |
| **Dataset / ontology metadata (VoID + DCAT + Dublin Core)** | [`ontology/void.ttl`](ontology/void.ttl) |
---
## Overview

<p align="center">
  <img src="perk_overview.png" alt="PERK resource overview" width="46%">
  &nbsp;&nbsp;
  <img src="kg_sample.png" alt="Sample PERK knowledge graph" width="46%">
</p>
<p align="center"><sub><b>Fig 1. (Left)</b> Resource overview. &nbsp; <b>Fig 2. (Right)</b> A snapshot of PERK.</sub></p>

Most knowledge graphs are built from public sources such as papers or the web. PERK
instead models a researcher's **own** scientific life as recorded in their inbox —
including in-progress work that may never appear in any publication. The goal is a
queryable personal graph that lets **autonomous agents answer questions and make
recommendations** over a researcher's activities (e.g. *"Which of my papers were under
review in 2019?"*, *"What meetings did I attend about the PKG project and what were the
agendas?"*).

## Motivation
This resource was motivated by a survey of researchers at our institute, which showed strong interest in personal research KGs (Chakraborty, Prantika, et al. "Bringing Order to Chaos: Conceptualizing a Personal Research Knowledge Graph for Scientists." IEEE Data Eng. Bull. 47.4 (2023): 43-56.). 

## Resources
The repository provides:

- **PATRA** — a corpus of synthetic academic email threads.
- **PERKOnto** — the ontology (14 entity types, 14 relation types) grounding the graph.
- **PRASHNA-PATRA** — a KG-QA benchmark (200 questions/answer pairs with Cypher).
- An **annotated gold set** of 2,372 triples for extraction evaluation.
- The full **construction + evaluation pipeline**: extraction → entity resolution →
  Neo4j graph build → triple & QA evaluation.

## Workflow
- The corpus of emails (PATRA) is first constructed/curated. Currently, the synthetic dataset is created by prompting an LLM. However, it requires post-processing to ensure that the corpus does not contain hallucinations and other errors (e.g., temporal inconsistencies).
- An ontology (PERKOnto) is then designed, capturing the entities and relations of interest.
- The PKG (PERK) is built by extracting triples from the email corpus conforming to ontological constraints. Currently, triples are extracted by prompting LLMs; the method does not guarantee perfect noise-free extraction.
- QA dataset (PRASHNA-PATRA) is built using the email corpus and the ontology (to restrict to ontology-specified entities and relations).

## PERKOnto

PERKOnto defines **14 entity types** and **14 relationship types** covering the research
collaboration domain. The machine-readable [`ontology/PERKOnto.json`](ontology/PERKOnto.json)
is used at runtime for ontology validation during KG cleaning (`clean_kg.py`),
schema-guided Cypher generation (`kg_eval.py`), and relationship ingestion during graph
construction (`build_perk.py`). Full serialisations (OWL, Turtle, RDF/XML, JSON-LD,
OWL/XML, N-Triples) are in [`ontology/`](ontology/).

**Generalising beyond NLP.** PERKOnto is domain-adaptable: the four NLP-specific classes
(`Task`, `Method`, `Metric`, `Dataset`) can be swapped for domain-specific ones while the
rest of the schema (people, emails, papers, venues, meetings, statuses) is reused. Three
example domain ontologies are included under
[`ontology/ontologies-for-other-fields/`](ontology/ontologies-for-other-fields/):

| Domain | Replacement classes |
|---|---|
| Computational chemistry | ResearchProblem, ComputationalMethod, Observable, ChemicalSystem |
| Gravitational physics | ResearchProblem, AnalysisMethod, PhysicalParameter, AstrophysicalSource, ObservationalData |
| Molecular biology | ResearchProblem, ExperimentalTechnique, Readout, BiologicalEntity, BiologicalSample |

<p align="center">
  <img src="ontology/PERKOnto.png" alt="PERKOnto ontology schema">
</p>
<p align="center"><sub><b>Fig 3.</b> The PERKOnto schema — 14 entity classes (nodes) and 14 relationship types (edges) modelling research collaboration in academic email.</sub></p>

The resource is actively maintained, with planned expansion to anonymised real emails and
to researchers in fields beyond computer science.

## Scope & Limitations

- **Synthetic emails, by necessity.** No public corpus of academic emails (real or
  synthetic) exists, and releasing real emails would compromise privacy. PATRA is
  therefore LLM-generated. The simulated timeline (April 2019 – March 2025) is fixed in
  the generation prompt and is independent of the model used.
- **Entities are extracted only from emails** — not from referenced papers — by design,
  so the graph reflects what the researcher actually discusses (including unpublished
  work). `Task` is interpreted broadly (paper writing, meeting organisation, etc.), not
  only research tasks. Enriching the graph from referenced papers is left to future work.
- **Single-pass LLM extraction is imperfect.** Even the strongest LLMs produce triples and synthetic emails
  that require post-processing; the non-trivial human-validation rejection rate motivates
  the cleaning/entity-resolution stages (see the paper, Sec. 7.2). Open-source models
  perform markedly worse than commercial ones.
- Given the limited annotated corpus and the high cost of email
  annotation, we adopt prompt-based in-context learning rather than supervised
  fine-tuning, and **systematically study** LLM-based PKG construction. The annotated
  corpus is released to support future supervised training.
- **Supervised baselines are not applicable**: they need large amounts of
  labelled, in-domain data to adapt to a new schema, which our 2,372-triple gold set
  cannot provide. Schema-free extractors align poorly with the ontology.
- Future work will focus on improving the accuracy of triple extraction by exploring various methods, such as simplifying the email text context, supervised training (fine-tuning / RLHF) of language models, using LLM-as-a-judge for triple verification and graph cleaning.

---
 
# Instructions for Users

## Repository Structure

```
PERK/
├── datasets/
│   ├── PATRA/                      # Synthetic email corpus
│   ├── PRASHNA_PATRA/              # QA benchmark
│   ├── extraction_gold/            # Annotated triples for extraction evaluation
│   └── neo4j_import/               # Per-type CSVs ready for Neo4j ingestion
├── ontology/
│   ├── PERKOnto.json               # Machine-readable ontology (used by pipeline)
│   ├── PERKOnto.ttl                # Turtle serialisation
│   └── PERKOnto.owx                # OWL/XML serialisation
├── results/
│   ├── entity_resolution/          # ER evaluation logs and error CSVs
│   ├── extractions/                # Per-model extracted entities and relations
│   ├── figures/                    # Generated plots (PDF)
│   └── qa/                         # KG-QA evaluation outputs
└── src/
    ├── prompts/                    # All LLM prompts as plain-text files
    ├── patra_generation/           # Synthetic email generation and preprocessing
    ├── extraction/                 # LLM-based KG extraction
    ├── entity_resolution/          # FAISS blocking, LLM resolution, node fusion
    ├── neo4j/                      # Graph construction and Neo4j ingestion
    └── evaluation/                 # Triple evaluation and KG-QA evaluation
```

---

## Installation

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

---

## Pipeline

### 1. Dataset Generation

Generate synthetic academic email threads using GPT-4.1:

```bash
python src/patra_generation/generate_patra.py \
    --prompt   src/prompts/patra_gen_prompt.txt \
    --output   datasets/PATRA/PATRA.txt \
    --n_threads 50
```

Post-process the raw output into clean, delimited email threads:

```bash
python src/patra_generation/postprocess_patra.py \
    --input  datasets/PATRA/PATRA_raw.txt \
    --output datasets/PATRA/PATRA.txt
```

---
### 2. Triple Extraction

Extract entities and relations from each email with an LLM. Open-source models
(Gemma, LLaMA, Qwen 7B/32B) run locally with in-process vLLM; gpt-oss-20b runs via a
local vLLM OpenAI-compatible server; GPT-5.1 run via the OpenAI API.

```bash
# GPT-5.1
python src/extraction/kg_extraction_pipeline.py \
    --model openai --model_path gpt-5.1 \
    --input_file datasets/PATRA/PATRA.txt \
    --output_dir results/extractions/openai/ \
    --prompt_file src/prompts/extraction_prompt.txt
```

**Arguments**

| Flag | Meaning |
|---|---|
| `--model` | Backend: `openai`, `gptoss`, `llama`, `gemma`, `qwen`, `qwen32b` |
| `--model_path` | HuggingFace model ID (local vLLM) **or** OpenAI model name (e.g. `gpt-5.1`,  `Qwen/Qwen2.5-7B-Instruct`). Optional for aliases with a default (`gptoss` → `openai/gpt-oss-20b`) |
| `--input_file` | Input corpus in PATRA format |
| `--output_dir` | Output root (`entity_extractions/`, `relation_extractions/`, `final_outputs/`) |
| `--prompt_file` | System prompt (default: `src/prompts/extraction_prompt.txt`) |
| `--gpu` | Physical GPU id(s) to pin for local vLLM models (PCI-bus order; comma list for tensor parallelism, e.g. `0,1`) |
| `--tensor_parallel_size` | vLLM tensor-parallel GPUs for local models (default `1`; Qwen2.5-32B fits on one A100 80 GB — only raise this to split across smaller GPUs) |
| `--gpu_memory_utilization` | vLLM GPU memory fraction (default `0.90`) |
| `--max_model_len` | vLLM max context length (default `8192`) |
| `--base_url` | OpenAI-compatible endpoint for a local server (vLLM serving gpt-oss-20b) |
| `--resume` | Skip emails already processed in `--output_dir` |

The OpenAI path auto-handles the gpt-5 family (which requires the default temperature);
local vLLM models use greedy decoding (`temperature=0`) for reproducibility. GPT-5.1 is
the only model accessed via API; all open-source models are served by vLLM
(Llama-3.1-8B-Instruct, Gemma-3-4b-it, Qwen2.5-7B/32B, gpt-oss-20b), run on
A100 (80 GB) / L40S (48 GB) GPUs.

### 3. Entity Resolution

(description + 1 example)

Run the full ER pipeline for a given extraction (FAISS blocking → LLM resolution → node fusion → evaluation):

```bash
cd results/extractions/openai/
bash ../../../src/entity_resolution/run_pipeline.sh openai 0.6547 0
```

**Pipeline steps:**

| Step | Script | Description |
|------|--------|-------------|
| 1 | `faiss_blocking.py` | Semantic candidate blocking with FAISS |
| 2 | `llm_judgement.py` | Qwen2.5-32B match/no-match classification |
| 3 | `node_fusion.py` | Transitive graph fusion + person property patching |
| 4 | `normalize_dates.py` | Date normalisation on fused entities |
| 5 | `evaluate_pipeline.py` | Precision / Recall / F1 against golden set |

**Calibrating the FAISS threshold:**

The auto-reject floor passed to `faiss_blocking.py` / `run_pipeline.sh` is calibrated
from manually annotated candidate pairs (`label` ∈ {`MATCH`, `NO_MATCH`}). For each
similarity cutoff the script sweeps precision/recall and reports the **auto-reject
threshold** (highest score retaining ≥99% recall — pairs below it bypass the LLM and
are auto-rejected) and an **auto-match threshold** (lowest score reaching ≥98%
precision — auto-accepted), if one exists:

```bash
python src/entity_resolution/calibrate_threshold.py \
    --datasets OpenAI=openai_annotated.csv Qwen=qwen32b_annotated.csv \
    --plot_output results/entity_resolution/calibration_plot.png \
    --log_output  results/entity_resolution/calibration_log.txt
```

This reproduces the thresholds used in the paper
([`results/entity_resolution/calibration_log.txt`](results/entity_resolution/calibration_log.txt)):

| Pipeline | Annotated pairs | Auto-reject τ (≥99% recall) | Auto-match (≥98% precision) |
|---|---|---|---|
| **OpenAI** (GPT-5.1) | 497 | **0.6547** | — none reaches 98% precision |
| **Qwen** (32B) | 499 | **0.6627** | — none reaches 98% precision |

No auto-match band exists for either pipeline: no similarity cutoff is precise enough
to safely auto-accept, so every above-floor pair (the "grey zone") is routed to the
Qwen2.5-32B LLM judge. The precision-recall curves and the corresponding thresholds are shown below.

![FAISS threshold calibration](results/entity_resolution/calibration_plot.png)


**Reported results of Entity Resolution:**

| Model | Precision | Recall | F1 |
|-------|-----------|--------|----|
| OpenAI | 0.6538 | 0.5730 | 0.6108 |

The final total number of entities is 7,207, and the number of relations is 16,814.
---

### 4. Neo4j Graph Construction

**Validate** entities and relations against PERKOnto:

```bash
python src/neo4j/clean_kg.py \
    --entities_in  openai_entities_fused_normdates.csv \
    --relations_in openai_relations_fused.csv \
    --entities_out openai_entities_clean.csv \
    --relations_out openai_relations_clean.csv \
    --ontology     ontology/PERKOnto.json
```

**Split** into per-type CSVs for Neo4j ingestion:

```bash
python src/neo4j/prepare_import.py \
    --entities  openai_entities_clean.csv \
    --relations openai_relations_clean.csv \
    --output    datasets/neo4j_import/
```

**Build** the graph:

```bash
python src/neo4j/build_perk.py \
    --data_dir datasets/neo4j_import/ \
    --ontology ontology/PERKOnto.json
```

**Wipe** the graph (before reimport):

```bash
python src/neo4j/wipe_kg.py --dry-run   # preview node count
python src/neo4j/wipe_kg.py             # delete all nodes and relationships
```

Credentials are read from `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD` in `.env`, or passed as `--uri`, `--user`, `--password`.

---

### 5. Evaluation

#### Extraction Evaluation (Sentence-BERT triple metrics)

The golden set is built once by sampling emails and annotating the candidate
triples:

```bash
python src/evaluation/sample_for_annotation.py \
    --entities  results/extractions/openai/entities_final.csv \
    --relations results/extractions/openai/relations_final.csv \
    --emails    datasets/PATRA/PATRA.txt \
    --output    datasets/extraction_gold/golden_set_candidates.csv \
    --n_emails  250
# annotate -> datasets/extraction_gold/refined_golden_set_target.csv
```

For each model, build its `comparison_triples.csv` (resolves entity ids to
label + type, attaches the extracted evidence sentence, and keeps only the
gold-annotated emails):

```bash
python src/evaluation/build_comparison_triples.py \
    --entities  results/extractions/gptoss/final_outputs/entities_final.csv \
    --relations results/extractions/gptoss/final_outputs/relations_final.csv \
    --output    results/extractions/gptoss/evaluation_triples/comparison_triples.csv \
    --source_type GptOss
```

Evaluate with Sentence-BERT — Subject / Object / Total Entities / Relation / Triple
micro P/R/F1 at τ=0.80, for both the source-sentence and full-email context
(per-context logs over the full τ sweep are written to `results/evaluation/`):

```bash
python src/evaluation/evaluate_llm_triples.py \
    --name GptOss \
    --pred results/extractions/gptoss/evaluation_triples/comparison_triples.csv \
    --tau  0.80 --gpu 0
```

Generate result plots (saved to `results/figures/extractions/`):

```bash
python src/evaluation/plot_results.py
```
Extraction quality across models (micro-F1 at τ = 0.80; **Source** = the source
sentence as context, **Full-Email** = the whole email as context):

|  |  |  |
|:---:|:---:|:---:|
| **Subject F1**<br>![Subject F1](results/figures/extractions/fig_sub.png) | **Object F1**<br>![Object F1](results/figures/extractions/fig_obj.png) | **Entity F1**<br>![Entity F1](results/figures/extractions/fig_ent.png) |
| **Relation F1**<br>![Relation F1](results/figures/extractions/fig_pred.png) | **Triple F1**<br>![Triple F1](results/figures/extractions/fig_tri.png) |  |

#### KG-QA Evaluation (PRASHNA-PATRA)

```bash
python src/evaluation/kg_eval.py \
    --model   gpt \
    --input   datasets/PRASHNA_PATRA/PRASHNA_PATRA.csv \
    --ontology ontology/PERKOnto.json
```

The `--model` flag selects which Neo4j instance to query via the corresponding env var prefix (e.g. `--model gpt` reads `GPT_NEO4J_URI`).

---



---
### Citation
Chakraborty, P., Sanyal, D. K., Majumdar, S., & Das, P. P. (2026). prantikaC/PERK: PERK (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.20542115
