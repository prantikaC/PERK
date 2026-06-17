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
| **Archived release (Zenodo, all versions)** | <https://doi.org/10.5281/zenodo.20542114> |
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

Run the full ER pipeline (FAISS blocking → LLM resolution → node fusion → evaluation) for
a given extraction. The third argument pins the GPU (default 0):

```bash
cd results/extractions/openai/
bash ../../../src/entity_resolution/run_pipeline.sh openai 0.6547 0
```

| Step | Script | Description |
|------|--------|-------------|
| 1 | `faiss_blocking.py` | Semantic candidate blocking with FAISS |
| 2 | `llm_judgement.py` | Match / no-match judging (Qwen2.5-32B via vLLM, or `--backend openai`) |
| 3 | `node_fusion.py` | Transitive graph fusion + person property patching |
| 4 | `normalize_dates.py` | Date normalisation on fused entities |
| 5 | `evaluate_pipeline.py` | Precision / Recall / F1 against the golden set |

**What it does.** The same real-world entity often surfaces under different wording across
emails; ER collapses these duplicates into one node. For example, the extractor produced two
separate `Method` nodes, *"OCR error correction module"* and *"OCR correction pipeline"*.
FAISS blocking flags the pair (cosine similarity **0.74** — above the **0.6547** auto-reject
floor, so it enters the LLM grey zone); the Qwen2.5-32B judge labels it **MATCH**; and
`node_fusion.py` merges them into a single `Method` node, re-pointing every `usedFor` /
`uses` / `evaluates` relation from both onto that one id. Pairs scoring below the floor are
auto-rejected without an LLM call.

The FAISS auto-reject floor is calibrated from manually annotated candidate pairs. Pairs
below it are auto-rejected; everything above (the "grey zone") is routed to the LLM judge.

| Pipeline | Annotated pairs | Auto-reject τ (≥99% recall) | Auto-match (≥98% precision) |
|---|---|---|---|
| OpenAI (GPT-5.1) | 497 | **0.6547** | none reaches 98% precision |
| Qwen (32B) | 499 | **0.6627** | none reaches 98% precision |

![FAISS threshold calibration](results/entity_resolution/calibration_plot.png)

To recompute it:

```bash
python src/entity_resolution/calibrate_threshold.py \
    --datasets OpenAI=openai_annotated.csv Qwen=qwen32b_annotated.csv \
    --plot_output results/entity_resolution/calibration_plot.png \
    --log_output  results/entity_resolution/calibration_log.txt
```

End-to-end ER (OpenAI pipeline): Precision **0.6538**, Recall **0.5730**, F1 **0.6108**.

---

### 4. Neo4j Graph Construction

A Neo4j graph is constructed for a total number of 7,207 entities and 16,814 relations.

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

**Extraction evaluation** is model-agnostic — set `MODEL` to any extraction run
(`openai`, `gptoss`, `llama`, `gemma`, `qwen7b`, `qwen32b`) and `NAME` to its display
label, then run the same three steps:

```bash
MODEL=gpt-oss         # directory under results/extractions/
NAME=GptOss           # label used in logs/plots and the comparison CSV

# 1) Build that model's comparison triples (ids -> label+type, attach evidence)
python src/evaluation/build_comparison_triples.py \
    --entities  results/extractions/$MODEL/final_outputs/entities_final.csv \
    --relations results/extractions/$MODEL/final_outputs/relations_final.csv \
    --output    results/extractions/$MODEL/comparison_triples.csv \
    --source_type $NAME

# 2) Score with Sentence-BERT (Subject/Object/Entity/Relation/Triple micro-P/R/F1)
python src/evaluation/evaluate_llm_triples.py \
    --name $NAME \
    --pred results/extractions/$MODEL/comparison_triples.csv \
    --tau  0.80 --gpu 0

# 3) After all models are scored, generate the comparison plots
python src/evaluation/plot_results.py        # writes results/figures/extractions/
```

To evaluate every model, loop over the runs:

```bash
declare -A MODELS=( [openai]=OpenAI [gpt-oss]=GptOss [llama]=Llama
                    [gemma]=Gemma [qwen7b]=Qwen7B [qwen32b]=Qwen32B )
for MODEL in "${!MODELS[@]}"; do
  NAME=${MODELS[$MODEL]}
  python src/evaluation/build_comparison_triples.py \
      --entities  results/extractions/$MODEL/final_outputs/entities_final.csv \
      --relations results/extractions/$MODEL/final_outputs/relations_final.csv \
      --output    results/extractions/$MODEL/comparison_triples.csv \
      --source_type $NAME
  python src/evaluation/evaluate_llm_triples.py \
      --name $NAME \
      --pred results/extractions/$MODEL/comparison_triples.csv \
      --tau 0.80 --gpu 0
done
python src/evaluation/plot_results.py
```

**KG-QA** over PRASHNA-PATRA (`--model` selects the Neo4j instance via its env-var prefix,
e.g. `gpt` → `GPT_NEO4J_URI`):

```bash
python src/evaluation/kg_eval.py \
    --model gpt \
    --input datasets/PRASHNA_PATRA/PRASHNA_PATRA.csv \
    --ontology ontology/PERKOnto.json
```

**No-KG baseline** — feed the whole PATRA corpus to a long-context LLM and ask each
question directly (same GPT-5.1 judge as `kg_eval.py`, so the accuracy is comparable):

```bash
python src/evaluation/llm_QA_on_PATRA.py \
    --corpus datasets/PATRA/PATRA.txt \
    --qa     datasets/PRASHNA_PATRA/PRASHNA_PATRA.csv \
    --output results/qa/longcontext_qa_results.csv \
    --fig    results/figures/qa/longcontext_qa_baseline.png
```

Extraction quality uses Sentence-BERT entity embeddings (max-pooled with context, Hungarian
alignment at τ = 0.80) for Subject / Object / Entity / Relation / Triple micro-P/R/F1, in
both source-sentence and full-email context modes.

---

## 6. Results

### 6.1 Extraction quality

Open-source models perform poorly. GPT-5.1 was therefore chosen for the final extraction. 

#### Comparison to alternative extractors

| Approach | Result |
|---|---|
| **Stanford OpenIE** (schema-free) | 0% of relations align with PERKOnto |
| **KGGen** (DSPy / GPT-4o, schema-free) | 0.4% of relations align with PERKOnto |

Schema-free and direct-prompting approaches confirm that **ontology constraints are
essential**; without them, the extraction task is ill-defined.

### 6.2 KG-QA (PRASHNA-PATRA)

Schema-guided KBQA over the constructed graph reaches **75.5%** accuracy (151/200)

**Does the knowledge graph beat brute-force long context?** As a no-KG baseline, we feed
the entire PATRA corpus (~570K tokens) to a long-context LLM (GPT-4.1) and ask each
question directly, scored by the *same* GPT-5.1 judge as `kg_eval.py`
([`src/evaluation/llm_QA_on_PATRA.py`](src/evaluation/llm_QA_on_PATRA.py)). It reaches
**55.0%** overall (110/200) — a strong showing for raw long context, but still well short
of the KG across every question type:

| QA accuracy | Overall | single-hop | multi-hop | reasoning | not available |
|---|:--:|:--:|:--:|:--:|:--:|
| **PERK KG-QA** (`kg_eval.py`) | **75.5%** | 72.3% | 74.6% | 75.0% | **84.6%** |
| Long-context GPT-4.1 (no KG) | 55.0% | 59.6% | 52.2% | 70.0% | 19.2% |


The KG wins by **~20 points overall** and is most decisive on **multi-hop** (74.6% vs
52.2%) and on **"not available"** questions (84.6% vs 19.2%): with the whole mailbox in
context the long-context model hallucinates answers instead of recognising when a fact is
absent, whereas the graph's structure makes missing information explicit. This confirms the
value of explicit graph construction over brute-force long-context retrieval.

---
### Citation
> Chakraborty, P., Sanyal, D. K., Majumdar, S., & Das, P. P. (2026). *prantikaC/PERK: PERK*.
> Zenodo. <https://doi.org/10.5281/zenodo.20542114>
