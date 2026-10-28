# LLM Comparison on Multiple-Choice Questions (Proprietary Models)

Query **proprietary / API-based LLMs** (OpenAI, Anthropic, DeepSeek, xAI, Mistral,
and optionally Google) on a questionnaire of German environmental-awareness
multiple-choice and numeric questions, under different **role / perspective
prompts**, and export the answers to a tidy Excel workbook.

This repository is the API-model counterpart to the companion open-source-model
framework and reproduces the proprietary-model results reported in the paper:

- **Paper:** <https://arxiv.org/pdf/2606.02741>
- **Open-source-model benchmarking framework** (self-hosted models via vLLM):
  <https://gitlab.opencode.de/uba-ki-lab/llm-questionnaire-benchmarking-framework> (MIT)
- **Published dataset** (human + LLM answers across roles, CC-BY-4.0):
  bundled under [`data/zenodo_data/`](data/zenodo_data/)

The two code repositories are complementary: this one runs the closed, API-only
models; the framework above runs open-weight models locally. Both write results
in the **same per-role Excel layout**, so the workbooks can simply be combined.

## Built on aisuite

All model calls go through [**aisuite**](https://github.com/andrewyng/aisuite),
the open-source unified LLM interface by Andrew Ng and contributors. aisuite
exposes a single OpenAI-style client (`ai.Client().chat.completions.create(...)`)
that talks to every provider behind one `"<provider>:<model>"` string, so adding
or swapping a model here is just a one-line edit in `MODEL_LIST` — no
provider-specific SDK code. This repository is essentially a thin, questionnaire-
specific wrapper around aisuite (prompt building, bootstrapping, parsing and
Excel export).

## What it does

[`run_llm_eval.py`](run_llm_eval.py) is self-contained. For every
`(model × role)` combination it:

1. reads the questions from `data/Fragen_Umweltbewusstsein.xlsx` (sheet `Deutsch_v3`),
   using the `Example`-rows as few-shot examples;
2. sends each question `N_BOOTSTRAP` times (default 3) at `temperature=0` and takes
   the **median** parsed answer, with retries and provider-level parallelism;
3. writes, into the configured `OUTPUT_DIR`:
   - `predictions_<model>_<role>.csv` — per-question raw replies, parsed values,
     median, and consistency flag;
   - `sample_prompts_<model>_<role>.txt` — the exact prompts sent (traceability);
   - `summary.csv` — accuracy / MAE / coverage per model (on scored items);
   - **`all_results.xlsx` — one sheet per role, questions × models** (this is the
     combinable workbook).

A complete reference run is shipped in [`example_output/`](example_output/)
(5 models × 11 roles) so you can inspect the output format without spending API
credits.

## Setup

```bash
# 1. Create the environment (minimal: aisuite, pandas, numpy, openpyxl, dotenv)
conda env create -f environment.yml
conda activate green-llm-eval

# 2. Configure API keys
cp .env.example .env
#   then edit .env and fill in keys for the providers you want to query
```

**You need your own API key for every provider you query** (OpenAI, Anthropic,
DeepSeek, xAI, Mistral, …). You only need keys for the providers that actually
appear in `MODEL_LIST` — querying a model bills your account with that provider.
Create the keys in each provider's developer console and paste them into `.env`.

Because aisuite handles the provider connections, the authentication is set up
exactly as aisuite documents it. See the per-provider setup guides for how to
obtain and configure each credential:

- aisuite provider guides: <https://github.com/andrewyng/aisuite/tree/main/guides>

> **Google / Gemini** is the fiddly one: aisuite reaches Gemini through Google
> Cloud Vertex AI, which needs a **service account** (`GOOGLE_PROJECT_ID`,
> `GOOGLE_REGION` and a `GOOGLE_APPLICATION_CREDENTIALS` JSON file) rather than a
> simple API key. Follow aisuite's Google guide
> (<https://github.com/andrewyng/aisuite/blob/main/guides/google.md>) and keep the
> service-account JSON **outside** version control. Gemini is disabled by default
> in `MODEL_LIST`.

## Running

Configuration lives in the constants at the top of [`run_llm_eval.py`](run_llm_eval.py):

- `MODEL_LIST` — which models to query (uncomment the ones you want);
- `ROLLEN` — the role/perspective prompts prepended to each question;
- `OUTPUT_DIR` — where results are written;
- `N_BOOTSTRAP`, `TEMPERATURE`, `MAX_TOKENS`, rate-limit and retry settings.

Then run:

```bash
python run_llm_eval.py
```

or, for a long unattended run with logging and crash-survival:

```bash
./run_overnight.sh
```

## Combining with open-source-model results

Both repositories emit `all_results.xlsx` with one sheet per role and the same
`question × model` layout. To build the full comparison, copy the model columns
from the open-source-model workbook into the matching role sheets of this one
(or vice versa).

## Analysis notebook

[`notebooks/1_compare_rollen.ipynb`](notebooks/1_compare_rollen.ipynb) reproduces
the role-comparison analysis of the paper: the 2×3 persona-prompting figure, the
role-sensitivity metric tables and the metric heatmap. It reads the combined
`role × model` dataset bundled under [`data/zenodo_data/`](data/zenodo_data/) and
runs out of the box (the notebook dependencies are part of `environment.yml`):

```bash
conda activate green-llm-eval
jupyter notebook notebooks/1_compare_rollen.ipynb
```

Generated figures (`rollen_paper_2x3.*`) and metric tables (`rollen_metrics*.xlsx`)
are written next to the notebook and are git-ignored.

## Data & licensing

- **Code** in this repository: MIT (see [`LICENSE`](LICENSE)).
- **`data/Fragen_Umweltbewusstsein.xlsx`** — the question set (input).
- **`data/zenodo_data/`** — the published companion dataset
  *"Comparative Evaluation of LLM Responses Across Role and Perspective
  Conditions"* by Kunkel, Hartwig, Voss, Schütt & Gellrich, licensed
  **CC-BY-4.0** (see its own `README.txt`). It is reused here under attribution.

## Citation

If you use this code or data, please cite the paper (<https://arxiv.org/pdf/2606.02741>)
and the Zenodo dataset.
