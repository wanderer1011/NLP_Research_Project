# Mental Manipulation Detection Pipelines

This repository contains two Python pipelines for detecting manipulative dialogue in the [MentalManip dataset](https://github.com/audreycs/MentalManip):

- `iap_pipeline/`: intent-aware prompting with configurable OpenAI, Anthropic, and Groq providers.
- `similarity_pipeline/`: sentence-embedding retrieval with threshold tuning and technique-label prediction.

The pipelines are independent and each has its own dependency file and command-line interface.

## Repository Layout

```text
iap_pipeline/
  README.md
  requirements.txt
  *.py
similarity_pipeline/
  requirements.txt
  *.py
P Manipulation-Aware Refusal Mechan.txt
```

Datasets, generated predictions, metrics, embedding caches, and presentation exports are intentionally excluded from Git. Download the MentalManip CSV files separately as described in the pipeline READMEs.

## Quick Start

### Intent-Aware Prompting

```powershell
cd iap_pipeline
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:GROQ_API_KEY = "your-key-here"
python main.py --dataset mentalmanip_maj.csv --max 20
```

Use `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` and select the provider with `--provider` when using those services.

### Similarity Pipeline

```powershell
cd similarity_pipeline
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py --dataset ../iap_pipeline/mentalmanip_maj.csv --max 20
```

The similarity pipeline downloads the sentence-transformers model on its first run.

## Data and Outputs

Download `mentalmanip_maj.csv` or `mentalmanip_con.csv` from the [MentalManip dataset directory](https://github.com/audreycs/MentalManip/tree/main/mentalmanip_dataset), then place the file where the selected command points. Output directories can be changed with each pipeline's `--output` option.

## Research Note

`P Manipulation-Aware Refusal Mechan.txt` contains the accompanying research notes and source context.
