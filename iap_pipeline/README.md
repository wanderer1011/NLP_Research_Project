# Intent-Aware Prompting (IAP) Pipeline

A manipulation detection pipeline that implements the **Intent-Aware Prompting** architecture on the [MentalManip](https://github.com/audreycs/MentalManip) dataset.

Instead of asking an LLM to classify a dialogue in a single pass, IAP computationally models "Theory of Mind" by explicitly separating *what is being said* from *why it is being said*, then combining both signals for a final decision.

## Architecture

```
┌─────────────────────────────┐
│  Step 1: Load Dialogue (D)  │
└──────────────┬──────────────┘
               │
       ┌───────┴───────┐
       ▼               ▼
┌──────────────┐ ┌──────────────┐
│  LLM Call 1  │ │  LLM Call 2  │   Step 2: Intent Extraction
│  Intent i_A  │ │  Intent i_B  │   (parallel, independent)
└──────┬───────┘ └──────┬───────┘
       │               │
       └───────┬───────┘
               ▼
┌─────────────────────────────┐
│         LLM Call 3          │   Step 3: Classification
│   r = LLM(D, i_A, i_B, P)  │   r ∈ {0, 1} + technique + confidence
└─────────────────────────────┘
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the dataset

Download `mentalmanip_maj.csv` (or `mentalmanip_con.csv`) from:
https://github.com/audreycs/MentalManip/tree/main/mentalmanip_dataset

Place it in the `iap_pipeline/` directory, or pass its path with `--dataset`:

```
iap_pipeline/
├── mentalmanip_maj.csv        ← put it here
├── main.py
├── ...
```

### 3. Set your API key

**OpenAI (default):**
```powershell
$env:OPENAI_API_KEY = "sk-..."
```

**Anthropic:**
```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

## Usage

```bash
# Quick test with 20 dialogues using the default Groq model
python main.py --dataset mentalmanip_maj.csv --max 20

# Use a stronger model for better accuracy
python main.py --model gpt-4o --max 100

# Use Anthropic Claude
python main.py --provider anthropic --model claude-sonnet-4-20250514 --max 50

# Full dataset run (will use significant API credits)
python main.py --dataset mentalmanip_maj.csv

# Tune concurrency and temperature
python main.py --max 100 --concurrency 10 --temperature 0.1
```

## Output

Results are saved to the `results/` directory:

- **`predictions.csv`** — Per-dialogue predictions with ground truth, predicted label, confidence, detected technique, and explanation.
- **`metrics.json`** — Aggregate metrics: accuracy, precision, recall, F1, and false negative rate.

## Project Structure

| File | Purpose |
|---|---|
| `main.py` | CLI entry point and argument parsing |
| `pipeline.py` | Orchestrates the full 3-step IAP pipeline with async concurrency |
| `intent_extractor.py` | Step 2: Extracts latent intent vectors for each speaker |
| `classifier.py` | Step 3: Final manipulation classification using dialogue + intents |
| `prompts.py` | All system/user prompt templates for Steps 2 and 3 |
| `llm_client.py` | Unified async LLM client (OpenAI / Anthropic) |
| `data_loader.py` | Loads and parses MentalManip CSV files |
| `evaluation.py` | Computes accuracy, precision, recall, F1, FNR metrics |
| `config.py` | Configuration dataclass |

## Key Metrics

The IAP approach targets **false negative rate reduction** — ensuring subtle manipulation is caught without flooding the system with false positives. The research reports up to a 30.5% reduction in false negatives compared to single-pass baselines.