# 🪔 Sanskriti-T2I

**Culturally grounded Text-to-Image generation for authentic and context-aware Indian cultural representations.**

Sanskriti-T2I turns a short query in *any Indian language* (or English) into a rich, culturally accurate, structured prompt for image generators such as Stable Diffusion or FLUX.

> Based on our work *"Retrieval-Augmented Multi-Agent Prompt Enrichment for Culturally Grounded Text-to-Image Generation in Indian Languages"*, accepted at **CARE Conference 2026**, IIT Roorkee (3–5 December 2026).

---

## Why this project?

Text-to-image models often fall back on generic or Westernized visuals when given prompts about Indian festivals, rituals, attire, art forms, or regional traditions, especially when the prompt is written in an Indian language. A short query like *"छठ पूजा अर्घ्य देती महिला"* carries cultural detail the image model does not know to draw.

Sanskriti-T2I fixes this **at the prompt level**. It retrieves real cultural context, filters it down to what is visually relevant, and writes a structured prompt, so any off-the-shelf image model gets the details it needs. No image-model retraining is required.

## How it works

```
User query (any Indian language / English)
        │
        ▼
  IndicBERT understanding  ─►  English concept
        │
        ▼
  Cultural web retrieval (Wikipedia, web sources)
        │
        ▼
  Context filtering with IndicBERT (keeps visual details only)
        │
        ▼
  Structured prompt + negative prompt
        │
        ▼
  Stable Diffusion / FLUX-ready output
```

1. **Query understanding:** the input is processed with IndicBERT and mapped to an English concept.
2. **Cultural retrieval:** relevant context is gathered from web and Wikipedia sources.
3. **Context filtering:** IndicBERT-based filtering extracts visual categories (e.g., attire, setting, objects) and drops irrelevant text, keeping the prompt token-efficient.
4. **Prompt construction:** the filtered details are assembled into structured prompt blocks plus a recommended negative prompt.

## Features

- 🌐 Multilingual input: Hindi, Bengali, Tamil, Telugu, Marathi, other Indian languages, and English
- 🔎 Retrieval-grounded cultural context with visible sources
- 🧹 Context filtering for token-efficient prompts
- 🎯 Structured prompt and negative prompt, ready for Stable Diffusion / FLUX
- 🖥️ Three ways to use it: CLI, Gradio web UI, and FastAPI REST API

## Repository structure

```
Sanskriti-T2I/
├── src/                # Core pipeline (CulturalPromptPipeline)
├── tests/              # Tests
├── main.py             # Command-line interface
├── app.py              # Gradio web UI
├── api.py              # FastAPI REST API
├── sd_input.py         # Helper: fetch prompts from the API for Stable Diffusion
├── requirements.txt    # Python dependencies
├── 1.png               # Project image
└── Nishant_and_Anuj_report_with_citations.pdf   # Project report
```

## Tech stack

Python · PyTorch · Hugging Face Transformers (IndicBERT) · Sentence-Transformers · LangChain · scikit-learn · Wikipedia API · BeautifulSoup · deep-translator · Gradio · FastAPI

## Installation

```bash
git clone https://github.com/nishant-0411/Sanskriti-T2I.git
cd Sanskriti-T2I

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

An internet connection is needed for web retrieval and for downloading model weights on first run.

## Usage

### 1. Command line

```bash
# Output only the final structured prompt
python main.py --query "छठ पूजा अर्घ्य देती महिला"

# Show the full pipeline (English concept, sources, filtered categories, prompt blocks)
python main.py --query "Kathakali dancer" --verbose
```

### 2. Web UI (Gradio)

```bash
python app.py
```

Open **http://127.0.0.1:7860**. Enter a query, or pick one of the sample queries. The UI shows the structured prompt, the negative prompt, and an inspection panel with the filtered visual details and the raw retrieved sources.

### 3. REST API (FastAPI)

```bash
python -m uvicorn api:app --reload
```

Interactive docs: **http://127.0.0.1:8000/docs**

```bash
curl -X POST http://127.0.0.1:8000/generate-prompt \
  -H "Content-Type: application/json" \
  -d '{"query": "பொங்கல் திருநாள் கொண்டாட்டம்", "verbose": true}'
```

**Response fields:** `status`, `query`, `english_concept`, `structured_prompt`, `negative_prompt`, `prompt_blocks`, and (when `verbose` is true) `filtered_visual_categories` and `retrieved_sources`.

### 4. Feed an image model

With the API running, `sd_input.py` provides `get_sd_prompts_from_api(query)`, which returns `(structured_prompt, negative_prompt)` ready to pass to your Stable Diffusion / FLUX pipeline.

```python
from sd_input import get_sd_prompts_from_api

prompt, negative_prompt = get_sd_prompts_from_api("দুর্গাপূজায় ধুনুচি নাচ")
```

## Example queries

| Language | Query |
|----------|-------|
| Hindi | छठ पूजा अर्घ्य देती महिला |
| Bengali | দুর্গাপূজায় ধুনুচি নাচ |
| Tamil | பொங்கல் திருநாள் கொண்டாட்டம் |
| Telugu | సంక్రాంతి భోగి మంటలు |
| Marathi | गुढीपाडवा मिरवणूक |
| English | Kathakali dancer expressing Navarasa |
| English | Warli tribal celebration with Tarpa dance |

## Results

Image generated from a structured prompt produced by Sanskriti-T2I:

**Prompt:**

```
During Durga Puja celebrations in Kolkata, an elderly man in a white dhoti and kurta
offered prayers near a brightly lit pandal decorated with marigold garlands and fairy lights
```

![Durga Puja pandal with a dhak player, generated from a Sanskriti-T2I prompt](Results\run_2.png)

## Report

The full write-up with methodology and citations is in [`Nishant_and_Anuj_report_with_citations.pdf`](Nishant_and_Anuj_report_with_citations.pdf).

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{lastname2026sanskriti,
  title     = {Retrieval-Augmented Multi-Agent Prompt Enrichment for Culturally Grounded Text-to-Image Generation in Indian Languages},
  author    = {Lastname, Nishant and Lastname, Anuj},
  booktitle = {Proceedings of the CARE Conference 2026},
  address   = {IIT Roorkee, India},
  month     = dec,
  year      = {2026},
  note      = {Submission ID: CARE-26-117},
  url       = {https://github.com/nishant-0411/Sanskriti-T2I}
}
```

## Authors

- **Nishant** ([@nishant-0411](https://github.com/nishant-0411))
- **Anuj** ([@anujr-shri](https://github.com/anujr-shri))