# Model Card - CLIP Text-to-Photo Search (P35)

**Models:** OpenAI CLIP ViT-B/32 (retrieval, MIT-licensed weights via Hugging Face); all-MiniLM-L6-v2 (BERT-family, Apache-2.0) to filter drifting query expansions; Llama-3.1-8B-Instant via Groq (GPT-style LLM, Meta Llama licence) for query expansion; sd-turbo (Stability AI community licence - check terms) for the optional generated fallback image.

**Intended use:** educational search over a small personal photo collection. Not for identifying people, surveillance, or moderation decisions.

**Data:** 200+ photos in the repo (state source and licence of every photo in README; use your own photos or openly licensed ones).

**Known limitations:** CLIP reflects web-scale biases (e.g. occupation and gender/appearance associations), struggles with counting, text in images, fine spatial relations and negation ("no people"). Scores are relative; the no-match threshold must be tuned per collection.

**Guardrails:** query filter blocks explicit-content and person-identification requests; generated images are labelled AI-generated; API keys held only in platform secrets.

**Measured results:** see report (hit@5 per mode, bias proportions, failure cases).
