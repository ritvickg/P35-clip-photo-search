> **Live app:** https://p35-clip-photo-search-yro8k7w72bkodxuuacxzgb.streamlit.app/ (deployed on Streamlit Community Cloud).
> Run locally with `streamlit run streamlit_app.py` (Streamlit version) or `python app.py` (Gradio version).
> Evaluation: `python eval_auto.py` and `python bias_auto.py` (relevance and gender labels are derived automatically from the Flickr8k captions).
> Photos: 240 images from the Flickr8k dataset, educational use. Optional keys: `GROQ_API_KEY` (LLM query expansion), `HF_TOKEN` (image generation).

# P35 - CLIP Text-to-Photo Search
Generative AI CA mini project. Search 200+ photos by words or by an example image.

| Unit | Where |
|---|---|
| U1 shared latent space, prompt comparison | `Latent space` tab, `core.latent_plot`, raw vs template vs expanded |
| U2 LLM query expansion (few-shot vs zero-shot, MiniLM drift filter) | `core.expand_query` |
| U3 CLIP indexing and cross-modal retrieval | `build_index.py`, `core.search` |
| U4 image generation when no photo matches | `core.generate_image` (sd-turbo) |
| U5 bias test, guardrail, licence note, model card | `bias_test.py`, `core.check_query`, `MODEL_CARD.md` |

## Run locally
```
pip install -r requirements.txt
python prepare_photos.py <your_photo_folder>   # resizes to 512px into photos/
python build_index.py
export GROQ_API_KEY=...   # optional: LLM expansion (falls back to templates)
export HF_TOKEN=...       # optional: image generation
python app.py
```
## Evaluate
```
python eval.py make    # label eval/review.csv with 1/0
python eval.py score
python bias_test.py    # label eval/bias_review.csv, run again
```
## Deploy (Hugging Face Spaces, Gradio SDK)
Create a Space, push this folder, add `GROQ_API_KEY` and `HF_TOKEN` under Settings -> Secrets. Commit `assets/embeddings.npy` and `assets/paths.json`.

## Photo sources and licences
(list them here)
