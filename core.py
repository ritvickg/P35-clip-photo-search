"""Core logic for P35 - CLIP Text-to-Photo Search (PyTorch).

U1 latent space   -> latent_plot()
U2 LLM expansion  -> expand_query()  (GPT-style LLM + BERT-family MiniLM drift filter)
U3 CLIP retrieval -> encode_text / encode_image / search
U4 generation     -> generate_image()
U5 guardrail      -> check_query()
"""
import json, os, re, time
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

ROOT = Path(__file__).parent
   PHOTO_DIR = ROOT / "photos" if (ROOT / "photos" / "img_0000.jpg").exists() else ROOT
EMB_PATH = ROOT / "assets" / "embeddings.npy"
META_PATH = ROOT / "assets" / "paths.json"

CLIP_NAME = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
BERT_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # BERT-family (MiniLM distilled from BERT-style encoders)
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")  # GPT-style decoder LLM served by Groq
NO_MATCH_THRESHOLD = float(os.getenv("NO_MATCH_THRESHOLD", "0.22"))  # CLIP cosine; tune on your photo set

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_clip = _proc = _bert = None


def _load_clip():
    global _clip, _proc
    if _clip is None:
        _clip = CLIPModel.from_pretrained(CLIP_NAME).to(DEVICE).eval()
        _proc = CLIPProcessor.from_pretrained(CLIP_NAME)
    return _clip, _proc


def _load_bert():
    global _bert
    if _bert is None:
        from sentence_transformers import SentenceTransformer
        _bert = SentenceTransformer(BERT_NAME, device=DEVICE)
    return _bert


# ---------------- encoders (shared text-image latent space) ----------------
@torch.no_grad()
def encode_text(texts):
    m, p = _load_clip()
    t = p(text=list(texts), return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    e = m.text_projection(m.text_model(**t).pooler_output)
    return torch.nn.functional.normalize(e, dim=-1).cpu().numpy()


@torch.no_grad()
def encode_images(images, batch=32):
    m, p = _load_clip()
    out = []
    for i in range(0, len(images), batch):
        x = p(images=images[i:i + batch], return_tensors="pt").to(DEVICE)
        e = m.visual_projection(m.vision_model(pixel_values=x['pixel_values']).pooler_output)
        out.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
    return np.concatenate(out)


def build_index():
    files = sorted(p for p in PHOTO_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    imgs = [Image.open(f).convert("RGB") for f in files]
    emb = encode_images(imgs)
    EMB_PATH.parent.mkdir(exist_ok=True)
    np.save(EMB_PATH, emb)
    META_PATH.write_text(json.dumps([f.name for f in files]))
    return len(files)


def load_index():
    return np.load(EMB_PATH), json.loads(META_PATH.read_text())


# ---------------- search ----------------
def search(query_vec, emb, names, k=5):
    sims = emb @ query_vec
    idx = np.argsort(-sims)[:k]
    return [(names[i], float(sims[i])) for i in idx]


def text_query_vec(query, mode="template"):
    """mode: raw | template (prompt-based: 'a photo of ...') | expanded (LLM, averaged)"""
    if mode == "raw":
        v = encode_text([query])[0]
    elif mode == "template":
        v = encode_text([f"a photo of {query}"])[0]
    else:
        variants = [query] + expand_query(query)
        v = encode_text([f"a photo of {x}" for x in variants]).mean(0)
    return v / np.linalg.norm(v)


# ---------------- U2: LLM query expansion ----------------
FEW_SHOT = (
    "Rewrite a vague photo-search query as 3 short, concrete visual descriptions.\n"
    "Return only a JSON list of 3 strings. Do not add people, brands or objects the query does not imply.\n"
    'Query: "cozy" -> ["a warm living room with a blanket and lamp", "a cup of tea by a window", "a candle on a wooden table"]\n'
    'Query: "{q}" ->'
)
ZERO_SHOT = 'Give 3 short, concrete visual descriptions for the photo search query "{q}". Return only a JSON list of strings.'


def _call_llm(prompt):
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return None
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": LLM_MODEL, "temperature": 0.3, "max_tokens": 200,
              "messages": [{"role": "user", "content": prompt}]},
        timeout=20,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def expand_query(q, prompt_style="few_shot", min_sim=0.35):
    """LLM expansion; BERT-family MiniLM drops expansions that drift away from the query."""
    prompt = (FEW_SHOT if prompt_style == "few_shot" else ZERO_SHOT).replace("{q}", q)
    try:
        raw = _call_llm(prompt)
        m = re.search(r"\[.*\]", raw or "", re.S)
        cands = json.loads(m.group(0)) if m else []
    except Exception:
        cands = []
    if not cands:  # offline fallback so the app never breaks
        cands = [f"{q}, outdoors", f"{q}, close-up", f"{q}, wide shot"]
    bert = _load_bert()
    e = bert.encode([q] + cands, normalize_embeddings=True)
    keep = [c for c, s in zip(cands, e[1:] @ e[0]) if s >= min_sim]
    return keep or cands[:1]


# ---------------- U4: generate when no photo matches ----------------
def generate_image(prompt):
    """sd-turbo via HF Inference API (token in secret HF_TOKEN). Returns PIL image or None."""
    tok = os.getenv("HF_TOKEN")
    if not tok:
        return None
    try:
        from huggingface_hub import InferenceClient
        return InferenceClient(token=tok).text_to_image(prompt, model=os.getenv("GEN_MODEL", "stabilityai/sd-turbo"))
    except Exception:
        return None


# ---------------- U5: guardrail ----------------
BLOCK_PATTERNS = [
    r"\b(nude|naked|nsfw|porn|explicit|gore|beheading)\b",
    r"\b(who is (this|that)|identify (this|the) person|find (photos|pictures) of (my )?(ex|neighbou?r|girlfriend|boyfriend))\b",
    r"\b(child|kid|minor|underage)\b.*\b(nude|naked|sexy)\b",
]


def check_query(q):
    """Returns (ok, message). Blocks explicit content and person-identification / stalking style queries."""
    for pat in BLOCK_PATTERNS:
        if re.search(pat, q.lower()):
            return False, "This search was blocked: it asks for explicit content or for identifying/tracking a person."
    return True, ""


# ---------------- U1: latent-space view ----------------
def latent_plot(emb, names, query_vecs=None, labels=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2).fit(emb)
    pts = pca.transform(emb)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(pts[:, 0], pts[:, 1], s=14, alpha=0.5, label="photos (image embeddings)")
    if query_vecs is not None:
        q = pca.transform(np.atleast_2d(query_vecs))
        ax.scatter(q[:, 0], q[:, 1], marker="*", s=220, c="red", label="text queries")
        for (x, y), l in zip(q, labels or []):
            ax.annotate(l, (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_title("CLIP shared text-image space (PCA of 512-d)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
