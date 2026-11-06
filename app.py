"""Gradio app - P35 CLIP Text-to-Photo Search.  Run: python app.py"""
import time
import gradio as gr
from PIL import Image
import core

EMB, NAMES = core.load_index()


def _gallery(hits):
    return [(Image.open(core.PHOTO_DIR / n), f"{n}  ({s:.3f})") for n, s in hits]


def run_search(text, example_img, mode, k):
    t0 = time.time()
    if example_img is not None:
        qv = core.encode_images([example_img.convert("RGB")])[0]
        info, gen_prompt = "Search by example image", None
    elif text and text.strip():
        ok, msg = core.check_query(text)
        if not ok:
            return [], msg, None
        qv = core.text_query_vec(text.strip(), mode)
        info, gen_prompt = f"Search by text ({mode})", text.strip()
    else:
        return [], "Type a query or upload an example image.", None
    hits = core.search(qv, EMB, NAMES, int(k))
    ms = (time.time() - t0) * 1000
    gen_img = None
    if hits[0][1] < core.NO_MATCH_THRESHOLD and gen_prompt:
        gen_img = core.generate_image(gen_prompt)
        info += f" | best score {hits[0][1]:.3f} below {core.NO_MATCH_THRESHOLD} -> no good photo match, generated an image instead (AI-generated)"
    return _gallery(hits), f"{info} | {ms:.0f} ms", gen_img


def show_expansion(q):
    ok, msg = core.check_query(q)
    if not ok:
        return msg
    few = core.expand_query(q, "few_shot")
    zero = core.expand_query(q, "zero_shot")
    return f"**Few-shot prompt:** {few}\n\n**Zero-shot prompt:** {zero}"


def show_latent(q):
    import numpy as np
    qs = [x.strip() for x in q.split(",") if x.strip()]
    qv = np.stack([core.text_query_vec(x, "template") for x in qs]) if qs else None
    return core.latent_plot(EMB, NAMES, qv, qs)


with gr.Blocks(title="CLIP Photo Search") as demo:
    gr.Markdown("# CLIP Text-to-Photo Search\nSearch a photo collection by words or by an example image. "
                "Built with CLIP (PyTorch) + LLM query expansion.")
    with gr.Tab("Search"):
        with gr.Row():
            text = gr.Textbox(label="Describe the photo", placeholder="a quiet street in the rain")
            img = gr.Image(type="pil", label="...or upload an example image")
        with gr.Row():
            mode = gr.Radio(["raw", "template", "expanded"], value="expanded", label="Query mode")
            k = gr.Slider(3, 12, value=5, step=1, label="Results")
        btn = gr.Button("Search")
        gal = gr.Gallery(label="Top matches", columns=5)
        info = gr.Markdown()
        gen = gr.Image(label="Generated image (only when no photo matches)")
        btn.click(run_search, [text, img, mode, k], [gal, info, gen])
    with gr.Tab("Query expansion (LLM)"):
        q2 = gr.Textbox(label="Vague query", value="cozy")
        out2 = gr.Markdown()
        gr.Button("Expand").click(show_expansion, q2, out2)
    with gr.Tab("Latent space"):
        q3 = gr.Textbox(label="Comma-separated queries", value="a dog, a beach, a city at night")
        plot = gr.Plot()
        gr.Button("Plot").click(show_latent, q3, plot)
    with gr.Tab("Responsible AI"):
        gr.Markdown(open("MODEL_CARD.md").read())

if __name__ == "__main__":
    demo.launch(share=True)
