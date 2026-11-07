"""Streamlit version of the P35 app (free hosting on Streamlit Community Cloud)."""
import os, time
import numpy as np
import streamlit as st
from PIL import Image

for k in ("GROQ_API_KEY", "HF_TOKEN"):
    try:
        if k in st.secrets:
            os.environ[k] = st.secrets[k]
    except Exception:
        pass
import core

st.set_page_config(page_title="CLIP Photo Search", layout="wide")


@st.cache_resource
def load():
    return core.load_index()


EMB, NAMES = load()
st.title("CLIP Text-to-Photo Search")
st.caption("Search a photo collection by words or by an example image. CLIP (PyTorch) + LLM query expansion.")
t1, t2, t3, t4 = st.tabs(["Search", "Query expansion (LLM)", "Latent space", "Responsible AI"])

with t1:
    text = st.text_input("Describe the photo", placeholder="a quiet street in the rain")
    up = st.file_uploader("...or upload an example image", type=["jpg", "jpeg", "png"])
    c1, c2 = st.columns(2)
    mode = c1.radio("Query mode", ["raw", "template", "expanded"], index=2, horizontal=True)
    k = c2.slider("Results", 3, 12, 5)
    if st.button("Search"):
        t0 = time.time(); gen_prompt = None
        if up is not None:
            qv = core.encode_images([Image.open(up).convert("RGB")])[0]; info = "Search by example image"
        elif text.strip():
            ok, msg = core.check_query(text)
            if not ok:
                st.error(msg); st.stop()
            qv = core.text_query_vec(text.strip(), mode); info = f"Search by text ({mode})"; gen_prompt = text.strip()
        else:
            st.warning("Type a query or upload an image."); st.stop()
        hits = core.search(qv, EMB, NAMES, k)
        st.write(f"{info} | {(time.time() - t0) * 1000:.0f} ms")
        cols = st.columns(min(5, len(hits)))
        for i, (n, s) in enumerate(hits):
            cols[i % len(cols)].image(str(core.PHOTO_DIR / n), caption=f"{n} ({s:.3f})")
        if hits[0][1] < core.NO_MATCH_THRESHOLD and gen_prompt:
            st.info(f"Best score {hits[0][1]:.3f} is below {core.NO_MATCH_THRESHOLD}: no good photo match.")
            g = core.generate_image(gen_prompt)
            if g is not None:
                st.image(g, caption="AI-generated image (no photo matched)")

with t2:
    q = st.text_input("Vague query", value="cozy")
    if st.button("Expand"):
        ok, msg = core.check_query(q)
        if not ok:
            st.error(msg)
        else:
            st.write("**Few-shot prompt:**", core.expand_query(q, "few_shot"))
            st.write("**Zero-shot prompt:**", core.expand_query(q, "zero_shot"))

with t3:
    q3 = st.text_input("Comma-separated queries", value="a dog, a beach, a city at night")
    if st.button("Plot"):
        qs = [x.strip() for x in q3.split(",") if x.strip()]
        qv = np.stack([core.text_query_vec(x, "template") for x in qs]) if qs else None
        st.pyplot(core.latent_plot(EMB, NAMES, qv, qs))

with t4:
    st.markdown(open("MODEL_CARD.md").read())
