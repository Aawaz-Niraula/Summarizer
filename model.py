import streamlit as st
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# --- Streamlit Page Config ---
st.set_page_config(page_title="📝 Flashy T5 Summarizer", layout="centered", page_icon="✍️")

# --- CSS for flashiness ---
st.markdown(
    """
    <style>
    .summary-box {
        background-color: #1e3a8a;  /* Dark blue */
        color: white;               /* White text */
        border-radius: 15px;
        padding: 20px;
        font-size: 16px;
        line-height: 1.6;
        white-space: pre-wrap;      /* preserve line breaks */
    }
    .input-box {
        background-color: #fff7e6;
        border-radius: 15px;
        padding: 20px;
        font-size: 16px;
    }
    .note {
        color: #0a66c2;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Header ---
st.title("✍️ T5-small Summarizer")
st.markdown(
    '<p class="note">💡 Recommended summary length: 25–35% of original paragraph. T5-small works best for moderately sized summaries.</p>',
    unsafe_allow_html=True
)

# --- Load Model ---
@st.cache_resource
def load_model():
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()

# --- Input Text ---
with st.container():
    st.markdown('<div class="input-box">', unsafe_allow_html=True)
    text_input = st.text_area("Paste your text here:", height=250)
    st.markdown('</div>', unsafe_allow_html=True)

# --- Suggested Summary Length ---
input_word_count = len(text_input.split())
suggested_min = max(20, int(input_word_count * 0.25))
suggested_max = max(30, int(input_word_count * 0.35))

summary_length = st.slider(
    "Desired summary length (words):",
    min_value=suggested_min,
    max_value=suggested_max,
    value=int((suggested_min + suggested_max) / 2),
    step=1
)
output_format = st.radio("Output format:", ["Paragraph", "Bullet Points"], horizontal=True)

# --- Summarize Button ---
if st.button("✨ Summarize"):
    if not text_input.strip():
        st.warning("Please enter some text to summarize.")
    else:
        # --- Chunking ---
        max_input_words = 100
        step = max_input_words // 4
        words = text_input.split()
        chunks = [words[i:i+max_input_words] for i in range(0, len(words), step)]

        chunk_summaries = []
        with st.spinner("Generating summary..."):
            for chunk in chunks:
                chunk_text = "summarize: " + " ".join(chunk)
                inputs = tokenizer(chunk_text, max_length=1024, truncation=True, return_tensors="pt").to(device)
                target_tokens = int(summary_length * 1.3)
                summary_ids = model.generate(
                    inputs["input_ids"],
                    num_beams=4,
                    max_length=target_tokens + 20,
                    min_length=max(20, target_tokens - 20),
                    length_penalty=3.0,
                    early_stopping=True,
                )
                chunk_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                chunk_summaries.append(chunk_summary)

        raw_summary = " ".join(chunk_summaries)

        # Limit to exact word count
        words_list = raw_summary.split()
        summary_limited = " ".join(words_list[:summary_length])

        # --- Format Summary ---
        if output_format == "Paragraph":
            summary_out = summary_limited
        else:
            # Manual sentence split for bullets
            sentences = [s.strip() for s in summary_limited.replace("\n", " ").split(". ") if len(s.strip()) > 0]
            if len(sentences) > 1:
                bullets_text = "\n".join([f"• {s}." for s in sentences[:-1]])
                final_sentence = sentences[-1]
                summary_out = f"{bullets_text}\n\n{final_sentence}"
            else:
                summary_out = sentences[0]

        # --- Display Summary ---
        st.subheader("📝 Summary")
        st.markdown(f'<div class="summary-box">{summary_out}</div>', unsafe_allow_html=True)

        # Word count info
        wc = len(summary_out.replace("•", "").split())
        st.caption(f"Word count: {wc} (target: {summary_length})")
