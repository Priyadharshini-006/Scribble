import os
import streamlit as st
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from utils import enhance_image, extract_text
import google.generativeai as genai

# ── Load .env (GEMINI_API_KEY) ─────────────────────────────────────────────
# FIX 3: load .env and read GEMINI_API_KEY correctly
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    st.warning("AIzaSyCFqrFbtZGTbJzmO273tvFi8jaq30qfDYg")

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Scribble to Digital", page_icon="📝")

st.title("📝 Scribble to Digital")
st.write("Convert messy handwritten notes into clean text & to-do lists")

uploaded_file = st.file_uploader("Upload notes image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    # FIX 1: use_column_width is deprecated → use_container_width
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_array = np.array(image)
    enhanced = enhance_image(img_array)

    st.image(enhanced, caption="Enhanced Image", use_container_width=True)

    with st.spinner("🔍 Extracting text with OCR..."):
        raw_text = extract_text(enhanced)

    st.subheader("📄 Raw OCR Text")
    st.text_area("OCR Output", raw_text, height=150, label_visibility="collapsed")

    if st.button("✨ Convert to Digital"):
        if not api_key:
            st.error("❌ Please set your GEMINI_API_KEY in a .env file or as an environment variable.")
            st.stop()

        with st.spinner("🤖 Processing with AI..."):
            prompt = f"""
            Clean this OCR text, correct spelling using context,
            and extract to-do tasks separately.

            OCR Text:
            {raw_text}

            Output format:
            Clean Notes:
            - ...

            To-Do List:
            - ...
            """

            result = None
            try:
                model = genai.GenerativeModel("gemini-2.5-flash")
                response = model.generate_content(prompt)
                result = response.text
            except Exception as e:
                result = (
                    "AI fallback: unable to call Gemini API.\n\n"
                    "Clean Notes:\n" + raw_text + "\n\n"
                    "To-Do List:\n- Review notes\n- Improve OCR pipeline\n- Validate results"
                )

        st.subheader("✅ Digital Output")
        # FIX 2: was st.text(result.replace('\n',' ').strip())
        # .replace('\n',' ') collapsed all line breaks → output was one unreadable line
        # → st.markdown preserves bullet points and paragraph breaks
        st.markdown(result)

        # ── TXT Download ──────────────────────────────────────────────────
        st.download_button(
            label="📥 Download as TXT",
            data=result.encode("utf-8"),
            file_name="scribble_to_digital_output.txt",
            mime="text/plain",
        )
