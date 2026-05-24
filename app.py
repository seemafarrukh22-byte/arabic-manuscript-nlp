import gradio as gr
import pytesseract
import cv2
import numpy as np
from PIL import Image
import re
from transformers import pipeline

# -----------------------------
# Load Arabic summarization model
# -----------------------------
summarizer = pipeline(
    "summarization",
    model="csebuetnlp/mT5_multilingual_XLSum",
    tokenizer="csebuetnlp/mT5_multilingual_XLSum"
)

# -----------------------------
# OCR + NLP Pipeline Function
# -----------------------------
def process_image(image):

    try:
        # Convert PIL image to NumPy array
        img = np.array(image)

        # Convert RGB to grayscale
        gray = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2GRAY
        )

        # Otsu Thresholding
        _, binarized_image = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # OCR Configuration
        custom_config = r'--oem 3 --psm 6'

        # Arabic OCR Extraction
        text = pytesseract.image_to_string(
            binarized_image,
            lang='ara',
            config=custom_config
        )

        # -----------------------------
        # Arabic Text Cleaning
        # -----------------------------
        clean_text = re.sub(
            r'\s+',
            ' ',
            text
        )

        clean_text = re.sub(
            r'[^\w\s\u0600-\u06FF]',
            '',
            clean_text
        )

        # Limit input size
        input_text = clean_text[:1200]

        # -----------------------------
        # Summarization
        # -----------------------------
        summary = summarizer(
            input_text,
            max_length=120,
            min_length=40,
            do_sample=False,
            num_beams=6,
            repetition_penalty=3.5,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

        return (
            clean_text,
            summary[0]["summary_text"]
        )

    except Exception as e:
        return (
            "Error during OCR processing.",
            str(e)
        )

# -----------------------------
# Gradio Interface
# -----------------------------
iface = gr.Interface(
    fn=process_image,

    inputs=gr.Image(
        type="pil",
        label="Upload Arabic Manuscript Image"
    ),

    outputs=[
        gr.Textbox(
            label="Extracted Arabic Text",
            lines=12
        ),

        gr.Textbox(
            label="AI Generated Summary",
            lines=8
        )
    ],

    title="Arabic Manuscript OCR & Summarization System",

    description="""
Upload a scanned Arabic manuscript image to perform:

• Arabic OCR extraction  
• Text preprocessing and cleaning  
• AI-powered summarization using transformer models  

This project demonstrates an end-to-end Arabic NLP workflow for historical manuscript analysis and digital humanities research.
""",

    theme="soft"
)

iface.launch()