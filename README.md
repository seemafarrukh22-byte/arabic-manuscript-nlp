# Arabic Manuscript NLP Pipeline

An end-to-end AI system for Arabic manuscript OCR and summarization.

---

## Project Overview

This project converts scanned Arabic manuscript pages into machine-readable text and automatically generates AI-based summaries.

Pipeline:

PDF/Image → OCR → Text Cleaning → NLP Summarization

---

## Features

- Arabic OCR using Tesseract
- Image preprocessing with OpenCV
- Arabic text normalization
- Transformer-based summarization
- Interactive web interface using Gradio

---

## Technologies Used

- Python
- OpenCV
- Tesseract OCR
- Hugging Face Transformers
- Gradio
- PyTorch

---

## Model Used

- csebuetnlp/mT5_multilingual_XLSum

---

## Current Limitations


The project demonstrates an end-to-end OCR-to-NLP workflow for Arabic manuscripts.

OCR quality depends heavily on manuscript clarity, scan resolution, and preprocessing methods. Multiple preprocessing strategies were experimentally evaluated, including thresholding and denoising techniques.

The final pipeline retained simpler preprocessing methods after observing that aggressive image enhancement reduced Arabic OCR readability.

Although summarization quality improved after preprocessing optimization and decoding adjustments, transformer-generated summaries may still occasionally contain contextual inaccuracies due to OCR noise and the complexity of historical Arabic texts.

---

## Research Significance

This project demonstrates:
- Arabic manuscript digitization
- OCR-to-NLP pipelines
- Transformer summarization workflows
- AI applications in Islamic and digital humanities research

---

##  Experimental Observations

This project experimentally evaluated the impact of image preprocessing on Arabic OCR quality.

Findings showed that:
- aggressive denoising reduced Arabic character readability
- simpler Otsu thresholding produced more stable OCR output
- OCR quality strongly influenced downstream summarization performance

These observations reflect real-world challenges in Arabic manuscript NLP pipelines.


## Author

Dr. Seema Tahir, Ph.D Islamic Studies, Subject Specialist with integrated AI techniques