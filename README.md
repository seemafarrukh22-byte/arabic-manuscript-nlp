---
title: Arabic Manuscript OCR-to-NLP Pipeline
emoji: 📜
colorFrom: indigo
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# Arabic OCR-to-NLP Pipeline (revised release)

Accompanies: Seema Tahir, *From Manuscript to Machine-Readable Text: A Case Study of an End-to-End OCR-to-NLP Pipeline for Arabic Textual Heritage* (Journal of Digital Islamicate Research, revision of JDIR-1030).

Code, ground-truth transcriptions and evaluation scripts: https://github.com/seemafarrukh22-byte/arabic-manuscript-nlp

Document-type presets correspond to the configurations reported in the article (see `pipeline.py`).
The summary is produced by a news-trained model (csebuetnlp/mT5_multilingual_XLSum) and is experimental.
