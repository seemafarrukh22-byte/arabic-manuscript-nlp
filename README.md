# Arabic OCR-to-NLP Pipeline for Arabic Textual Heritage

Code, ground-truth data and results accompanying:

> Tahir, S. *From Manuscript to Machine-Readable Text: A Case Study of an End-to-End OCR-to-NLP Pipeline for Arabic Textual Heritage.* (under review).

**Live demo:** https://huggingface.co/spaces/seematahir-ai/arabic-manuscript-nlp

## What is in this repository

| Path | Description |
|---|---|
| `pipeline.py` | Preprocessing presets, Tesseract OCR, text cleaning, CER/WER with the article's normalisation |
| `app.py`, `Dockerfile`, `requirements.txt` | Gradio application as deployed on Hugging Face Spaces (pinned versions) |
| `data/ground_truth/` | Manually verified transcriptions of al-Nawawī's introduction to *al-Arbaʿīn* in three sources |
| `evaluation/` | Scripts that produce every figure reported in the article |
| `results/` | Raw results: CER/WER per configuration and page, OCR output, summaries |
| `nawwi.pdf` | The manuscript test pages (Alukah, al-Nuskha 4) |
| `Arabic Manuscript Pipeline.ipynb`, `ocr_text.txt`, `clean_text.txt`, `summary.txt`, `ocr_page1.txt`, `page_1_clean.png`, `apt.txt` | Files from the original (first) submission, kept for transparency |

## Test material
| Source | Reference | In repository? |
|---|---|---|
| A. Handwritten manuscript | Alukah, *Makhṭūṭat Kitāb al-Arbaʿīn al-Nawawiyya* (al-Nuskha 4), https://www.alukah.net/library/0/156219/ — 3 PDF pages = 6 manuscript pages | Yes (`nawwi.pdf`) |
| B. Early print | *Matn al-Arbaʿīn al-Nawawiyya*, Aleppo: al-Maktaba al-Adabiyya, [c. 1900]. Qatar National Library, QNL:00018580 — pp. 4–9 | No — obtain from https://ediscovery.qnl.qa/islandora/object/QNL:00018580 and save as `data/sources/qnl_p1.png` … `qnl_p6.png` |
| C. Modern typeset | *al-Arbaʿīn fī mabānī al-Islām wa-qawāʿid al-aḥkām* (Barnāmaj Muhimmāt al-ʿIlm, al-Kitāb al-Khāmis), Internet Archive `20200814_20200814_2226` — pp. 271, 273 | No — download from https://archive.org/details/20200814_20200814_2226 and save as `data/sources/ia_modern.pdf` |

## Ground-truth conventions
Main text block only (no catchwords, waqf stamp, running heads, page numbers or commentary); spelling as in each source;
honorific ligatures (ﷺ etc.) omitted. Before scoring, diacritics and tatwīl are removed and alif, yāʾ, hamza and tāʾ marbūṭa
forms are normalised (`pipeline.normalise`). Transcriptions were verified line by line by the author.

## Presets and measured accuracy (CER %, normalised)
| Preset | Settings | Manuscript | Early print | Modern print |
|---|---|---|---|---|
| `original` | Otsu, PSM 6 (first submission) | 74.0 | 47.0 | 16.9 |
| `manuscript` | page split, ink threshold 90 + speck removal, PSM 4, line filter | **56.6** | 52.6 | 21.0 |
| `print_old` | grayscale, ×3, PSM 6, line filter | – | **33.6** | – |
| `print_clean` | grayscale, PSM 4 | – | – | **8.9** |

Article figures were computed on the cropped main-text area; the web app processes the whole page, so full-page scores can be higher.

## Reproducing the results (run from the repository root)
```bash
pip install opencv-python-headless pytesseract pymupdf jiwer numpy   # plus Tesseract with the Arabic model
python evaluation/ocr_eval.py nawwi.pdf data/ground_truth/manuscript_alukah_nuskha4.txt results/manuscript_configs_C0-C5.json
python evaluation/grid.py evaluation nawwi.pdf data/ground_truth/manuscript_alukah_nuskha4.txt results/manuscript_grid_32configs.json
python evaluation/matrix.py              # needs the source files listed above in data/sources/
python evaluation/summary_compare.py     # downloads csebuetnlp/mT5_multilingual_XLSum (~2.3 GB)
```

## Environment used for the reported results
OCR: Ubuntu Linux, Python 3.13.16, PDF pages rasterised with PyMuPDF at 400 dpi, Tesseract 5.3.4 (Leptonica 1.82.0),
Arabic model from `tesseract-ocr-ara` 4.1.0, OpenCV 5.0.0, pytesseract 0.3.13, NumPy 2.5.3, jiwer 4.0.0.
Summaries: Google Colab, Python 3.13.16, Transformers 5.20.2, PyTorch 2.11.0 (CPU).
The deployed app uses the versions pinned in `requirements.txt` (Python 3.11, Debian Tesseract 5.3.0); on the first
manuscript opening it gives CER 60.5% against 54.4% in the evaluation environment.

## Limitations
Tesseract's Arabic model is trained on printed text; handwritten naskh is not recognised reliably, and results on
handwriting are sensitive to small differences in rasterisation and Tesseract version. The summariser
(csebuetnlp/mT5_multilingual_XLSum) is trained on news and may invent content even from correct text.

## Licence
Code: MIT. Ground-truth transcriptions and results: CC BY 4.0.
