"""Revised article: how OCR quality affects the downstream summary.

Runs the SAME mT5 model and SAME generation settings as the paper (Table 4) on three inputs:
  1. ground_truth          - the manually verified transcription
  2. ocr_paper_pipeline    - OCR output of the original pipeline
  3. ocr_improved_pipeline - OCR output of the improved pipeline
Uses the model directly (works with both older and newer versions of transformers).
Results are saved to summary_results.txt in this folder.
"""
import json, os, sys
import torch, transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

HERE = os.path.dirname(os.path.abspath(__file__))
texts = json.load(open(os.path.join(HERE, "..", "results", "summary_inputs.json"), encoding="utf-8"))

MODEL = "csebuetnlp/mT5_multilingual_XLSum"
print("Loading model (first time may take a few minutes)...")
tok = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
model.eval()

lines = [f"python {sys.version.split()[0]} | transformers {transformers.__version__} | torch {torch.__version__}",
         "settings: input[:1200] chars, max_length=120, min_length=40, num_beams=6, "
         "repetition_penalty=3.5, no_repeat_ngram_size=3, do_sample=False", ""]

for name, text in texts.items():
    enc = tok(text[:1200], return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        ids = model.generate(**enc, max_length=120, min_length=40, do_sample=False,
                             num_beams=6, repetition_penalty=3.5,
                             no_repeat_ngram_size=3, early_stopping=True)
    out = tok.decode(ids[0], skip_special_tokens=True)
    print(f"\n=== {name} ===\n{out}")
    lines += [f"=== {name} ===", out, ""]

with open(os.path.join(HERE, "..", "results", "summary_results.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\nDone. Results saved to summary_results.txt")
