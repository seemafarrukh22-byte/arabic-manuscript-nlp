"""Gradio app for the Arabic OCR-to-NLP pipeline (revised release)."""
import gradio as gr
from PIL import Image
import pipeline as P

PRESET_LABELS = {
    'Handwritten manuscript (improved pipeline)': 'manuscript',
    'Printed book - clean / modern': 'print_clean',
    'Printed book - old or low resolution': 'print_old',
    'Original pipeline (first submission)': 'original',
}
_summarizer = None


def get_summarizer():
    global _summarizer
    if _summarizer is None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        name = 'csebuetnlp/mT5_multilingual_XLSum'
        tok = AutoTokenizer.from_pretrained(name, use_fast=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(name)
        model.eval()
        _summarizer = (tok, model)
    return _summarizer


def summarise(text):
    import torch
    tok, model = get_summarizer()
    enc = tok(text[:1200], return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        ids = model.generate(**enc, max_length=120, min_length=40, do_sample=False, num_beams=6,
                             repetition_penalty=3.5, no_repeat_ngram_size=3, early_stopping=True)
    return tok.decode(ids[0], skip_special_tokens=True)


def load_input(file, page_no):
    path = file if isinstance(file, str) else file.name
    if path.lower().endswith('.pdf'):
        # PyMuPDF rasterisation at 400 dpi, as in the article's evaluation
        # (other renderers, e.g. poppler, give slightly different pixels and different OCR output).
        import fitz
        page = fitz.open(path)[int(page_no) - 1]
        pix = page.get_pixmap(dpi=400)
        return Image.frombytes('RGB' if pix.n < 4 else 'RGBA', (pix.w, pix.h), pix.samples).convert('RGB')
    return Image.open(path).convert('RGB')


def process(file, page_no, doc_type, split, reference, do_summary):
    if file is None:
        return None, 'Please upload an image or PDF.', '', '', ''
    try:
        img = load_input(file, page_no)
        preset = PRESET_LABELS[doc_type]
        processed, raw, cleaned = P.run(img, preset, split=split)
        score = ''
        if reference and reference.strip():
            c, w = P.cer_wer(reference, raw)
            score = f'CER {c*100:.1f}%  |  WER {w*100:.1f}%  (normalised as in the article)'
        summary = ''
        if do_summary:
            if len(cleaned.split()) < 30:
                summary = 'Text too short or OCR quality too low for summarisation.'
            else:
                summary = summarise(cleaned)
        return processed[0], raw, cleaned, score, summary
    except Exception as e:
        return None, f'Error: {e}', '', '', ''


with gr.Blocks(title='Arabic OCR-to-NLP Pipeline') as demo:
    gr.Markdown(
        '# Arabic OCR-to-NLP Pipeline\n'
        'Open-source workflow for Arabic textual heritage: preprocessing, Tesseract OCR, cleaning, '
        'and an experimental summary. Accompanies the article *From Manuscript to Machine-Readable Text* '
        '(under review).\n\n'
        '**Choose the document type**: no single preprocessing setting suits all material. Measured accuracy (CER) '
        'on al-Nawawī\'s introduction: modern print ≈ 1–16%, old print ≈ 34%, handwritten manuscript ≈ 57%.\n\n'
        '**Warning:** the summary uses a news-trained model (mT5 XL-Sum). It may invent content even from correct text, '
        'and must not be relied on for scholarly purposes.')
    with gr.Row():
        with gr.Column():
            f = gr.File(label='Upload page image (PNG/JPG) or PDF', file_types=['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.pdf'])
            page = gr.Number(value=1, precision=0, label='PDF page number')
            dt = gr.Dropdown(list(PRESET_LABELS), value='Handwritten manuscript (improved pipeline)', label='Document type')
            sp = gr.Checkbox(label='Image is a two-page opening (split at the binding)', value=False)
            ref = gr.Textbox(lines=4, label='Optional: reference transcription (to compute CER/WER)', rtl=True)
            summ = gr.Checkbox(label='Generate experimental summary (slow; first run downloads ~2.3 GB model)', value=False)
            btn = gr.Button('Run', variant='primary')
        with gr.Column():
            out_img = gr.Image(label='Processed image (first page)')
            out_raw = gr.Textbox(lines=8, label='Raw OCR text', rtl=True)
            out_clean = gr.Textbox(lines=6, label='Cleaned text', rtl=True)
            out_score = gr.Textbox(label='Accuracy against reference')
            out_sum = gr.Textbox(lines=4, label='Experimental summary (mT5 XL-Sum)', rtl=True)
    btn.click(process, [f, page, dt, sp, ref, summ], [out_img, out_raw, out_clean, out_score, out_sum])

if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=7860)
