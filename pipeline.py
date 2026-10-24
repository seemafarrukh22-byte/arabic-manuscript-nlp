"""Arabic OCR-to-NLP pipeline (revised release).

Preprocessing presets correspond to the configurations evaluated in the article:
  original    - Otsu binarisation, PSM 6           (original submission; "config. A")
  manuscript  - ink threshold 90 + speck removal, PSM 4, line filter ("config. B")
  print_clean - grayscale, PSM 4                     (best for clean modern print)
  print_old   - grayscale, upscale x3, PSM 6, line filter (best for old / low-res print)
"""
import re
import numpy as np
import cv2
import pytesseract

AR = re.compile(r'[ء-ي]')
DIAC = re.compile(r'[ؐ-ًؚ-ٰٟۖ-ۭـ]')

PRESETS = {
    'original':    dict(binarise='otsu',  scale=1, psm=6, line_filter=False),
    'manuscript':  dict(binarise='ink90', scale=1, psm=4, line_filter=True),
    'print_clean': dict(binarise='gray',  scale=1, psm=4, line_filter=False),
    'print_old':   dict(binarise='gray',  scale=3, psm=6, line_filter=True),
}


def to_gray(img):
    img = np.asarray(img)
    if img.ndim == 3:
        img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY)
    return img


def split_opening(g):
    """Split a two-page opening at the binding: darkest column in the central third.
    Returns [right_page, left_page] (Arabic reading order)."""
    h, w = g.shape
    band = g[int(h * .1):int(h * .9), int(w * .35):int(w * .65)]
    x = int(w * .35) + int(np.argmin(band.mean(axis=0)))
    return [g[:, x + 5:], g[:, :x - 5]]


def binarise(g, method):
    if method == 'gray':
        return g
    if method == 'otsu':
        return cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if method == 'ink90':
        b = np.where(g < 90, 0, 255).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(255 - b, 8)
        b[np.isin(lab, np.where(st[:, 4] < 40)[0])] = 255   # remove specks < 40 px
        return b
    raise ValueError(method)


def line_filter(text):
    """Keep lines that are mostly Arabic letters and contain at least two Arabic words."""
    keep = []
    for ln in text.splitlines():
        toks = [t for t in ln.split() if len(AR.findall(t)) >= 2]
        letters = len(AR.findall(ln))
        allc = len(re.sub(r'\s', '', ln)) or 1
        if letters / allc >= 0.7 and len(toks) >= 2:
            keep.append(' '.join(toks))
    return '\n'.join(keep)


def clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s؀-ۿ]', '', text)
    return text.strip()


def ocr_page(g, preset):
    cfg = PRESETS[preset]
    if cfg['scale'] > 1:
        g = cv2.resize(g, None, fx=cfg['scale'], fy=cfg['scale'], interpolation=cv2.INTER_CUBIC)
    b = binarise(g, cfg['binarise'])
    text = pytesseract.image_to_string(b, lang='ara', config=f"--oem 3 --psm {cfg['psm']}")
    if cfg['line_filter']:
        text = line_filter(text)
    return b, text


def run(image, preset='manuscript', split=False):
    """OCR one page image (or a two-page opening if split=True).
    Returns (processed images, raw text, cleaned text)."""
    g = to_gray(image)
    pages = split_opening(g) if split else [g]
    outs = [ocr_page(p, preset) for p in pages]
    raw = '\n\n'.join(t for _, t in outs)
    return [b for b, _ in outs], raw, clean(raw)


# ---------- evaluation (same normalisation as the article) ----------
def normalise(t):
    t = DIAC.sub('', t)
    t = re.sub('[أإآٱ]', 'ا', t)
    t = t.replace('ى', 'ي').replace('ئ', 'ي').replace('ؤ', 'و').replace('ء', '').replace('ة', 'ه')
    t = re.sub(r'[^ء-ي\s]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()


def cer_wer(reference, hypothesis):
    import jiwer
    r, h = normalise(reference), normalise(hypothesis)
    if not r:
        return None, None
    if not h:
        return 1.0, 1.0
    return jiwer.cer(r, h), jiwer.wer(r, h)
