"""OCR evaluation for JDIR-1030 revision.
Runs several preprocessing/OCR configurations on the al-Arbaʿīn manuscript spreads
and reports CER/WER against the verified ground truth."""
import re, sys, json
import numpy as np, cv2, pytesseract, pymupdf as fitz, jiwer


# ---------- normalisation (applied to both GT and OCR) ----------
DIAC = re.compile(r'[ؐ-ًؚ-ٰٟۖ-ۭـ]')
def norm(t):
    t = DIAC.sub('', t)
    t = re.sub('[أإآٱ]', 'ا', t)
    t = t.replace('ى', 'ي').replace('ئ', 'ي').replace('ؤ', 'و').replace('ء', '').replace('ة', 'ه')
    t = re.sub(r'[^ء-ي\s]', ' ', t)      # keep Arabic letters only
    return re.sub(r'\s+', ' ', t).strip()

def load_gt(path):
    pages, cur = {}, None
    for line in open(path, encoding='utf-8'):
        line = line.strip()
        m = re.search(r'\((\d[RL])\)', line)
        if line.startswith('##') and m: cur = m.group(1); pages[cur] = []; continue
        if not line or line.startswith('#'): continue
        if cur: pages[cur].append(line)
    return {k: ' '.join(v) for k, v in pages.items()}

def scores(ref, hyp):
    r, h = norm(ref), norm(hyp)
    return jiwer.cer(r, h) if h else 1.0, jiwer.wer(r, h) if h else 1.0

# ---------- image helpers ----------
def render(page, dpi):
    p = page.get_pixmap(dpi=dpi)
    img = np.frombuffer(p.samples, np.uint8).reshape(p.h, p.w, p.n)[:, :, :3]
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def otsu(g):
    return cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def split_spread(g):
    """Find the dark binding line in the central third; return (right_page, left_page)."""
    h, w = g.shape
    band = g[int(h*.1):int(h*.9), int(w*.35):int(w*.65)]
    col = band.mean(axis=0)
    x = int(w*.35) + int(np.argmin(col))
    return g[:, x+5:], g[:, :x-5]

def text_block(g):
    """Crop to the main text block using projection profiles of the binarised page."""
    b = 255 - otsu(g)
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    h, w = b.shape
    rows = b.sum(1) / 255; cols = b.sum(0) / 255
    ys = np.where(rows > w*0.02)[0]; xs = np.where(cols > h*0.02)[0]
    if len(ys) == 0 or len(xs) == 0: return g
    # ignore page borders: trim 3% margins
    y0, y1 = max(ys[0], int(h*.03)), min(ys[-1], int(h*.97))
    x0, x1 = max(xs[0], int(w*.04)), min(xs[-1], int(w*.96))
    return g[y0:y1, x0:x1]

def lines_from(g, n_expected=None):
    """Segment text lines via horizontal projection of the binarised block."""
    b = 255 - otsu(g)
    b = cv2.dilate(b, np.ones((1, 25), np.uint8))
    prof = cv2.GaussianBlur((b.sum(1)/255).astype(np.float32).reshape(-1, 1), (1, 31), 0).ravel()
    thr = prof.max() * 0.15
    on = prof > thr
    segs, s = [], None
    for i, v in enumerate(on):
        if v and s is None: s = i
        if not v and s is not None:
            if i - s > g.shape[0]*0.03: segs.append((s, i))
            s = None
    if s is not None: segs.append((s, len(on)))
    out = []
    for a, z in segs:
        pad = int((z-a)*0.35)
        out.append(g[max(0, a-pad):min(g.shape[0], z+pad), :])
    return out

def ocr(img, psm):
    return pytesseract.image_to_string(img, lang='ara', config=f'--oem 3 --psm {psm}')

# ---------- configurations ----------
def run(cfg, page, dpi=400):
    g = render(page, dpi)
    name = cfg
    if cfg == 'C0_raw_spread':             # no binarisation (gray only), whole spread
        return {'S': ocr(g, 6)}
    if cfg == 'C1_paper_otsu_spread':      # configuration described in the paper
        return {'S': ocr(otsu(g), 6)}
    if cfg == 'C2_deployed_adaptive':      # configuration of the current HF Space
        g3 = cv2.resize(render(page, 300), None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        g3 = cv2.bilateralFilter(g3, 9, 75, 75)
        b = cv2.adaptiveThreshold(g3, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
        return {'S': ocr(b, 11)}
    R, L = split_spread(g)
    if cfg == 'C3_split_otsu':
        return {'R': ocr(otsu(R), 6), 'L': ocr(otsu(L), 6)}
    if cfg == 'C4_split_crop_otsu':
        return {'R': ocr(otsu(text_block(R)), 6), 'L': ocr(otsu(text_block(L)), 6)}
    if cfg == 'C5_split_crop_lines':
        res = {}
        for k, P in (('R', R), ('L', L)):
            blk = text_block(P)
            res[k] = '\n'.join(ocr(otsu(li), 7) for li in lines_from(blk))
        return res
    raise ValueError(cfg)

CFGS = ['C0_raw_spread', 'C1_paper_otsu_spread', 'C2_deployed_adaptive',
        'C3_split_otsu', 'C4_split_crop_otsu', 'C5_split_crop_lines']

if __name__ == '__main__':
    PDF, GT, OUT = sys.argv[1:4]
    gt = load_gt(GT)
    doc = fitz.open(PDF)
    results, texts = {}, {}
    for cfg in (sys.argv[4].split(',') if len(sys.argv) > 4 else CFGS):
        refs, hyps = [], []
        texts[cfg] = {}
        for i, page in enumerate(doc):
            out = run(cfg, page)
            n = str(i+1)
            if 'S' in out:   # whole spread vs GT right+left (reading order)
                refs.append(gt[n+'R'] + ' ' + gt[n+'L']); hyps.append(out['S'])
            else:
                refs += [gt[n+'R'], gt[n+'L']]; hyps += [out['R'], out['L']]
            texts[cfg][n] = out
        R = ' '.join(refs); H = ' '.join(hyps)
        c, w = scores(R, H)
        results[cfg] = {'CER': round(c*100, 1), 'WER': round(w*100, 1)}
        print(cfg, results[cfg], flush=True)
    json.dump({'results': results, 'texts': texts}, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
