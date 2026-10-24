import sys, re, json, itertools
sys.path.insert(0, sys.argv[1])
import numpy as np, cv2, pytesseract, pymupdf as fitz
from ocr_eval import render, split_spread, load_gt, scores, norm

PDF, GT, OUT = sys.argv[2:5]
gt = load_gt(GT)
doc = fitz.open(PDF)

def binarise(g, how):
    if how == 'otsu':
        return cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if how.startswith('fix'):
        t = int(how[3:]); b = np.where(g < t, 0, 255).astype(np.uint8)
    if how == 'sauvola':
        from skimage.filters import threshold_sauvola
        b = np.where(g > threshold_sauvola(g, window_size=51, k=0.3), 255, 0).astype(np.uint8)
    # remove specks
    n, lab, st, _ = cv2.connectedComponentsWithStats(255 - b, 8)
    small = np.isin(lab, np.where(st[:, 4] < 40)[0]); b[small] = 255
    return b

def crop_block(b):
    ink = (b == 0).astype(np.uint8)
    h, w = ink.shape
    rows, cols = ink.sum(1), ink.sum(0)
    ys = np.where(rows > w * 0.03)[0]; xs = np.where(cols > h * 0.03)[0]
    if len(ys) < 2 or len(xs) < 2: return b
    y0, y1 = max(ys[0], int(h * .04)), min(ys[-1], int(h * .96))
    x0, x1 = max(xs[0], int(w * .05)), min(xs[-1], int(w * .95))
    return b[y0:y1, x0:x1]

AR = re.compile(r'[ء-ي]')
def line_filter(text):
    keep = []
    for ln in text.splitlines():
        toks = [t for t in ln.split() if len(AR.findall(t)) >= 2]
        letters = len(AR.findall(ln)); allc = len(re.sub(r'\s', '', ln)) or 1
        if letters / allc >= 0.7 and len(toks) >= 2:
            keep.append(' '.join(toks))
    return '\n'.join(keep)

cache = {}
def pages(i):
    if i not in cache:
        cache[i] = split_spread(render(doc[i], 400))
    return cache[i]

grid = list(itertools.product(['otsu', 'fix90', 'fix120', 'sauvola'], [False, True], [4, 6], [False, True]))
res = []
for binz, crop, psm, filt in grid:
    per = {}
    for i in range(3):
        R, L = pages(i)
        for k, P in (('R', R), ('L', L)):
            b = binarise(P, binz)
            if crop: b = crop_block(b)
            t = pytesseract.image_to_string(b, lang='ara', config=f'--oem 3 --psm {psm}')
            if filt: t = line_filter(t)
            per[f'{i+1}{k}'] = t
    tune = [k for k in per if k[0] in '12']; test = [k for k in per if k[0] == '3']
    def agg(keys):
        c, w = scores(' '.join(gt[k] for k in keys), ' '.join(per[k] for k in keys))
        return round(c * 100, 1), round(w * 100, 1)
    row = {'bin': binz, 'crop': crop, 'psm': psm, 'filter': filt,
           'tune': agg(tune), 'test': agg(test), 'all': agg(list(per)), 'texts': per}
    res.append(row)
    print(binz, crop, psm, filt, 'tune', row['tune'], 'test', row['test'], 'all', row['all'], flush=True)
json.dump(res, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False)
