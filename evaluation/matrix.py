"""Apply the SAME two pipeline configurations to all three source types."""
import sys, re, json; sys.path.insert(0, 'code')
import numpy as np, cv2, pytesseract, pymupdf as fitz
from ocr_eval import scores, render, split_spread, load_gt
AR = re.compile(r'[ء-ي]')
def lf(text):
    keep = []
    for ln in text.splitlines():
        toks = [t for t in ln.split() if len(AR.findall(t)) >= 2]
        letters = len(AR.findall(ln)); allc = len(re.sub(r'\s', '', ln)) or 1
        if letters/allc >= 0.7 and len(toks) >= 2: keep.append(' '.join(toks))
    return '\n'.join(keep)
def otsu(g): return cv2.threshold(g, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
def fix90(g):
    b = np.where(g < 90, 0, 255).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(255-b, 8)
    b[np.isin(lab, np.where(st[:, 4] < 40)[0])] = 255; return b
def A(g): return pytesseract.image_to_string(otsu(g), lang='ara', config='--oem 3 --psm 6')          # original paper
def B(g): return lf(pytesseract.image_to_string(fix90(g), lang='ara', config='--oem 3 --psm 4'))   # improved
def gtsec(path, tag):
    gt, cur = {}, None
    for l in open(path, encoding='utf-8'):
        l = l.strip()
        if l.startswith('## '+tag): cur = l[3:5].strip(); gt[cur] = []; continue
        if l.startswith('#') or not l: continue
        if cur: gt[cur].append(l)
    return {k: ' '.join(v) for k, v in gt.items()}
res = {}
# 1 handwritten manuscript (pages split; same split used for both configs)
gt = load_gt('gt_v2.txt'); doc = fitz.open('data/sources/nawwi.pdf')
imgs = {}
for i in range(3):
    R, L = split_spread(render(doc[i], 400)); imgs[f'{i+1}R'] = R; imgs[f'{i+1}L'] = L
res['manuscript'] = (gt, imgs)
# 2 old print (QNL screenshots, main text box only)
BOX = {'P1': (10,386,18,611), 'P2': (80,355,106,584), 'P3': (86,410,20,602), 'P4': (89,415,69,598), 'P5': (99,371,18,600), 'P6': (94,479,74,601)}
gtp = gtsec('gt_print_draft.txt', 'P'); imgs = {}
for k, (a, b, x0, x1) in BOX.items():
    g = cv2.imread(f'print/p{k[1]}.png', 0)[a+3:b-3, x0+3:x1-3]
    imgs[k] = cv2.resize(g, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)   # bring ~100 dpi screenshots near 300 dpi
res['old_print'] = (gtp, imgs)
# 3 modern typeset (born-digital, rendered 300 dpi)
gtm = gtsec('gt_modern.txt', 'M'); d = fitz.open('print/modern.pdf'); imgs = {}
for k, i, top in (('M1', 1, .2), ('M2', 2, .09)):
    p = d[i].get_pixmap(dpi=300); g = cv2.cvtColor(np.frombuffer(p.samples, np.uint8).reshape(p.h, p.w, p.n)[:, :, :3], cv2.COLOR_RGB2GRAY)
    imgs[k] = g[int(g.shape[0]*top):]
res['modern_print'] = (gtm, imgs)
out = {}
for src, (gt, imgs) in res.items():
    out[src] = {}
    for name, f in (('A_original', A), ('B_improved', B)):
        tx = {k: f(imgs[k]) for k in gt}
        c, w = scores(' '.join(gt[k] for k in gt), ' '.join(tx[k] for k in gt))
        per = {k: round(scores(gt[k], tx[k])[0]*100, 1) for k in gt}
        out[src][name] = {'CER': round(c*100, 1), 'WER': round(w*100, 1), 'per_page_CER': per}
        print(src, name, out[src][name], flush=True)
json.dump(out, open('matrix_results.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
