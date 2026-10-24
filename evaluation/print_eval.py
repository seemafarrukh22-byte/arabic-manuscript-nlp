import sys, re, json; sys.path.insert(0, sys.argv[1])
import cv2, numpy as np, pytesseract
from ocr_eval import scores, norm
d = sys.argv[2]
gt, cur = {}, None
for l in open(d+'/gt_print_draft.txt', encoding='utf-8'):
    l = l.strip()
    if l.startswith('## P'): cur = l[3:5].strip(); gt[cur] = []; continue
    if l: gt[cur].append(l)
gt = {k: ' '.join(v) for k, v in gt.items()}
BOX = {'P1': (10,386,18,611), 'P2': (80,355,106,584), 'P3': (86,410,20,602), 'P4': (89,415,69,598), 'P5': (99,371,18,600), 'P6': (94,479,74,601)}
def prep(i, how, scale):
    a,b,x0,x1 = BOX[i]; g = cv2.imread(f'{d}/print/p{i[1]}.png', 0)[a+3:b-3, x0+3:x1-3]
    if scale > 1: g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if how == 'gray': return g
    if how == 'otsu': return cv2.threshold(g, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
res = {}
for how, scale, psm in [('gray',1,6),('otsu',1,6),('otsu',3,6),('otsu',3,4),('gray',3,6)]:
    per = {k: pytesseract.image_to_string(prep(k, how, scale), lang='ara', config=f'--oem 3 --psm {psm}') for k in gt}
    c, w = scores(' '.join(gt[k] for k in gt), ' '.join(per[k] for k in gt))
    key = f'{how}|x{scale}|psm{psm}'; res[key] = {'CER': round(c*100,1), 'WER': round(w*100,1), 'texts': per}
    print(key, res[key]['CER'], res[key]['WER'], flush=True)
json.dump(res, open(d+'/print_results.json','w',encoding='utf-8'), ensure_ascii=False)
