import numpy as np
import json
import kenlm
import cPickle
import argparse
import string

import matplotlib.pyplot as plt

import sys
import os
import sklearn
import re

if __name__ == "__main__":
    COCO_EVAL_LIB = os.path.abspath("../../Libraries/new_coco_cap_spice/coco-caption")
else:
    COCO_EVAL_LIB = os.path.abspath("../../Libraries/new_coco_cap/coco-caption")
#COCO_EVAL_LIB = os.path.abspath("../../Libraries/new_coco_cap_spice/coco-caption")

sys.path.append(COCO_EVAL_LIB)
#sys.path.append(os.path.abspath("../StyleTranslation/context_trapdoor"))
from pycocoevalcap.bleu.bleu import Bleu
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
sys.path.append(os.path.abspath("./RomanceClassify/"))
from data_prep import FeatureExtractor

class SentenceEvaluator(object):

    def __init__(self, bleu = True, lm=True, clf=False, allcoco = False):
        self.do_bleu = bleu
        self.do_lm = lm
        self.do_clf = clf
        self.do_allcoco = allcoco
        

        if self.do_lm:
            self.lm = kenlm.LanguageModel("./data/lm_romance.bin")

        if self.do_clf:
            clf = cPickle.load(open("./RomanceClassify/lr_cap_rom_classifier_bigram.pik", "rb"))
            self.fx = clf["fe"]
            self.clf = clf["clf"]

        if self.do_allcoco:
            dataType='val2014'
            annFile='%s/annotations/captions_%s.json'%(COCO_EVAL_LIB,dataType)
            self.coco = COCO(annFile)

        self.clear()

    def clear(self):
        self.eval_store_gt = {}
        self.eval_store_gen = {}

    def add_gen_sent(self, fname, sent, overwrite=False):
        if isinstance(sent, list):
            sent = " ".join(sent)    
        if overwrite or fname not in self.eval_store_gen:
            self.eval_store_gen[fname] = [sent]


    def add_gt_sent(self, fname, sent):
        if isinstance(sent, list):
            sent = " ".join(sent)    
        if fname not in self.eval_store_gt:
            self.eval_store_gt[fname] = []
        self.eval_store_gt[fname].append(sent)

    def get_lm_score_all(self):
        all_scores = []
        all_sents = []
        for sent in self.eval_store_gen.values():
            full_scores = self.lm.full_scores(sent[0], bos=True, eos=True)
            lp, gram, oov = zip(*full_scores)
            #score = np.sum(lp)
            # change to log base 2
            lp /= np.log10(2)
            lp *= -1
            mean_score = np.mean(lp)
            all_scores.append(mean_score)
            all_sents.append(sent[0])
        return np.array(all_scores), all_sents

    def get_clf_score(self):
        sents = [" ".join(s) for s in self.eval_store_gen.values()]
        X = self.fx.transform(sents)
        tot = np.sum(self.clf.predict(X))
        return tot / float(X.shape[0])

    def get_num_sent(self):
        return len(self.eval_store_gen)

    def get_lm_score(self):
        scores,_ = self.get_lm_score_all()
        return np.mean(scores)

    def get_bleu_score(self):
        bleu = Bleu()
        scores = bleu.compute_score(self.eval_store_gt, self.eval_store_gen)[0]
        return scores

    def get_all_coco(self):
        seen_images = set()
        js_out = []
        for fname, cap_tok in self.eval_store_gen.items():
            if isinstance(cap_tok, list):
                caption = " ".join(cap_tok)
            if all(f in string.digits for f in fname):
                image_id = int(fname)
            else:
                image_id = int(re.findall("_0*?([1-9][0-9]+?)\.jpg", fname)[0])
            if image_id in seen_images:
                continue
            seen_images.add(image_id)
            js_out.append({"image_id":image_id, "caption":caption})

        TEMP_FILE = "/tmp/mscoco_val_in__TemporaryFile.json"
        json.dump(js_out, open(TEMP_FILE, "w"))

        res = self.coco.loadRes(TEMP_FILE)
        cocoEval = COCOEvalCap(self.coco, res)
        cocoEval.params['image_id'] = res.getImgIds()

        cocoEval.evaluate()


def senteval_on_file(input_file, textonly=False, clf=False, allcoco=False, raw=False, always_norm=False):
    se = SentenceEvaluator(clf=clf, allcoco=allcoco)

    if textonly:
        sys.path.append(os.path.abspath("../StyleTranslation/context_trapdoor"))
        from tokenizer import TextNormalizer, TextNormalizerOptions
        tno = TextNormalizerOptions(lowercase=True, remove_punct=True, remove_empty=True,
                replace_numbers=True, nltk_split=True)

        fin = open(input_file, "r")
        for i,line in enumerate(fin):
            line = line.strip()
            line_norm = " ".join(TextNormalizer.normalize_sentence(line, options=tno))
            if raw:
                se.add_gen_sent("%d" % i, line)
                se.add_gt_sent("%d" % i, line)
            else:
                se.add_gen_sent("%d" % i, line_norm)
                se.add_gt_sent("%d" % i, line_norm)
        print se.get_lm_score()
    else:

        sys.path.append(os.path.abspath("../StyleTranslation/context_trapdoor"))
        from tokenizer import TextNormalizer, TextNormalizerOptions
        tno = TextNormalizerOptions(lowercase=True, remove_punct=True, remove_empty=True,
                replace_numbers=True, nltk_split=True)
        #line_norm = " ".join(TextNormalizer.normalize_sentence(line, options=tno))

        data = json.load(open(input_file, "r"))
        for i, line in enumerate(data):
            if "out" in line and "gt" in line and "fname" in line:
                out = line["out"]
                if always_norm:
                    out = " ".join(TextNormalizer.normalize_sentence(line["out"], options=tno))
                se.add_gen_sent(line["fname"], out)
                se.add_gt_sent(line["fname"], line["gt"])
            elif "caption" in line:
                out = line["caption"]
                if always_norm:
                    out = " ".join(TextNormalizer.normalize_sentence(line["caption"], options=tno))
                se.add_gen_sent(line["image_id"], out)
                se.add_gt_sent(line["image_id"], out)
    return se

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", help="the input json file to test")
    ap.add_argument("--lm_hist", action="store_true", help="plot a histogram of language model scores")
    ap.add_argument("--textonly", action="store_true", help="is the input text only")
    ap.add_argument("--norm", action="store_true", help="re-run the text normalization, safest option but slow")
    args = ap.parse_args()

    se = senteval_on_file(args.input, clf=True, allcoco=True, always_norm = args.norm)

    print "File:", args.input
    print "Num Sent:", se.get_num_sent()
    print "CLF Score:", se.get_clf_score()
    print "LM Score:", se.get_lm_score()
    print "BLEU Score:", se.get_bleu_score()
    se_raw = senteval_on_file(args.input, clf=False, allcoco=True, raw=True, always_norm = args.norm)
    se_raw.get_all_coco()
    print "--------------------------------"

    if args.lm_hist:
        scores,_ = se.get_lm_score_all()
        plt.hist(scores, bins=50)
        plt.show()


if __name__ == "__main__":
    main()
        
