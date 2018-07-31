import os
import numpy as np
import re
import nltk

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

#from textstat.textstat import textstat
from pyreadability import pyreadability as pyrb
from edit_distance import EditDistance

class PerplexityEvaluator(object):
    def __init__(self):
        self.probs = []

    def add_pdfs_and_seqs(self, pdfs, gt_seq, s = None, e = None): 
        pad_seq = gt_seq.get_padded_seqs(s, e)
        mask = gt_seq.get_masks(s, e)
        for sent_idx in xrange(pdfs.shape[0]):
            sent_probs = []
            for word_idx in xrange(pdfs.shape[1]):
                if mask[sent_idx, word_idx] > 0.5:
                    gt_token = pad_seq[sent_idx, word_idx]
                    sent_probs.append(pdfs[sent_idx, word_idx, gt_token])
            self.probs.append(sent_probs)

    def get_ppl_mean_sdev(self):
        for row in self.probs:
            for v in row:
                print v,
            print ""
        flat_prob = np.array([v for row in self.probs for v in row])
        num_chars = flat_prob.shape[0]
        log_prob = -np.log2(flat_prob)
        ppl_mean = np.mean(log_prob)
        ppl_sdev = np.std(log_prob)
        print len(flat_prob)
        return ppl_mean, ppl_sdev

    def get_ppl(self):
        ppl, _ = self.get_ppl_mean_sdev()
        return ppl

    def __repr__(self):
        ret_str = "PPL: 2 ** (%.5f +- %.5f)" % self.get_ppl_mean_sdev()
        return ret_str
        

class SentenceEvaluator(object):
    def __init__(self):
        self.gt = {}
        self.gen = {}
        self.count = 0
        self.bleu = Bleu()
        self.rouge = Rouge()
        self.rb = pyrb.Readability(syllable_counter=pyrb.CMUDictCounter())
        #self.meteor = Meteor()
        #self.cider = Cider()

    def add_sentence_pair(self, generated, ground_truth):
        if not isinstance(generated, str):
            print "ERROR:", generated
            print type(generated)
        assert isinstance(generated, str)
        assert isinstance(ground_truth, str)

        self.gt[self.count] = [ground_truth]
        self.gen[self.count] = [generated]
        self.count += 1

    def add_pairs(self, generated, ground_truth):
        assert len(generated) == len(ground_truth)
        
        for gen, gt in zip(generated, ground_truth):
            self.add_sentence_pair(gen, gt)

    def clear(self):
        self.gt = {}
        self.gen = {}
        self.count = 0

    def edit_distance(self):
        ed = EditDistance()

        total_dist = 0
        total_norm_dist = 0
        op_count = {'m': 0, 'i': 0, 'd': 0, 'r': 0}
        op_count_norm = {'m': 0, 'i': 0, 'd': 0, 'r':0}
        num_examples = len(self.gt)
        num_examples = max(num_examples, 1)
        for i in self.gt.keys():
            gt = self.gt[i][0].split()
            gen = self.gen[i][0].split()

            max_len = float(max(len(gt), len(gen)))
            max_len = max(max_len, 1.0)
            dist = ed.compute(gt, gen)
            total_dist += dist
            total_norm_dist += dist / max_len

            ops = ed.operations()
            for op in ops:
                op_count[op] += 1
                op_count_norm[op] += 1.0 / max_len

        mean_dist = total_dist / float(num_examples)
        mean_norm_dist = total_norm_dist / float(num_examples)
        
        for op in op_count:
            op_count[op] /= float(num_examples)
            op_count_norm[op] /= float(num_examples)

        return mean_dist, mean_norm_dist, op_count, op_count_norm

    def bleu_score(self):
        score, scores = self.bleu.compute_score(self.gt, self.gen)
        return score

    def bleu_scores(self):
        score, scores = self.bleu.compute_score(self.gt, self.gen)
        return np.array(scores).T

    def rouge_score(self):
        return self.rouge.compute_score(self.gt, self.gen)[0]

    def meteor_score(self):
        return self.meteor.compute_score(self.gt, self.gen)[0]

    def cider_score(self):
        return self.cider.compute_score(self.gt, self.gen)[0]

    def _get_words_per_sequence(self, lst):
        lens = [len(a[0].split()) for a in lst]
        return np.array(lens, dtype=np.int32)

    def _get_words_per_sentence(self, lst):
        lens = []
        for a in lst:
            for s in nltk.sent_tokenize(a[0]):
                lens.append(len(s.split()))
        return np.array(lens, dtype=np.int32)

    def mean_words_per_sentence_gt(self):
        return np.mean(self._get_words_per_sentence(self.gt.values()))

    def mean_words_per_sentence_gen(self):
        return np.mean(self._get_words_per_sentence(self.gen.values()))

    def mean_words_per_sentence_diff(self):
        gt_wps = self._get_words_per_sequence(self.gt.values())
        gen_wps = self._get_words_per_sequence(self.gen.values())
        return np.mean(gt_wps - gen_wps)

    def _get_sentence_list(self, sent_map):
        text = []
        for sent in sent_map.values():
            text.append(sent[0])
        return text

    def _get_sentence_list_gt(self):
        return self._get_sentence_list(self.gt)

    def _get_sentence_list_gen(self):
        return self._get_sentence_list(self.gen)

    def _text_stats_str(self, sentences):
        text = []
        for sent in sentences:
            sent_strip = sent.strip()
            if len(sent_strip) == 0 or sent_strip[-1] != '.':
                text.append(sent_strip + '.')
            else:
                text.append(sent_strip)
        text = " ".join(text)
        stat_str = ""
        try:
            fre = self.rb.flesch_kincaid_reading_ease(text)
            stat_str += "Flesch reading ease: %s\n" % str(fre)
            #si = textstat.smog_index(text)
            #stat_str += "Smog index: %s\n" % str(si)
            fkg = self.rb.flesch_kincaid_grade_level(text)
            stat_str += "Flesch kincaid grade: %s\n" % str(fkg)
            cli = self.rb.coleman_liau_index(text)
            stat_str += "Coleman liau index: %s\n" % str(cli)
            ari = self.rb.automated_readability_index(text)
            stat_str += "Automated redability index: %s\n" % str(ari)
            dcrs = self.rb.dale_chall_readability(text)
            stat_str += "Dale chall readability score: %s\n" % str(dcrs)
            #lwf = textstat.linsear_write_formula(text)
            #stat_str += "Linsear write formula: %s\n" % str(lwf)
            #gf = textstat.gunning_fog(text)
            #stat_str += "Gunning fog: %s\n" % str(gf)
        except Exception as e:
            stat_str += "Text quality is poor: caused an exeption during evaluation."
            print e
        
        return stat_str
        
    def __repr__(self):
        #for i in self.gt:
        #    print self.gt[i]
        #    print self.gen[i]
        #    print ""
        bleu = self.bleu_score()
        rouge = self.rouge_score()
        #meteor = self.meteor_score()
        #cider = self.cider_score()

        rep = "Evaluation Results (%d pairs):\n" % len(self.gt)
        rep += "Bleu: %s\n" % str(bleu)
        rep += "Rouge: %s\n" % str(rouge)
        #rep += "Meteor: %s\n" % str(meteor)
        #rep += "Cider: %s\n" % str(cider)

        words_per_sentence_gt = self.mean_words_per_sentence_gt()
        rep += "Mean words per sentence ground-truth: %f\n" %  words_per_sentence_gt
        words_per_sentence_gen = self.mean_words_per_sentence_gen()
        rep += "Mean words per sentence generated: %f\n" %  words_per_sentence_gen
        words_per_sentence_diff = self.mean_words_per_sentence_diff()
        rep += "Mean words per sentence diff 'mean(|gt| - |gen|)': %f\n" % words_per_sentence_diff

        rep += "--------Generated Readability Stats:--------\n"
        rep += self._text_stats_str(self._get_sentence_list_gen())
        rep += "--------Ground Truth Readability Stats:--------\n"
        rep += self._text_stats_str(self._get_sentence_list_gt())
        return rep

    def print_edit_distance(self):
        mean_dist, mean_norm_dist, op_average, op_average_norm = self.edit_distance()
        print "EditDistance Stats:"
        print "mean_dist:", mean_dist
        print "mean_norm_dist", mean_norm_dist
        print "op_average:", op_average
        print "op_average_norm:", op_average_norm
