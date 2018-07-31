import numpy as np
from collections import Counter

class Toks:
    UNK = 3
    EOS = 2
    SOS = 1
    PAD = 0
    TOTAL_RES = 4

def tokenize_text(text, vsize=10000, res=Toks.TOTAL_RES, unk_idx=Toks.UNK, lower_case=False,
    word_to_idx = None, idx_to_word = None):

    tt = []
    tc = Counter()
    for line in text:
        if isinstance(line, list):
            if lower_case:
                add_line = [w.lower() for w in line]
            else:
                add_line = line
        else:
            if lower_case:
                add_line = line.lower().split()
            else:
                add_line = line.split()
        tt.append(add_line)
        tc.update(tt[-1])
    if word_to_idx is None or idx_to_word is None:
        mc = [v[0] for v in tc.most_common(vsize)]
        idx_to_word = ["RES%d" % i for i in xrange(res)]
        idx_to_word.extend(mc)
        word_to_idx = {w: i for i, w in enumerate(idx_to_word)}
    
    bias = np.ones(len(idx_to_word), dtype=np.float32)
    tok_text = []
    for line in tt:
        nline = []
        for w in line:
            if w not in word_to_idx:
                idx = unk_idx
            else:
                idx = word_to_idx[w]
            nline.append(idx)
            bias[idx]+=1
                
        tok_text.append(nline)
    
    bias = np.log(bias)
    bias -= np.median(bias)
    
    return idx_to_word, word_to_idx, tok_text, bias

def untokenize(sent, idx_to_word, unk_idx = Toks.UNK, eos_idx=Toks.EOS, to_text = False, suffix_token=None):
    text = []
    for s in sent:
        if s == eos_idx:
            break
        if s == unk_idx and unk_idx > 0:
            r = u"UNK"
        elif s < Toks.TOTAL_RES:
            continue
        else:
            r = idx_to_word[s]
        text.append(r)
    if suffix_token is not None:
        text.append(suffix_token)
    if to_text:
        text = ' '.join(text)
        
    return text

def pad_text(tok_text, max_slen=20, sos_idx=Toks.SOS, eos_idx=Toks.EOS):
    all_text = np.zeros((len(tok_text), max_slen), dtype=np.int)
    for j,line in enumerate(tok_text):
        all_text[j, 0] = sos_idx
        all_text[j, 1] = eos_idx
        s = 1
        e = max_slen-2
        for i in xrange(min(len(line), e)):
            c = i + s
            all_text[j, c] = line[i]
            all_text[j, c+1] = eos_idx
    return all_text


