import numpy as np
import spacy
import os
import subprocess
import json
import networkx as nx
import collections

def get_sw_set():
    from nltk.corpus import stopwords
    sw = stopwords.words('english')
    sw.extend(['two', 'nt', 're', 'll', ',', '.', 'm', 'd', 've', 'nah', 'one', 'UNKNOWN', 's'])
    sw_set = set(sw)
    
    return sw_set

def remove_stopwords_sents(sentences):

    sw_set = get_sw_set()

    all_sents = []
    for sent in sentences:
        new_sent = []
        for w in sent:
            if w not in sw_set:
                new_sent.append(w)
        all_sents.append(new_sent)

    return all_sents

def remove_unknown(data, encoder=True, remove_order=False):
    assert encoder
    text = data.enc_in.get_text()
    text_new = []
    for sent in text:
        sent_new = []
        for w in sent:
            if w != "UNKNOWN":
                sent_new.append(w)
        if remove_order:
            np.random.shuffle(sent_new)
        text_new.append(sent_new)

    data.enc_in.from_text(text_new)
    return data

def read_framenet(fname):
    tokens = []
    frames = []
    for line in open(fname, "r"):
        js = json.loads(line)
        tokens.append(js["tokens"])
        frames.append(js["frames"])
    return tokens, frames

def frames_to_name_span_dict(frames):
    frame_dicts = []
    for frame in frames:
        fd = {}
        for anns in frame:
            tg = anns['target']
            start = tg['spans'][0]['start']
            end = tg['spans'][0]['end']
            name = tg['name']
            fd[(start, end)] = name
        frame_dicts.append(fd)
    return frame_dicts

def run_semfor(text, threads=12, batch_size = 50000, temp_dir="/localdata/u4534172/tmp/", 
        semfor_bin="/localdata/u4534172/Libraries/semafor/bin", fid="any", from_cache=False, nlp=None):

    if nlp is None:
        nlp = spacy.load('en')

    binary_path = os.path.join(semfor_bin, "runSemafor.sh")
    temp_in_path = os.path.join(temp_dir, "semfor_text_in_%s.txt" % fid)
    temp_out_path = os.path.join(temp_dir, "semfor_json_out_%s.json" % fid)

    text_new = []
    for doc in nlp.pipe([" ".join(c).decode('ascii', 'ignore') for c in text], 
            n_threads=threads, batch_size=batch_size):
        text_parse = []
        for tok in doc:
            text_parse.append(tok.orth_)
        text_new.append(" ".join(text_parse))
    
    if not from_cache or not os.path.exists(temp_out_path):
        # remove existing file if already there
        if os.path.exists(temp_out_path):
            os.remove(temp_out_path)

        # write the text to the temp directory
        fout = open(temp_in_path, "w")
        for line in text_new:
            fout.write(line+"\n")
        fout.close()

        #run semfor to parse the text
        p = subprocess.Popen([binary_path, temp_in_path, temp_out_path, str(threads)], cwd=semfor_bin)
        p.wait()

    # read the json
    tokens, frames = read_framenet(temp_out_path)
    frame_dicts_list = frames_to_name_span_dict(frames)
    assert len(text_new) == len(frame_dicts_list)
    frame_dicts = dict(zip(text_new, frame_dicts_list))

    return frame_dicts

def read_framenet_graph(path = "/localdata/u4534172/COCO/mscoco_framenet_parent_graph.pik"):
    parent_graph = nx.read_gpickle(path)
    return parent_graph

def find_verb_parent(tok):
    cur_tok = tok
    if cur_tok == cur_tok.head:
        return None
    while True:
        cur_tok = cur_tok.head
        if cur_tok.pos_ == "VERB" and cur_tok.dep_ != "aux":
            return cur_tok
        if cur_tok == cur_tok.head:
            return None

def find_compound(tok):
    res = []
    for child in tok.lefts:
        if child.dep_ == "compound" and child.head == tok:
            res.append(child)
    res.append(tok)
    for child in tok.rights:
        if child.dep_ == "compound" and child.head == tok:
            res.append(child)
    return res

def find_subject(tok):
    for child in tok.children:
        if child.dep_ == "nsubj" and child.pos_ == "NOUN":
            return child
    if tok.dep_ in ["acl", "relcl"] and tok.head.pos_ == "NOUN":
        return tok.head
    return None

def get_all_verb_subj_pairs(doc):
    sv = []
    for tok in doc:
        if tok.dep_ == "nsubj" and tok.pos_ == "NOUN":
            vp = find_verb_parent(tok)
            if vp is not None:
                sv.append((vp.i, [v.i for v in find_compound(tok)]))
        if tok.dep_ in ["acl", "relcl"] and tok.pos_ == "VERB" and tok.head.pos_ == "NOUN":
            sv.append((tok.i, [v.i for v in find_compound(tok.head)]))
        if tok.dep_ == "xcomp" and tok.head.pos_ == "VERB":
            subj = find_subject(tok.head)
            if subj is not None:
                sv.append((tok.i, [v.i for v in find_compound(subj)]))
    return sv

class VerbToFrameNet(object):
    def __init__(self, text, fid=None, nlp=None):
        if fid is None:
            fid = str(hash(tuple([tuple(t) for t in text])))
        self.frame_dicts = run_semfor(text, fid=fid, from_cache=True, nlp = nlp)
        self.parent_graph = read_framenet_graph()

    def verb_to_frame_raw(self, doc_txt, idx, orth):
        framenet_prefix = "FRAMENET"
        frames = self.frame_dicts[doc_txt]
        fk = (idx, idx+1)
        fk2 = (idx, idx+2)
        fkn = (idx-1, idx+1)
        if fk in frames:
            fname = frames[fk]
            fname_raw = fname
            parent = self.parent_graph.adj[fname].keys()[0]
            if self.parent_graph.nodes[parent]['count'] >= 200:
                fname = framenet_prefix+self.parent_graph.adj[fname].keys()[0].replace("_", "")
            else:
                fname = None
        elif fk2 in frames:
            fname = frames[fk2]
            fname_raw = fname
            parent = self.parent_graph.adj[fname].keys()[0]
            if self.parent_graph.nodes[fname]['count'] >= 200:
                fname = framenet_prefix+self.parent_graph.adj[fname].keys()[0].replace("_", "")
            else:
                fname = None
        elif fkn in frames:
            fname = frames[fkn]
            fname_raw = fname
            parent = self.parent_graph.adj[fname].keys()[0]
            if self.parent_graph.nodes[fname]['count'] >= 200:
                fname = framenet_prefix+self.parent_graph.adj[fname].keys()[0].replace("_", "")
            else:
                fname = None
        else:
            fname = None
            fname_raw = orth

        return fname, fname_raw

    def verb_to_frame(self, doc_txt, idx, orth):
        fname, _ = self.verb_to_frame_raw(doc_txt, idx, orth)
        return fname


def get_word_to_keep(w, raw_text_line, lemmatize, append_pos, framenet, pos_list, sw_set, vfn, to_keep = None, verb_subj = None):
    allowed_colocates = [["hot", "dog"]]

    word = w.orth_.encode('ascii','ignore')
    pos = w.pos_
    lemma = w.lemma_.encode('ascii','ignore')

    store_word = word
    if lemmatize:
        store_word = lemma
    if append_pos:
        store_word = store_word + pos*3

    next_lemma = ""
    if w.i+1 < len(w.doc):
        next_lemma = w.nbor(1).lemma_
    last_lemma = ""
    if w.i > 0:
        last_lemma = w.nbor(-1).lemma_
    in_colocates = False
    for co in allowed_colocates:
        if co[0] == lemma and co[1] == next_lemma:
            in_colocates = True
        if co[0] == last_lemma and co[1] == lemma:
            in_colocates = True

    #print word, pos, lemma,
    if in_colocates:
        return store_word
    elif word in sw_set or lemma in sw_set:
        return None
    elif framenet and pos == "VERB" and to_keep is None:
        frame = vfn.verb_to_frame(raw_text_line, w.i, store_word)
        if frame is not None:
            return frame
    elif framenet and pos == "VERB" and to_keep is not None:
        if w.i not in verb_subj:
            return None
        for subj in verb_subj[w.i]:
            if subj in to_keep:
                frame = vfn.verb_to_frame(raw_text_line, w.i, store_word)
                if frame is not None:
                    return frame
                else:
                    return None

    elif pos in pos_list:
        return None
    else:
        return store_word

    return None


def remove_style_text(text, threads, batch_size, remove_order = False, using_importance = False, lemmatize=False, append_pos=False, 
        pos_list = ["ADJ", "PRT", "PUNCT", "ADV", "PRON", "CONJ"], framenet=False):
    sw_set = get_sw_set()

    nlp = spacy.load('en')
    vfn = None
    if framenet:
        vfn = VerbToFrameNet(text, nlp=nlp)
        

    text_new = []
    for doc in nlp.pipe([" ".join(c).decode('ascii', 'ignore') for c in text], 
            n_threads=threads, batch_size=batch_size):
        raw_text_line = " ".join([w.orth_ for w in doc])
        text_line = []
        if framenet:
            verb_subj = get_all_verb_subj_pairs(doc)
            vsubj_dict = collections.defaultdict(list)
            for v, subjs in verb_subj:
                vsubj_dict[v].extend(subjs)
                 
            to_keep = []
            for w in doc:
                wk = get_word_to_keep(w, raw_text_line, lemmatize, append_pos, framenet, pos_list, sw_set, vfn, to_keep=None, verb_subj=None)
                if wk is not None:
                    to_keep.append(w.i)
        else:
            verb_subj = None
            vsubj_dict = None
            to_keep = None
        for w in doc:
            wk = get_word_to_keep(w, raw_text_line, lemmatize, append_pos, framenet, pos_list, sw_set, vfn, to_keep, vsubj_dict)
            if wk is not None:
                text_line.append(wk)
        if remove_order:
            np.random.shuffle(text_line)
        text_new.append(text_line)
    return text_new


def remove_style2(data, threads, batch_size, encoder=True, remove_order = False, use_importance=False):
    using_importance = False
    if use_importance and data.important_tokens is not None:
        using_importance = True
        if encoder:
            data.enc_in.from_text(data.important_tokens)
        else:
            data.dec_out.from_text(data.important_tokens)

    if encoder:
        text = data.enc_in.get_text()
    else:
        text = data.dec_out.get_text()

    text_new = remove_style_text(text, threads, batch_size, remove_order = remove_order, using_importance=use_importance)

    if encoder:
        data.enc_in.from_text(text_new)
    else:
        data.dec_out.from_text(text_new)
        data.dec_in.from_text(text_new)
    return data

class VocabMap(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def map_tokens(self, tokens):
        return tokens

class VocabMapGlove(VocabMap):
    def __init__(self, tokenizer, topk=10, temp=0.1):
        super(VocabMapGlove, self).__init__(tokenizer)
        nlp = spacy.load('en')
        word_vectors = []
        special = []
        for w in tokenizer.index_to_word:
            if w in tokenizer.special_tokens:
                special.append(True)
                word_vectors.append(nlp.vocab[u""].vector)
            else:
                special.append(False)
                word_vectors.append(nlp.vocab[w.decode()].vector)
        word_vectors = np.array(word_vectors)
        special = np.array(special, dtype=np.bool)

        word_vectors = word_vectors / np.sqrt(np.sum(word_vectors**2, axis=1, keepdims=True))
        sim = np.dot(word_vectors, word_vectors.T)
        sim[:,special] = -10.0
        sim[special, :] = -10.0

        ap = np.argpartition(-sim, topk, axis=1)
        weights = sim[np.arange(sim.shape[0])[:,None], ap[:, :topk]]
        weights = np.exp(weights/temp)
        weights /= np.sum(weights, axis=1, keepdims=True)
        self.weights = weights
        self.candidates = ap[:, :topk]
        self.special = special

        #for w in np.random.choice(sim.shape[0], size=topk):
        #    print tokenizer.index_to_word[w], ":"
        #    for v,c in enumerate(ap[w,:topk]):
        #        print tokenizer.index_to_word[c],
        #        print weights[w,v]
        #    print ""

    def map_tokens(self, tokens):
        new_tokens = []
        for sent in tokens:
            new_sent = []
            for w in sent:
                if self.special[w]:
                    new_sent.append(w)
                else:
                    c = np.random.choice(self.candidates[w], p=self.weights[w])
                    new_sent.append(c)
            new_tokens.append(new_sent)
        if isinstance(tokens, np.ndarray):
            new_tokens = np.array(new_tokens, dtype=tokens.dtype)
        return new_tokens
