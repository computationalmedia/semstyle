from nltk.corpus import treebank
import os
import sys
import cPickle
import json
import numpy as np
import theano

sys.path.append(os.path.abspath("./util"))
from tokenizer import TokenSequences, TextNormalizer, TextNormalizerOptions, TextTokenizer

WORD_DROP_TOKEN = "WORDDROPTOKEN"

class DataSet(object):
    def __init__(self, enc_in, dec_in, dec_out, img = None, filenames = None):
        self.enc_in = enc_in
        self.dec_in = dec_in
        self.dec_out = dec_out
        self.img = img
        self.filenames = filenames
        self.important_tokens = None

    def exp_seq(self, seqs, c):
        if seqs is None:
            return seqs
        new_seqs = []
        for s in seqs:
            for i in xrange(c):
                new_seqs.append(s)
        return new_seqs

    def expand(self, c):

        seq = self.exp_seq(self.enc_in.get_seqs(), c)
        self.enc_in.from_seqs(seq)

        seq = self.exp_seq(self.dec_in.get_seqs(), c)
        self.dec_in.from_seqs(seq)

        seq = self.exp_seq(self.dec_out.get_seqs(), c)
        self.dec_out.from_seqs(seq)

        self.img = np.repeat(self.img, c, axis=0)

        fname = self.exp_seq(self.filenames, c)
        self.filenames = fname

        if self.important_tokens is not None:
            imp = self.exp_seq(self.important_tokens, c)
            self.important_tokens = imp


    def get_img_emb(self, s, e):
        return self.img[s:e]

    def save_tokenizer(self, save_object=None):
        if save_object is None:
            save_object = {}
        save_object["input_tokenizer"] = self.enc_in.tokenizer
        save_object["output_tokenizer"] = self.dec_out.tokenizer

        return save_object

    def shuffle(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        order = self.dec_in.shuffle()
        self.enc_in.shuffle(order)
        self.dec_out.shuffle(order)
        if self.img is not None:
            self.img = self.img[order]
        if self.filenames is not None:
            self.filenames = [self.filenames[i] for i in order]
        if self.important_tokens is not None:
            self.important_tokens = [self.important_tokens[i] for i in order]

class JointTokenSequences(object):
    def __init__(self, tseq_a, tseq_b):
        self.tseq_a = tseq_a
        self.tseq_b = tseq_b

        self.num_example = min(tseq_a.get_num_seqs(), tseq_b.get_num_seqs())
        
    def get_vocab_size(self):
        pass

    def get_word_count(self):
        pass

    def shuffle(self, order):
        new_order = self.tseq_a.shuffle(order)
        self.tseq_b.shuffle(new_order)
        return new_order

    def get_num_seqs(self):
        return self.num_example*2

    def get_seqs(self, start = None, end = None):
        s1 = (start+1)/2
        s2 = start/2
        e1 = (end+1)/2
        e2 = (end)/2

        seqs_1 = self.tseq_a.get_seqs(s1, e1)
        seqs_2 = self.tseq_b.get_seqs(s2, e2)
        
        seq = []
        for i in xrange(max(len(seqs_1), len(seqs_2))):
            if start % 2 == 0:
                seq.append(seqs_1[i])
                if i < e2 - s2:
                    seq.append(seqs_2[i])
            else:
                seq.append(seqs_2[i])
                if i < e1 - s1:
                    seq.append(seqs_1[i])
        return seq


def sents_to_ascii(sents):
    for i in xrange(len(sents)):
        for j in xrange(len(sents[i])):
            sents[i][j] = sents[i][j].decode('ascii', 'ignore')
    return sents


def read_romance(input_vocab_size = 10000, output_vocab_size = 10000, seq_len = 10, 
        input_tokenizer = None, output_tokenizer = None, max_load = 0):
    all_sents = []

    i = 0
    for line in open("../PhraseImageEmbedding/data/text/romance_small.txt"):
        if line.strip():
            all_sents.append(line)
            i+=1
            if i == max_load and max_load != 0:
                break

    return read_dataset(all_sents, input_vocab_size, output_vocab_size, seq_len, 
            input_tokenizer = input_tokenizer, output_tokenizer = output_tokenizer)

def read_captions_raw(max_load=0):
    import json
    a = json.load(open("./data/coco/dataset.json", "r"))
    
    num_sent = 0
    all_sents = []

    for img in a['images']:
        for sent in img['sentences']:
            if max_load != 0 and num_sent > max_load:
                return all_sents
            all_sents.append(sent['raw'])
            num_sent+=1

    return all_sents

def read_captions(input_vocab_size = 10000, output_vocab_size = 10000, seq_len = 10, 
        input_tokenizer = None, output_tokenizer = None, max_load=0):

    all_sents = read_captions_raw(max_load)

    return read_dataset(all_sents, input_vocab_size, output_vocab_size, seq_len, 
            input_tokenizer = input_tokenizer, output_tokenizer = output_tokenizer)

def read_captions_switch(input_vocab_size=10000, output_vocab_size = 10000, seq_len = 10,
        input_tokenizer = None, output_tokenizer = None, max_load=0):

    import json
    a = json.load(open("./data/coco/dataset.json", "r"))
    
    num_sent = 0
    all_sents = [[] for i in xrange(5)]

    for img in a['images']:
        if max_load != 0 and num_sent > max_load:
            return all_sents
        num_sent+=1

        if len(img['sentences']) < 5:
            continue

        for si, sent in enumerate(img['sentences']):
            if si == 5:
                break
            all_sents[si].append(sent['tokens'])

    data = []
    for i in xrange(5):
        data.append(read_dataset(all_sents[i], input_vocab_size, output_vocab_size, seq_len, 
                input_tokenizer = input_tokenizer, output_tokenizer = output_tokenizer))
        if input_tokenizer is None:
            input_tokenizer = data[i].enc_in.tokenizer
        if output_tokenizer is None:
            output_tokenizer = data[i].dec_out.tokenizer

    
    seq_enc_in = []
    seq_dec_in = []
    seq_dec_out = []
    for i in xrange(5):
        for j in xrange(5):
            if i == j:
                continue
            seq_enc_in.extend(data[i].enc_in.get_seqs())
            seq_dec_in.extend(data[i].dec_in.get_seqs())
            seq_dec_out.extend(data[i].dec_out.get_seqs())

    enc_in = TokenSequences(input_tokenizer, reverse=True, start_pad = False, seq_len=seq_len)
    enc_in.from_seqs(seq_enc_in)
    dec_in = TokenSequences(input_tokenizer, reverse=False, start_pad = True, seq_len=seq_len)
    dec_in.from_seqs(seq_dec_in)
    dec_out = TokenSequences(output_tokenizer, reverse=False, start_pad = False, seq_len=seq_len)
    dec_out.from_seqs(seq_dec_out)

    data_full = DataSet(enc_in, dec_in, dec_out)

    return data_full

def read_captions_images_raw(input_vocab_size = 10000, output_vocab_size = 10000, seq_len = 10, 
    input_tokenizer = None, output_tokenizer = None, val=False, test=False, max_size = 0, get_feats=True, max_load=0,
    read_importance = False, full_dataset = False, flk=False):

    if max_load != 0:
        max_size = max_load

    FEATS_SIZE = 2048
    if flk:
        feature_file = "./data/stylenet/flk30_v3.pik"
        feature_file_b = "./data/stylenet/flk30_v3.pik"
    elif val or test:
        #feature_file = "./data/resized_coco_val_fc7.pik"
        feature_file = "./data/coco/coco_val_v3.pik"
        feature_file_b = "./data/coco/coco_train_v3.pik"
    else:
        feature_file = "./data/coco/coco_train_v3.pik"
        feature_file_b = "./data/coco/coco_val_v3.pik"
        #feature_file = "./data/resized_coco_train_fc7.pik"

    if get_feats:
        feats = cPickle.load(open(feature_file, "rb"))
        if full_dataset:
            feats.update(cPickle.load(open(feature_file_b, "rb")))

        if max_size != 0:
            nfeats = {}
            ngot = 0
            for k in feats:
                nfeats[k] = feats[k]
                ngot += 1
                if ngot == max_size:
                    break
            feats = nfeats
    if flk:
        json_file = "./data/stylenet/flk30.json"
    elif read_importance:
        json_file = "./data/coco/dataset_imp.json"
    elif full_dataset:
        json_file = "./data/coco/coco_dataset_full.json"
    else:
        json_file = "./data/coco/dataset.json"
    js = json.load(open(json_file, "r"))

    tokens = []
    imp_sents = []
    img_id = []
    img_filename = []
    img_id_to_filename = {}
    tok_to_feat = []
    c = 0
    got_count = 0

    if not test:
        for i,img in enumerate(js["images"]):
            if full_dataset:

                if val:
                    if img["extrasplit"] == "train":
                        continue 
                else:
                    if img["extrasplit"] == "val":
                        continue
            else:
                if val:
                    if img["split"] != "val":
                        continue
                else:
                    if test:
                        if img["split"] == "test":
                            c+=1
                        if img["split"] != "val" and img["split"] != "test":
                            continue
                    else:
                        if img["split"] != "train":
                            continue


            #only keep sentences for images we actually have features/images
            if get_feats and img["filename"] not in feats:
                continue

            for sen in img["sentences"]:
                tokens.append(sen["tokens"])
                if read_importance:
                    imp_sents.append(sen["tok_imp"])
                img_id.append(img["imgid"])
                img_filename.append(img["filename"])
                img_id_to_filename[img["imgid"]] = img["filename"]
                tok_to_feat.append(i)
                got_count+=1
            if got_count > max_load and max_load > 0:
                break
    else:
        id_v = 0
        for k,v in feats.items():
            tokens.append(["cat"])
            img_id.append(id_v)
            img_filename.append(k)
            img_id_to_filename[id_v] = k
            tok_to_feat.append(id_v)
            id_v+=1


    #print len(img_id_to_filename)
    #print "c", c

    #print "Res:", len(img_filename)
    #print "Imgs:", len(js["images"])
    #try:
    #    print "Feats:", len(feats)
    #except:
    #    pass

    X = None
    if get_feats:
        X = np.zeros((len(tok_to_feat), FEATS_SIZE), dtype=theano.config.floatX)
        for i, fname in enumerate(img_filename):
            X[i, :] = feats[fname]

    return tokens, X, img_filename, imp_sents


def read_captions_images(input_vocab_size = 10000, output_vocab_size = 10000, seq_len = 10, 
    input_tokenizer = None, output_tokenizer = None, val=False, test=False, max_size = 0, get_feats=True, max_load=0,
    read_importance = False, full_dataset = False, style_remove_func=None, rm_style_in=False, rm_style_out=False,
    prepend_type=None, flk=False):

    tokens, X, img_filename, imp_sents = read_captions_images_raw(input_vocab_size, output_vocab_size, seq_len,
    input_tokenizer, output_tokenizer, val, test, max_size, get_feats, max_load,
    read_importance, full_dataset, flk=flk)

    type_tags = None
    if prepend_type is not None:
        type_tags = [prepend_type]*len(tokens)

    data = read_dataset(tokens, input_vocab_size, output_vocab_size, seq_len, 
            input_tokenizer = input_tokenizer, output_tokenizer = output_tokenizer, mscoco=True,
            extraencode=full_dataset, style_remove_func = style_remove_func, rm_style_in = rm_style_in, rm_style_out = rm_style_out, type_tags=type_tags)
    data.img = X
    data.filenames = img_filename
    if read_importance:
        data.important_tokens = imp_sents

    return data


def read_treebank(input_vocab_size = 10000, output_vocab_size = 10000, seq_len = 10):
    all_sents = []
    for fname in treebank.fileids():
        sents = treebank.sents(fname)
        if sents:
            all_sents.extend(sents)
    
    return read_dataset(all_sents, input_vocab_size, output_vocab_size, seq_len)


def read_dataset(all_sents, input_vocab_size, output_vocab_size, seq_len, 
        input_tokenizer=None, output_tokenizer=None, mscoco=False, extraencode=False, style_remove_func = None, rm_style_in = False, rm_style_out=False, type_tags=None):

    type_tag_set = []
    if type_tags is not None:
        type_tag_set = list(set(type_tags))

    if mscoco:
        tno = TextNormalizerOptions(lowercase=True, remove_punct=False, remove_empty=True, replace_numbers=False,
                nltk_split=False)
    else:
        tno = TextNormalizerOptions()

    text_norm = TextNormalizer()
    if extraencode:
        all_sents_new = []
        for sent in all_sents:
            all_sents_new.append([v.encode('ascii', 'ignore') for v in sent])
        all_sents = all_sents_new
    all_sents = text_norm.normalize_text(all_sents, options = tno)

    # make sure the sentence is no longer than the max sequence length
    for s in xrange(len(all_sents)):
        if len(all_sents[s]) > seq_len:
            all_sents[s] = all_sents[s][:seq_len]

    if rm_style_in:
        all_sents_in = style_remove_func(all_sents)
    else:
        all_sents_in = all_sents
    if input_tokenizer is None:
        input_tokenizer = TextTokenizer(input_vocab_size, [WORD_DROP_TOKEN] + type_tag_set)
        input_tokenizer.fit(all_sents_in)

    if rm_style_out:
        all_sents_out = style_remove_func(all_sents)
    else:
        all_sents_out = all_sents
    if output_tokenizer is None:
        output_tokenizer = TextTokenizer(output_vocab_size, [WORD_DROP_TOKEN])
        output_tokenizer.fit(all_sents_out)

    enc_in = TokenSequences(input_tokenizer, reverse=True, start_pad = False, seq_len=seq_len)
    all_sent_in_new = all_sents_in
    if type_tags is not None:
        all_sent_in_new = [[type_tags[i]] + sent for i, sent in enumerate(all_sents_in)]
    enc_in.from_text(all_sent_in_new)
    dec_in = TokenSequences(output_tokenizer, reverse=False, start_pad = True, seq_len=seq_len)
    dec_in.from_text(all_sents_out)
    dec_out = TokenSequences(output_tokenizer, reverse=False, start_pad = False, seq_len=seq_len)
    dec_out.from_text(all_sents_out)

    return DataSet(enc_in, dec_in, dec_out)

def read_cap_and_rom(input_vocab_size = 10000, output_vocab_size = 10000, seq_len = 10, 
        input_tokenizer = None, output_tokenizer = None, max_load = 0, style_remove_func = None, 
        val=False, test=False, rom_filename="./data/romance_test_filtered_sample_1m.txt", shuffle=True):

    tokens, _, _, _ = read_captions_images_raw(input_vocab_size, output_vocab_size, seq_len,
    input_tokenizer, output_tokenizer, val, test, max_size=max_load, get_feats=False, max_load=max_load,
    read_importance=False, full_dataset=True) 

    rom_sents = []

    i = 0
    for line in open(rom_filename, "r"):
        if line.strip():
            rom_sents.append(line)
            i+=1
            if i == max_load and max_load != 0:
                break


    tokens = [' '.join(sent) for sent in tokens]

    all_tokens = []
    for sent in tokens:
        all_tokens.append(sent.encode('ascii', 'ignore'))
    tokens = all_tokens

    type_names = ["ROMANCEDATASET", "CAPTIONDATASET"]

    all_sents = rom_sents + tokens
    type_tags = [type_names[0]] * len(rom_sents) + [type_names[1]] * len(tokens)
    #print len(rom_sents), len(tokens)
    #sys.exit(0)

    data = read_dataset(all_sents, input_vocab_size, output_vocab_size, seq_len, input_tokenizer = input_tokenizer,
            output_tokenizer = output_tokenizer, style_remove_func=style_remove_func, rm_style_in = (style_remove_func is not None), type_tags=type_tags, extraencode=False)
    if shuffle:
        data.shuffle()
    return data


def read_romance_new(input_vocab_size = 10000, output_vocab_size = 10000, seq_len = 10, 
        input_tokenizer = None, output_tokenizer = None, max_load = 0, style_remove_func = None, 
        rom_filename="./data/romance_test_filtered_sample_1m.txt"):
    all_sents = []

    i = 0
    #for line in open("./data/romance_all.txt"):
    for line in open(rom_filename, "r"):
    #for line in open("./data/romance_20.txt"):
        if line.strip():
            all_sents.append(line)
            i+=1
            if i == max_load and max_load != 0:
                break

    return read_dataset(all_sents, input_vocab_size, output_vocab_size, seq_len, 
            input_tokenizer = input_tokenizer, output_tokenizer = output_tokenizer, 
            style_remove_func=style_remove_func, rm_style_in=(style_remove_func is not None))

def read_romance_stylenet(input_vocab_size = 10000, output_vocab_size = 10000, seq_len = 10, 
        input_tokenizer = None, output_tokenizer = None, max_load = 0, style_remove_func = None, train=True):
    all_sents = []

    i = 0
    if train:
        fin_name = "./data/stylenet/romantic/romantic_train.txt"
    else:
        fin_name = "./data/stylenet/romantic/romantic_train.txt"

    for line in open(fin_name, "r"):
        if line.strip():
            all_sents.append(line)
            i+=1
            if i == max_load and max_load != 0:
                break

    return read_dataset(all_sents, input_vocab_size, output_vocab_size, seq_len, 
            input_tokenizer = input_tokenizer, output_tokenizer = output_tokenizer, 
            style_remove_func=style_remove_func, rm_style_in=(style_remove_func is not None))

def get_stylenet_coco_joint_vocab(input_vocab_size = 10000, output_vocab_size = 10000, seq_len = 10, 
        input_tokenizer = None, output_tokenizer = None, max_load = 0, style_remove_func = None, train=True, flk=False):
    i = 0
    if train:
        fin_name = "./data/stylenet/romantic/romantic_train.txt"
    else:
        fin_name = "./data/stylenet/romantic/romantic_train.txt"

    all_sents = []
    for line in open(fin_name, "r"):
        if line.strip():
            all_sents.append(line)
            i+=1
            if i == max_load and max_load != 0:
                break

    tokens, _, _, _ = read_captions_images_raw(input_vocab_size, output_vocab_size, seq_len,
    input_tokenizer, output_tokenizer, val=False, test=False, max_size=max_load, get_feats=False, max_load=max_load,
    read_importance=False, full_dataset=True, flk=flk) 

    tokens = [' '.join(sent) for sent in tokens]

    all_tokens = []
    for sent in tokens:
        all_tokens.append(sent.encode('ascii', 'ignore'))
    tokens = all_tokens

    all_sents.extend(tokens)

    data = read_dataset(all_sents, input_vocab_size, output_vocab_size, seq_len, 
            input_tokenizer = input_tokenizer, output_tokenizer = output_tokenizer, 
            style_remove_func=style_remove_func, rm_style_in=(style_remove_func is not None))
    return data.enc_in.tokenizer, data.dec_out.tokenizer


def read_romance_new_importance(input_vocab_size = 10000, output_vocab_size = 10000, seq_len = 10, 
        input_tokenizer = None, output_tokenizer = None, max_load = 0):
    all_sents = []
    all_imp_sents = []

    i = 0
    #for line in open("./data/romance_all.txt"):
    js = json.load(open("./data/romance_test_filtered_sample_1m_importance.json", "r"))
    for row in js:
        all_sents.append(" ".join(row["tok"]))
        all_imp_sents.append(row["tok_imp"])
        i+=1
        if i == max_load and max_load != 0:
            break

    data = read_dataset(all_sents, input_vocab_size, output_vocab_size, seq_len, 
            input_tokenizer = input_tokenizer, output_tokenizer = output_tokenizer)
    data.important_tokens = all_imp_sent
    return data

def read_caption_color(input_vocab_size = 10000, output_vocab_size = 10000, seq_len = 10, 
        input_tokenizer = None, output_tokenizer = None, max_load = 0):
    all_sents = []

    i = 0
    #for line in open("./data/romance_all.txt"):
    for line in open("./data/coco_only_colors.txt"):
        if line.strip():
            all_sents.append(line)
            i+=1
            if i == max_load and max_load != 0:
                break

    return read_dataset(all_sents, input_vocab_size, output_vocab_size, seq_len, 
            input_tokenizer = input_tokenizer, output_tokenizer = output_tokenizer)

def read_humor_new(input_vocab_size = 10000, output_vocab_size = 10000, seq_len = 10, 
        input_tokenizer = None, output_tokenizer = None, max_load = 0):
    all_sents = []

    i = 0
    for line in open("./data/humor_all.txt"):
        if line.strip():
            all_sents.append(line)
            i+=1
            if i == max_load and max_load != 0:
                break

    return read_dataset(all_sents, input_vocab_size, output_vocab_size, seq_len, 
            input_tokenizer = input_tokenizer, output_tokenizer = output_tokenizer)
