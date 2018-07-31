import os
import re
import sys
import json
import numpy as np
import argparse
import datetime
import cPickle

import theano
import theano.tensor as T

from data_reader import *
import lasagne as la
from lasagne.utils import floatX
from saveable_la_model import SaveableModel

sys.path.append(os.path.abspath("./util"))
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from lasagne.random import get_rng
from pycocoevalcap.bleu.bleu import Bleu
from tokenizer import TokenSequences, TextNormalizer, TextNormalizerOptions, TextTokenizer

from style_remover import remove_style2, remove_style_text
#def remove_style2(data, *args, **kwargs):
#    return data

class GenImgTags(object):

    def __init__(self, iargs):
        self.imp = {}

        self.iargs = iargs
        if iargs.large:
            self.imp["EMB_SIZE"] = 512
            self.imp["HIDDEN_SIZE"] = 512
            self.imp["BATCH_SIZE"] = 128
            self.imp["SPACY_THREADS"] = 12
            self.imp["SPACY_BATCH"] = 10000
            self.max_load = 0
        elif iargs.small:
            self.imp["EMB_SIZE"] = 128
            self.imp["HIDDEN_SIZE"] = 64
            self.imp["BATCH_SIZE"] = 128
            self.imp["SPACY_THREADS"] = 2
            self.imp["SPACY_BATCH"] = 2000
            self.max_load = 1000
        elif os.path.exists("use_gpu.config") or self.iargs.gpu:
            self.imp["EMB_SIZE"] = 353
            self.imp["HIDDEN_SIZE"] = 191
            self.imp["BATCH_SIZE"] = 128
            self.imp["SPACY_THREADS"] = 8
            self.imp["SPACY_BATCH"] = 10000
            self.max_load = 0
        else:
            self.imp["EMB_SIZE"] = 128
            self.imp["HIDDEN_SIZE"] = 64
            self.imp["BATCH_SIZE"] = 128
            self.imp["SPACY_THREADS"] = 2
            self.imp["SPACY_BATCH"] = 2000
            self.max_load = 1000

        self.imp["WORD_DROPOUT"] = 0.0
        self.imp["WORD_DROPOUT_IN"] = 0.25

        self.imp["IMG_EMB_SIZE"] = 2048
        self.imp["SEQ_LEN"] = 16
        self.imp["IN_VOCAB"] = 10000
        self.imp["OUT_VOCAB"] = 10000
        self.imp["GRAD_CLIP"] = 5.0
        self.imp["LEARNING_RATE"] = 0.001
        self.imp["DROPOUT_RATE"] = 0.5
        self.imp["REGULARIZE_WEIGHT"] = 1e-6
        self.imp["READ_IMP"] = False

        if hasattr(self.iargs, "lr") and self.iargs.lr:
            self.imp["LEARNING_RATE"] = self.iargs.lr

    class EncDecModel(SaveableModel):
        def __init__(self, args, imp):
            self.model_is_built = False
            self.iargs = args
            self.imp = imp
            super(GenImgTags.EncDecModel, self).__init__()

        def build(self, in_vocab_size, out_vocab_size, word_count, temperature=1, gum_temp=1.0):
            self.in_vocab_size = in_vocab_size
            self.out_vocab_size = out_vocab_size
            self.word_count = word_count
            self.var["temp"] = theano.shared(np.float32(gum_temp), "temp")
            self.var["dec_in"] = T.imatrix()
            self.net["l_dec_in"] = la.layers.InputLayer((self.imp["BATCH_SIZE"], self.imp["SEQ_LEN"]), input_var = self.var["dec_in"])

            self.var["img_input"] = T.matrix()
            self.net["l_img_input"] = la.layers.InputLayer((self.imp["BATCH_SIZE"], self.imp["IMG_EMB_SIZE"]), input_var = self.var["img_input"])

            # image encoding
            self.net["l_img_emb_one"] = la.layers.DenseLayer(la.layers.dropout(self.net["l_img_input"], self.imp["DROPOUT_RATE"]), 
                    nonlinearity=la.nonlinearities.tanh, num_units=self.imp["HIDDEN_SIZE"])
            hidden_input = self.net["l_img_emb_one"]

            # decoder
            self.net["l_emb_in"] = la.layers.EmbeddingLayer(self.net["l_dec_in"], self.out_vocab_size, self.imp["HIDDEN_SIZE"])
            self.net["l_dec_gru"] = la.layers.GRULayer(self.net["l_emb_in"], num_units=self.imp["HIDDEN_SIZE"], hid_init=hidden_input, 
                    grad_clipping=floatX(self.imp["GRAD_CLIP"]), learn_init=True)

            self.net["l_dec_gru_rs"] = la.layers.ReshapeLayer(self.net["l_dec_gru"], (-1, self.imp["HIDDEN_SIZE"]))
            self.net["l_out"] = la.layers.DenseLayer(self.net["l_dec_gru_rs"], num_units = self.out_vocab_size, nonlinearity=la.nonlinearities.softmax)

            self.model_is_built = True

        def build_generator(self):
            self.expr["out"] = la.layers.get_output(self.net["l_out"], deterministic=True and (not self.iargs.force_nondet))

            vars_in = [self.var["img_input"], self.var["dec_in"]]
            generate = theano.function(vars_in, self.expr["out"])
            self.th_generate = generate
            return generate

        def build_trainer_image(self):
            self.var["dec_mask_out"] = T.matrix()
            self.var["dec_out"] = T.imatrix()

            to_regularize = [self.net["l_img_emb_one"]]#, self.net["l_img_emb_two"]]

            self.expr["out"] = la.layers.get_output(self.net["l_out"], deterministic=False)

            self.expr["reg"] = la.regularization.regularize_layer_params(to_regularize, la.regularization.l2)
            self.expr["cce"] = la.objectives.categorical_crossentropy(self.expr["out"], self.var["dec_out"].ravel())
            self.expr["cost"] = la.objectives.aggregate(self.expr["cce"], weights=self.var["dec_mask_out"].ravel(), mode="normalized_sum") 
            
            self.expr["loss"] = self.expr["cost"] + (self.expr["reg"] * self.imp["REGULARIZE_WEIGHT"])

            all_params = la.layers.get_all_params(self.net["l_out"], trainable=True)
            if self.iargs.sgd:
                updates = la.updates.sgd(self.expr["loss"], all_params, floatX(self.imp["LEARNING_RATE"]))
            else:
                updates = la.updates.adam(self.expr["loss"], all_params, floatX(self.imp["LEARNING_RATE"]))

            vars_in = [self.var["img_input"], self.var["dec_in"], self.var["dec_out"], self.var["dec_mask_out"]]
            vars_out = [self.expr["cost"], self.expr["reg"]]

            train = theano.function(vars_in, vars_out, updates=updates)

            self.th_train_img = train
            return train


        def set_train_data(self, data):
            self.data = data

        def set_test_data(self, test_data):
            self.test_data = test_data

        def word_dropout(self, seq, mask, token, drop_frac = -1):
            if drop_frac == -1:
                drop_frac = self.imp["WORD_DROPOUT"]
            if drop_frac > 0.0:
                chosen = np.random.random(seq.shape) < drop_frac
                chosen = np.logical_and(chosen, mask > 0.5)
                seq[chosen] = token
            return seq

        def train_image(self, data, save_name = None, save_object = {}, degen_image=False):
            if degen_image:
                from nltk.corpus import stopwords
                sw = stopwords.words('english')
                sw_idx = []
                for w in sw:
                    if w in self.test_data.dec_in.tokenizer.word_to_index:
                        sw_idx.append(data.dec_in.tokenizer.word_to_index[w])
                sw_idx = np.array(sw_idx)
                aw_bvec = np.ones((data.dec_in.tokenizer.get_vocab_size(),), dtype=np.int32)
                aw_bvec[sw_idx] = 0

            num_examples = data.dec_in.get_num_seqs()
            num_batches = num_examples / self.imp["BATCH_SIZE"]
            smooth_loss = 0
            for epoch in xrange(20):
                data.shuffle()
                #self.generate(allow_unk=False)
                print "Epoch over:",
                self.val_score(0, 10000000)

                for i in xrange(num_batches):
                    s = i * self.imp["BATCH_SIZE"]
                    e = (i+1) * self.imp["BATCH_SIZE"]

                    img_emb_in = data.get_img_emb(s, e)

                    dec_in = data.dec_in.get_padded_seqs(s, e)
                    dec_out = data.dec_out.get_padded_seqs(s, e)
                    dec_out_mask = data.dec_out.get_masks(s, e, dtype=theano.config.floatX)

                    dec_in_mask = data.dec_in.get_masks(s, e, dtype=theano.config.floatX)
                    
                    rvals = self.th_train_img(img_emb_in, dec_in , dec_out, dec_out_mask)
                    cur_loss = rvals[0]

                    if epoch == 0 and i == 0:
                        smooth_loss = cur_loss
                    smooth_loss = smooth_loss * 0.95 + cur_loss*0.05
                    epoch_frac = epoch + (i / float(num_batches))
                    print rvals
                    print "Epoch %f, Loss %f, Smooth Loss %f" % (epoch_frac, cur_loss, smooth_loss)
                    print ""
                    if i % 1000 == 0:
                        #self.generate(allow_unk=False)
                        self.val_score(0, 10)

                        save_object = self.save_to_object(save_object)
                        filename = save_name + "%02d_e.pik" % epoch
                        cPickle.dump(save_object, open(filename, "wb"), protocol=2)
                if save_name is not None:
                    save_object = self.save_to_object(save_object)
                    filename = save_name + "%02d_e.pik" % epoch
                    cPickle.dump(save_object, open(filename, "wb"), protocol=2)

        def val_score(self, s_start = 0, num_batches=2):
            bs = self.imp["BATCH_SIZE"]
            bleu = Bleu()
            eval_store_gen = {}
            eval_store_gt = {}
            num_examples = self.test_data.dec_in.get_num_seqs()
            max_num_batches = num_examples / bs
            for i in xrange(min(num_batches, max_num_batches)):
                s = s_start+bs*i
                e = s_start+bs*(i+1)
                gen_txt = self.generate(s=s, allow_unk=False)
                gt_txt = self.test_data.dec_out.get_text(s, e)
                fnames = self.test_data.filenames[s:e]
                for g, f in zip(gen_txt, fnames):
                    if f not in eval_store_gen:
                        eval_store_gen[f] = [" ".join(g)]

                for g, f in zip(gt_txt, fnames):
                    if f not in eval_store_gt:
                        eval_store_gt[f] = []
                    eval_store_gt[f].append(" ".join(g))
            print bleu.compute_score(eval_store_gt, eval_store_gen)[0]

                



        def generate(self, s = 0, allow_unk=True, no_feedback=False, sample=False, force_good = False, output_tokenizer=None):
            # NB: assumes output and input tokenizers are the same

            if output_tokenizer is not None:
                out_tokenizer = output_tokenizer
            else:
                out_tokenizer = self.test_data.dec_out.tokenizer

            bs = self.imp["BATCH_SIZE"]
            e = s + bs
            from nltk.corpus import stopwords
            sw = stopwords.words('english')
            sw_idx = []
            for w in sw:
                if w in out_tokenizer.word_to_index:
                    sw_idx.append(out_tokenizer.word_to_index[w])
            sw_idx = np.array(sw_idx)

            img_in = self.test_data.get_img_emb(s, e)
            dec_in = np.zeros((self.imp["BATCH_SIZE"], self.imp["SEQ_LEN"]), dtype=np.int32)
            for i in xrange(self.imp["SEQ_LEN"]):
                if no_feedback:
                    dec_in_tmp = np.ones_like(dec_in) * out_tokenizer.special_token_index[WORD_DROP_TOKEN]
                else:
                    dec_in_tmp = dec_in

                res = self.th_generate(img_in, dec_in_tmp)
                res = res.reshape((self.imp["BATCH_SIZE"], self.imp["SEQ_LEN"], self.out_vocab_size))
                if i+1 < self.imp["SEQ_LEN"]:
                    if allow_unk == False:
                        unk_id = out_tokenizer.special_token_index["UNKNOWN"]
                        res[:, i, unk_id] = 0.0
                    if force_good:
                        print "Before:", res[:, i, 0]
                        res[:, :4, 0] = 0.0 # must be at least this long
                        if i > 0: 
                            # ignore duplicate non-stopwords
                            seen = dec_in[:, :(i+1)]
                            res_saved = np.array(res[:, i, :], dtype=theano.config.floatX)
                            print 'res_saved before', res_saved[:,0]
                            print res_saved.shape
                            res[:, i, seen] = 0
                            res[:, i, sw_idx] = res_saved[:, sw_idx]
                            # ignore duplicate adjacent words of any kind
                            res[:, i, dec_in[:, i]] = 0
                            res[:, i, 0] = res_saved[:, 0]
                            print res[:, i, 0]
                            print 'res_saved after', res_saved[:, 0]
                    if sample:
                        for j in xrange(dec_in.shape[0]):
                            temp = 0.5
                            res[j, i] = res[j, i] ** (1.0/temp)
                            res[j, i] /= np.sum(res[j, i])
                            dec_in[j, i+1] = np.random.choice(res.shape[2], p=res[j, i])
                    else:
                        dec_in[:, i+1] = np.argmax(res[:, i], axis=1)

            gen_seq = TokenSequences(out_tokenizer, reverse=False, start_pad=True, seq_len = self.imp["SEQ_LEN"])
            gen_seq.from_padded_seqs(dec_in, is_reversed=False, is_start_pad=True)

            enc_txt = self.test_data.enc_in.get_text(s, s+bs)

            for txt_in, txt_out in zip(enc_txt, gen_seq.get_text()):
                print "IN :", txt_in
                print "OUT:", txt_out
                print ""

            return gen_seq.get_text()

        def _process_word_count(self, word_count):
            wc = np.array(word_count, dtype=theano.config.floatX)
            wc = np.log(wc/np.sum(wc))
            wc -= np.median(wc)
            return wc

    def train(self):
        load_model = False
        input_tokenizer = None
        output_tokenizer = None
        if self.iargs.infile:
            save_model = cPickle.load(open(self.iargs.infile, "rb"))
            load_model = True
            input_tokenizer = save_model['input_tokenizer']
            output_tokenizer = save_model['output_tokenizer']

        pos_list = ["ADJ", "PRT", "PUNCT", "ADV", "PRON", "CONJ"]

        if self.iargs.extended_pos:
            pos_list.extend(["ADP", "VERB", "X", "NUM", "DET", "PROPN"])
        elif self.iargs.extended_pos_wverbs:
            pos_list.extend(["ADP", "X", "NUM", "DET", "PROPN"])

        style_remove_func = lambda x : remove_style_text(x, threads=self.imp["SPACY_THREADS"], batch_size=self.imp["SPACY_BATCH"], lemmatize=self.iargs.lemmatize, append_pos = args.use_pos, pos_list = pos_list, framenet=self.iargs.framenet)

        data = read_captions_images(self.imp["IN_VOCAB"], self.imp["OUT_VOCAB"], self.imp["SEQ_LEN"], val=False, max_load = self.max_load, read_importance=self.imp["READ_IMP"], full_dataset=True, input_tokenizer = input_tokenizer, output_tokenizer = output_tokenizer, style_remove_func=style_remove_func, rm_style_out=True)
        #data = remove_style2(data, threads=self.imp["SPACY_THREADS"], batch_size=self.imp["SPACY_BATCH"], encoder=False)
        if not self.iargs.infile:
            print "dec_out:"
            print data.dec_out.tokenizer.word_counter.get_top_k_words(100)
            print "Size:", data.dec_out.tokenizer.get_vocab_size(), "\n"
            print "dec_in:"
            print data.dec_in.tokenizer.word_counter.get_top_k_words(100)
            print "Size:", data.dec_in.tokenizer.get_vocab_size(), "\n"
            print "enc_in:"
            print data.enc_in.tokenizer.word_counter.get_top_k_words(100)
            print "Size:", data.enc_in.tokenizer.get_vocab_size(), "\n"

        data_val = read_captions_images(self.imp["IN_VOCAB"], self.imp["OUT_VOCAB"], self.imp["SEQ_LEN"], val=True, 
                max_load = self.max_load, read_importance=self.imp["READ_IMP"], 
                input_tokenizer=data.dec_in.tokenizer, output_tokenizer=data.dec_out.tokenizer, full_dataset=True,
                style_remove_func=style_remove_func, rm_style_out=True)

        #data_val = remove_style2(data_val, threads=self.imp["SPACY_THREADS"], batch_size=self.imp["SPACY_BATCH"], encoder=False)

        model = self.EncDecModel(self.iargs, self.imp)
        model.set_test_data(data_val)
        model.build(in_vocab_size = data.dec_in.get_vocab_size(), out_vocab_size=data.dec_out.get_vocab_size(), 
                word_count=data.dec_out.tokenizer.get_word_count(), temperature=1)
        model.build_trainer_image()
        if load_model:
            model.load_from_object(save_model)
        model.build_generator()

        if self.iargs.infile is not None:
            model.load_from_object(save_model)

        save_object = {}
        save_object = data.save_tokenizer(save_object)
        model.train_image(data, save_name = self.iargs.outfile, save_object = save_object, degen_image=self.iargs.degenerate_img_model)

    def test(self):
        save_model = cPickle.load(open(self.iargs.infile, "rb"))

        data = read_captions_images(self.imp["IN_VOCAB"], self.imp["OUT_VOCAB"], self.imp["SEQ_LEN"], val=True,
            input_tokenizer = save_model['input_tokenizer'], output_tokenizer = save_model['output_tokenizer'], max_load=self.max_load, full_dataset=True)
        data.shuffle()

        model = self.EncDecModel(self.iargs, self.imp)
        model.set_test_data(data)
        model.build(in_vocab_size = data.dec_in.get_vocab_size(), out_vocab_size=data.dec_out.get_vocab_size(), 
                word_count=data.dec_out.tokenizer.get_word_count())
        model.build_generator()
        model.load_from_object(save_model)
        model.generate(allow_unk=False)

        bs = self.imp["BATCH_SIZE"]
        num_examples = data.img.shape[0]
        num_batch = num_examples / bs
        output_data = []
        seen = set()
        for b in xrange(num_batch):
            s = b*bs
            e = (b+1)*bs
            out_txt = model.generate(s, allow_unk=False)
            fnames = data.filenames[s:e]
            for o,f in zip(out_txt, fnames):
                if f not in seen:
                    seen.add(f)
                    #image_id = int(re.findall("_0*?([1-9][0-9]*?)\.jpg", f)[0])
                    image_id = f.strip()
                    output_data.append({"caption":" ".join(o), "image_id":image_id})
        json.dump(output_data, open(args.json, "w"))

    def setup_test_from_input(self):
        save_model = cPickle.load(open(self.iargs.infile, "rb"))

        self.model = self.EncDecModel(self.iargs, self.imp)
        self.model.build(in_vocab_size = save_model['input_tokenizer'].get_vocab_size(), 
                out_vocab_size = save_model['output_tokenizer'].get_vocab_size(), 
                word_count= save_model['output_tokenizer'].get_word_count())
        self.model.build_generator()
        self.model.load_from_object(save_model)
        self.loaded_save_model = save_model
        #model.generate(allow_unk=False)

    def test_from_input(self, data, s):
        self.model.set_test_data(data)
        return self.model.generate(allow_unk=False, output_tokenizer = self.loaded_save_model['output_tokenizer'], s = s)
        #save_model = cPickle.load(open(self.iargs.infile, "rb"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--outfile", default="./models/def_model_saved", help="where to save the model file")
    ap.add_argument("-j", "--json", help="where to save generated output as json")
    ap.add_argument("-i", "--infile", help="the input model file")
    ap.add_argument("-I", "--infile2", help="the second input model file")
    ap.add_argument("--sgd", action="store_true", help="learn using sgd")
    ap.add_argument("--lr", type=float, help="use the learning rate")
    ap.add_argument("--gpu", action="store_true", help="use gpu params")
    ap.add_argument("--large", action="store_true", help="use large param")
    ap.add_argument("--small", action="store_true", help="use small param")
    ap.add_argument("--force_nondet", action="store_true", help="force the use of non-deterministic layers during generation: for debugging")
    ap.add_argument("--degenerate_img_model", action="store_true")
    ap.add_argument("--lemmatize", action="store_true", help="remove word morphology information")
    ap.add_argument("--use_pos", action="store_true", help="add pos tags to input text")
    ap.add_argument("--extended_pos", action="store_true", help="use the more aggressive pos tag removal")
    ap.add_argument("--extended_pos_wverbs", action="store_true", help="use the extended pos list with added verbs")
    ap.add_argument("--framenet", action="store_true", help="reduce verbs to their framenet tags")
    ap.add_argument("mode", choices = ["train", "test"])
    args = ap.parse_args()
    gen = GenImgTags(args)

    if args.mode == "train":
        gen.train()
    elif args.mode == "test":
        gen.test() 
    else:
        pass
