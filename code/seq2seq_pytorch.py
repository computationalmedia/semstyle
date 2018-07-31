
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import sys

from collections import Counter
import json
import argparse
from torch.nn.utils.clip_grad import clip_grad_value_

from text_processing import tokenize_text, untokenize, pad_text, Toks

cuda = True
device = 0

rom_train_path = "/localdata/u4534172/SemStyleModelsNew/romance_wo_style.json"
coco_train_path = "./coco/coco_dataset_full_rm_style.json"

model_path = "./models/"
seq_to_seq_test_model_fname = "seq_to_txt_state.tar"
epoch_to_save_path = lambda epoch: model_path+"seq_to_txt_state_%d.tar" % int(epoch)

BATCH_SIZE=128
ROM_STYLE = "ROMANCETOKEN"
COCO_STYLE = "MSCOCOTOKEN"

def get_data(train=True, maxlines = -1, test_style=ROM_STYLE):
    input_text = []
    input_rems_text = []

    if train:
        js = json.load(open(rom_train_path, "r"))
        c = 0
        for line in js:
            sent = line[0]
            input_text.append(sent)
            rem_style = line[1]
            input_rems_text.append(rem_style + [ROM_STYLE])
            c+=1
            if maxlines > 0 and c == maxlines:
                break
        
    c = 0
    js = json.load(open(coco_train_path, "r"))
    for i, img in enumerate(js["images"]):
        if train and img["extrasplit"] == "val":
            continue
        if (not train) and img["extrasplit"] == "train":
            continue
        if maxlines > 0 and c == maxlines:
            break
        for sen in img["sentences"]:
            if train:
                input_rems_text.append(sen["rm_style_tokens"] + [COCO_STYLE])
            else:
                input_rems_text.append(sen["rm_style_tokens"] + [test_style])
            c+=1
            if maxlines > 0 and c == maxlines:
                break

            input_text.append(sen["tokens"])

    return input_text, input_rems_text

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        assert hidden_size % 2 == 0

        self.hidden_size = hidden_size
        self.input_size = input_size
        
        self.hidden_init_tensor = torch.zeros(2, 1, self.hidden_size/2, requires_grad=True)
        nn.init.normal_(self.hidden_init_tensor, mean=0, std=0.05)
        self.hidden_init = torch.nn.Parameter(self.hidden_init_tensor, requires_grad=True)
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.emb_drop = nn.Dropout(0.2)
        self.gru = nn.GRU(hidden_size, hidden_size/2, batch_first=True, bidirectional=True)
        self.gru_out_drop = nn.Dropout(0.2)
        self.gru_hid_drop = nn.Dropout(0.3)
        
    def forward(self, input, hidden, lengths):
        emb = self.emb_drop(self.embedding(input))
        pp = torch.nn.utils.rnn.pack_padded_sequence(emb, lengths, batch_first=True)
        out, hidden = self.gru(pp, hidden)
        out = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)[0]
        out = self.gru_out_drop(out)
        hidden = self.gru_hid_drop(hidden)
        return out, hidden
    
    def initHidden(self, bs):
        return self.hidden_init.expand(2, bs, self.hidden_size/2).contiguous()

class DecoderAttn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, out_bias):
        super(DecoderAttn, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.emb_drop = nn.Dropout(0.2)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.gru_drop = nn.Dropout(0.2)
        self.mlp = nn.Linear(hidden_size*2, output_size)
        if out_bias is not None:
            out_bias_tensor = torch.tensor(out_bias, requires_grad=False)
            self.mlp.bias.data[:] = out_bias_tensor
        self.logsoftmax = nn.LogSoftmax(dim=2)
        
        self.att_mlp = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_softmax = nn.Softmax(dim=2)
    
    def forward(self, input, hidden, encoder_outs):
        emb = self.embedding(input)
        out, hidden = self.gru(self.emb_drop(emb), hidden)
        
        out_proj = self.att_mlp(out)
        enc_out_perm = encoder_outs.permute(0, 2, 1)
        e_exp = torch.bmm(out_proj, enc_out_perm)
        attn = self.attn_softmax(e_exp)
        
        ctx = torch.bmm(attn, encoder_outs)
        
        full_ctx = torch.cat([self.gru_drop(out), ctx], dim=2)
        
        out = self.mlp(full_ctx)
        out = self.logsoftmax(out)
        return out, hidden, attn

def build_model(enc_vocab_size, dec_vocab_size, dec_bias = None, hid_size=512, loaded_state = None):
    enc = Encoder(enc_vocab_size, hid_size)
    dec = DecoderAttn(dec_vocab_size, hid_size, dec_vocab_size, dec_bias)
    if loaded_state is not None:
        enc.load_state_dict(loaded_state['enc'])
        dec.load_state_dict(loaded_state['dec'])
    if cuda:
        enc = enc.cuda(device=device)
        dec = dec.cuda(device=device)
    return enc, dec

def build_trainers(enc, dec, loaded_state=None):
    learning_rate = 0.001
    lossfunc = nn.NLLLoss(ignore_index=0)

    enc_optim = optim.Adam(enc.parameters(), lr=learning_rate)
    dec_optim = optim.Adam(dec.parameters(), lr=learning_rate)
    if loaded_state is not None:
        enc_optim.load_state_dict(load_state['enc_optim'])
        dec_optim.load_state_dict(load_state['dec_optim'])
    return enc_optim, dec_optim, lossfunc

def generate(enc, dec, enc_padded_text, L=20):
    enc.eval()
    dec.eval()
    with torch.no_grad():
        # run the encoder
        order, enc_pp, enc_lengths = make_packpadded(0, enc_padded_text.shape[0], enc_padded_text)
        hid = enc.initHidden(enc_padded_text.shape[0])
        out_enc, hid_enc = enc(enc_pp, hid, enc_lengths)
        
        hid_enc = torch.cat([hid_enc[0,:, :], hid_enc[1,:,:]], dim=1).unsqueeze(0)

        # run the decoder step by step
        dec_tensor = torch.ones((enc_padded_text.shape[0]), L+1, dtype=torch.long) * Toks.SOS
        if cuda:
            dec_tensor = dec_tensor.cuda(device=device)
        last_enc = hid_enc
        for i in xrange(L):
            out_dec, hid_dec, attn = dec.forward(dec_tensor[:,i].unsqueeze(1), last_enc, out_enc)
            out_dec[:, 0, Toks.UNK] = -np.inf # ignore unknowns
            #out_dec[torch.arange(dec_tensor.shape[0], dtype=torch.long), 0, dec_tensor[:, i]] = -np.inf
            chosen = torch.argmax(out_dec[:,0],dim=1)
            dec_tensor[:, i+1] = chosen
            last_enc = hid_dec
    
    return dec_tensor.data.cpu().numpy()[np.argsort(order)]

def make_packpadded(s, e, enc_padded_text, dec_text_tensor = None):

    text = enc_padded_text[s:e]
    lengths = np.count_nonzero(text, axis=1)
    order = np.argsort(-lengths)
    new_text = text[order]
    new_enc = torch.tensor(new_text)
    if cuda:
        new_enc = new_enc.cuda(device=device)

    
    if dec_text_tensor is not None:
        new_dec = dec_text_tensor[s:e][order].contiguous()
        leng = torch.tensor(lengths[order])
        if cuda:
            leng.cuda(device=device)
        return order, new_enc, new_dec, leng
    else:
        leng = torch.tensor(lengths[order])
        if cuda:
            leng.cuda(device=device)
        return order, new_enc, leng

def save_state(enc, dec, enc_optim, dec_optim, dec_idx_to_word, dec_word_to_idx, enc_idx_to_word, enc_word_to_idx, epoch):
    state = {'enc':enc.state_dict(), 'dec':dec.state_dict(),
             'enc_optim':enc_optim.state_dict(), 'dec_optim':dec_optim.state_dict(),
            'dec_idx_to_word':dec_idx_to_word, 'dec_word_to_idx':dec_word_to_idx,
            'enc_idx_to_word':enc_idx_to_word, 'enc_word_to_idx':enc_word_to_idx}
    torch.save(state, epoch_to_save_path(epoch))

def setup_test():
    if not cuda:
        loaded_state = torch.load(model_path+seq_to_seq_test_model_fname,
                map_location='cpu')
    else:
        loaded_state = torch.load(model_path+seq_to_seq_test_model_fname)

    enc_idx_to_word = loaded_state['enc_idx_to_word']
    enc_word_to_idx = loaded_state['enc_word_to_idx']
    enc_vocab_size = len(enc_idx_to_word)

    dec_idx_to_word = loaded_state['dec_idx_to_word']
    dec_word_to_idx = loaded_state['dec_word_to_idx']
    dec_vocab_size = len(dec_idx_to_word)

    enc, dec = build_model(enc_vocab_size, dec_vocab_size, loaded_state = loaded_state)

    return {'enc': enc, 'dec': dec, 'enc_idx_to_word':enc_idx_to_word, 'enc_word_to_idx':enc_word_to_idx,
        'enc_vocab_size':enc_vocab_size, 'dec_idx_to_word': dec_idx_to_word, 
        'dec_word_to_idx': dec_word_to_idx, 'dec_vocab_size':dec_vocab_size}

def test(setup_data, input_seqs = None, test_style=ROM_STYLE):

    if input_seqs is None:
        _, input_rems_text = get_data(train = False, test_style=test_style)
    else:
        input_rems_text = input_seqs
        slen = len(input_seqs)
        for i in xrange(slen):
            input_rems_text[i].append(test_style)

    _, _, enc_tok_text, _ = tokenize_text(input_rems_text,
        idx_to_word = setup_data['enc_idx_to_word'], word_to_idx = setup_data['enc_word_to_idx'])
    enc_padded_text = pad_text(enc_tok_text)

    dlen = enc_padded_text.shape[0]
    num_batch = dlen/BATCH_SIZE
    if dlen % BATCH_SIZE != 0:
        num_batch+=1
    res = []
    for i in xrange(num_batch):
        dec_tensor = generate(setup_data['enc'], setup_data['dec'], enc_padded_text[i*BATCH_SIZE:(i+1)*BATCH_SIZE])
        res.append(dec_tensor)

    all_text = []
    res = np.concatenate(res, axis=0)
    for row in res:
        utok = untokenize(row, setup_data['dec_idx_to_word'], to_text=True)
        all_text.append(utok)
    return all_text

    #for i in xrange(100):
    #    print "IN :", untokenize(enc_padded_text[i], enc_idx_to_word, to_text=True)
    #    print "GEN:", untokenize(dec_tensor[i], dec_idx_to_word, to_text=True), "\n"

def train():

    input_text, input_rems_text = get_data(train = True)

    dec_idx_to_word, dec_word_to_idx, dec_tok_text, dec_bias = tokenize_text(input_text, lower_case=True, vsize=20000)
    dec_padded_text = pad_text(dec_tok_text)
    dec_vocab_size = len(dec_idx_to_word)

    enc_idx_to_word, enc_word_to_idx, enc_tok_text, _ = tokenize_text(input_rems_text)
    enc_padded_text = pad_text(enc_tok_text)
    enc_vocab_size = len(enc_idx_to_word)

    dec_text_tensor = torch.tensor(dec_padded_text, requires_grad=False)
    if cuda:
        dec_text_tensor = dec_text_tensor.cuda(device=device)
    
    enc, dec = build_model(enc_vocab_size, dec_vocab_size, dec_bias = dec_bias)
    enc_optim, dec_optim, lossfunc = build_trainers(enc, dec)
    
    num_batches = enc_padded_text.shape[0] / BATCH_SIZE

    sm_loss = None
    enc.train()
    dec.train()
    for epoch in xrange(0, 13):
        print "Starting New Epoch: %d" % epoch
        
        order = np.arange(enc_padded_text.shape[0])
        np.random.shuffle(order)
        enc_padded_text = enc_padded_text[order]
        dec_text_tensor.data = dec_text_tensor.data[order]

        for i in xrange(num_batches):
            s = i * BATCH_SIZE
            e = (i+1) * BATCH_SIZE
            
            _, enc_pp, dec_pp, enc_lengths = make_packpadded(s, e, enc_padded_text, dec_text_tensor)

            enc.zero_grad()
            dec.zero_grad()
            
            hid = enc.initHidden(BATCH_SIZE)

            out_enc, hid_enc = enc.forward(enc_pp, hid, enc_lengths)
            
            hid_enc = torch.cat([hid_enc[0,:, :], hid_enc[1,:,:]], dim=1).unsqueeze(0)
            out_dec, hid_dec, attn = dec.forward(dec_pp[:,:-1], hid_enc, out_enc)

            out_perm = out_dec.permute(0, 2, 1)
            dec_text_tensor.shape
            loss = lossfunc(out_perm, dec_pp[:,1:])
            
            if sm_loss is None:
                sm_loss = loss.data
            else:
                sm_loss = sm_loss*0.95 + 0.05*loss.data

            loss.backward()
            clip_grad_value_(enc_optim.param_groups[0]['params'], 5.0)
            clip_grad_value_(dec_optim.param_groups[0]['params'], 5.0)
            enc_optim.step()
            dec_optim.step()
            
            #del loss
            if i % 100 == 0:
                print "Epoch: %.3f" % (i/float(num_batches) + epoch,), "Loss:", sm_loss
                print "GEN:", untokenize(torch.argmax(out_dec,dim=2)[0,:], dec_idx_to_word)
                #print "GEN:", untokenize(torch.argmax(out_dec,dim=2)[1,:], dec_idx_to_word)
                print "GT:", untokenize(dec_pp[0,:], dec_idx_to_word)
                print "IN:", untokenize(enc_pp[0,:], enc_idx_to_word)

                print torch.argmax(attn[0], dim=1)
                print "--------------"
        save_state(enc, dec, enc_optim, dec_optim, dec_idx_to_word, dec_word_to_idx, enc_idx_to_word, enc_word_to_idx, epoch)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    args = ap.parse_args()

    if args.train:
        train()
    else:
        r = setup_test()
        test(r)

if __name__ == "__main__":
    main()
