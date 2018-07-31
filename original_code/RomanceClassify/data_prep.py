import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer

DATA_PATH = "/home/amathews/NewProjects/GenCont/data/"

def read_coco(fname = DATA_PATH+"coco/dataset.json"):
    lines = []
    js = json.load(open(fname, "r"))
    splits = dict()
    for img in js["images"]:
        if img["split"] not in splits:
            splits[img["split"]] = 0
        splits[img["split"]] += 1
        
        if img["split"] != "train":
            continue
        
        img_to_lines = []
        for sent in img["sentences"]:
            img_to_lines.append(sent["raw"])
        lines.append((img["filename"], img_to_lines))
        
    coco_text = [s for im in lines for s in im[1]]
    return coco_text

def read_rom(fname = DATA_PATH+"romance_test_filtered_sample_1m.txt"):
    lines = []
    for line in open(fname, "r"):
        lines.append(line.strip())
    return lines

def build_text_dataset(coco_text, rom_text):
    X_text = coco_text + rom_text
    Y = np.zeros(len(X_text), dtype=np.int32)
    Y[len(coco_text):] = 1
    return X_text, Y

def get_train_test(X_text, Y):
    np.random.seed(123)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X_text, Y, np.arange(len(X_text)), train_size=0.9, shuffle=True, stratify=Y)
    return X_train, X_test, y_train, y_test, idx_train, idx_test

class FeatureExtractor(object):
    def __init__(self, count=True, hashing=False):
        self.count = count
        self.hashing = hashing
        self.was_fit = False

    def fit(self, X_text):
        if self.count:
            self.cv = CountVectorizer()
            self.cv.fit(X_text)
        if self.hashing:
            self.hv = HashingVectorizer(ngram_range=(1,2), norm=None, alternate_sign=False, binary=True)
            self.hv.fit(X_text)

        self.was_fit = True
        return

    def transform(self, X_text):
        assert self.was_fit
        if self.count:
            X_count = self.cv.transform(X_text) 
            X = X_count
        if self.hashing:
            X_hashing = self.hv.transform(X_text)
            X = X_hashing
        if self.hashing and self.count:
            X = np.hstack([X_count, X_hashing])
        return X

def read_mymethod(fname = "../CrowdFlower/romance_attn_attlm_01_08_17.json"):
    data = json.load(open(fname, "r"))
    df = pd.DataFrame(data)
    df = df.drop_duplicates(subset=["fname"], keep="first")
    df = df.reset_index(drop=True)
    text_out = df["out"]
    return text_out

def read_showtell(fname = "../CrowdFlower/gen_img_sent_all_test_26_07_17.json"):
    data_showtell = json.load(open(fname, "r"))
    df_showtell = pd.DataFrame(data_showtell)
    df_showtell['fname'] = df_showtell["image_id"].map(lambda x: "COCO_val2014_%012d.jpg" % x)
    df_showtell = df_showtell.drop_duplicates(subset=["fname"], keep="first")
    df_showtell = df_showtell.reset_index(drop=True)
    text_showtell = df_showtell["caption"]
    return text_showtell
    
