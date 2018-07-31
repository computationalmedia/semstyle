import keras
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model
import numpy as np
import glob
import os
import cPickle


def one_image():
    model = InceptionV3(weights='imagenet', include_top=False)

    img_path = "/data3/DataSet/COCO/val2014/COCO_val2014_000000000074.jpg"

    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    print x.shape
    print x.dtype
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    preds = np.mean(preds, axis=(1, 2))
    print preds
    print preds.shape

def all_image():
    model = InceptionV3(weights='imagenet', include_top=False)

    img_path = "/data3/DataSet/COCO/train2014/"
    filenames = glob.glob(img_path+"*.jpg")

    batch_size = 128
    x = np.zeros((batch_size, 299, 299, 3), dtype=np.float32)
    cur_pos = 0
    all_out = {}
    cur_fnames = []
    c = 0
    cmax = len(filenames)
    for fname in filenames:
        c += 1
        try:
            img = image.load_img(fname, target_size=(299, 299))
        except Exception as e:
            print e
            img = np.zeros((299, 299, 3), dtype=np.float32)
        x[cur_pos] = image.img_to_array(img)
        cur_fnames.append(os.path.basename(fname))
        
        if cur_pos == batch_size-1: 
            x = preprocess_input(x)

            preds = model.predict(x)
            preds = np.mean(preds, axis=(1, 2))
            for k,v in zip(cur_fnames,preds):
                all_out[k] = v

            cur_fnames = []
            #print preds
            print "%d/%d\n" %(c , cmax)
            cur_pos = 0
            x[:,:,:,:] = 0.0
        else:
            cur_pos += 1
    if cur_pos != 0:
        x = preprocess_input(x)

        preds = model.predict(x[:cur_pos])
        preds = np.mean(preds, axis=(1, 2))
        for k,v in zip(cur_fnames,preds):
            all_out[k] = v
    cPickle.dump(all_out, open("data/coco/coco_train_v3.pik", "wb"), protocol=2)


all_image()
