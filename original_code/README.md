Original code for SemStyle. It is suggested that you look at, and use, the pytorch rewrite as this theano version is overly complex and very challenging to get working. If you are intent of getting this working you will need the GRU with attention that is included in ```https://github.com/almath123/Lasagne.git``` under the `gru_attention` branch.

For training the term generator:

THEANO_FLAGS=mode=FAST_RUN,device=gpu2,floatX=float32,lib.cnmem=1 stdbuf -o 0 python gen_img_tags_fast.py train -o "./models/img_to_tags_framenet_02_11_17" --large --lemmatize --use_pos --extended_pos --framenet | tee ./models/img_to_tags_framenet_02_11_17.txt

For training the full model:

THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32,lib.cnmem=1 stdbuf -o 0 python gen_from_attn_fast.py train -o models/both_test_extrastrip_framenet_02_11_17_ --dataset both --large --lemmatize --use_pos --vocabsize 20000 --seqlen 22 --extended_pos --framenet |  tee models/both_test_extrastrip_framenet_02_11_17.txt

For testing the full model:

THEANO_FLAGS=mode=FAST_RUN,device=gpu2,floatX=float32,lib.cnmem=1 stdbuf -o 0 python gen_from_attn_fast.py test_img -i models/both_test_extrastrip_framenet_02_11_17_14_e.pik -I models/img_to_tags_framenet_02_11_1719_e.pik --large --num_test_batches 100000 -j models/both_test_extrastrip_framenet_04_11_17_14_e.json --lemmatize --use_pos --seqlen 22 --vocabsize 20000 --extended_pos --dataset both --framenet |  tee models/both_test_extrastrip_framenet_04_11_17_14_e.txt
