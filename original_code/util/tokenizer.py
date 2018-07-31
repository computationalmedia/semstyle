import string
import nltk
import collections
import copy
import numpy as np
import theano
import spacy

class TextNormalizerOptions(object):
    """ Configuration information for the sentence splitter and normalizer"""
    def __init__(self, lowercase=True, remove_punct=True, remove_empty=True, replace_numbers=True,
            words_to_preserve=set(), nltk_split=True):
        assert isinstance(words_to_preserve, collections.Iterable)
        self.lowercase = lowercase
        self.remove_punct = remove_punct
        self.remove_empty = remove_empty
        self.words_to_preserve = set([w for w in words_to_preserve])
        self.replace_numbers = replace_numbers
        if self.replace_numbers:
            self.words_to_preserve.add("NUMBER")
        self.nltk_split = nltk_split
        if nltk_split:
            self.tokenizer = nltk.tokenize.treebank.TreebankWordTokenizer()
            self.tok = lambda s: self.tokenizer.tokenize(s)
        else:
            self.tok = lambda s: s.split()

    def addWordToPreserve(self, word):
        self.words_to_preserve.add(word)

    def getLowercase(self):
        return self.lowercase

    def getRemovePunct(self):
        return self.remove_punct

    def getRemoveEmpty(self):
        return self.remove_empty

    def isReplaceNumbers(self):
        return self.replace_numbers

    def isWordToPreserve(self, word):
        return (word in self.words_to_preserve)

    def isNLTKSplit(self):
        return self.nltk_split

    def word_tokenize(self, s):
        return self.tok(s)

class TextNormalizer(object):

    """Splits and normalizes sentences"""

    _punc = ''.join([c for c in string.punctuation if c not in ",.\""])
    _ttable = string.maketrans("","")

    @classmethod
    def split_sentence(cls, sent, options=None):
        """Splits a sentence into its individual words"""
        if options is not None and not options.isNLTKSplit():
            return sent.split()
        else:
            return options.word_tokenize(sent)

    @classmethod
    def normalize_word(cls, w, options = TextNormalizerOptions()):
        wout = str(w.decode('utf-8', 'ignore').encode('ascii','ignore').strip())
        if options.isReplaceNumbers():
            if wout.isdigit():
                wout = "NUMBER"
        if not options.isWordToPreserve(wout):
            if options.getLowercase():
                wout = str.lower(wout)
            if options.getRemovePunct():
                wout = wout.translate(cls._ttable, cls._punc + string.digits)
            wout = wout.strip()

        return str(wout)

    @classmethod
    def normalize_sentence(cls, sent, options = TextNormalizerOptions()):
        #break the sentence into words
        if type(sent) is list:
            sent_splt = sent
        else:
            sent_enc = str(sent.decode('utf-8', 'ignore').encode('ascii','ignore').strip())
            sent_splt = cls.split_sentence(sent_enc, options)

        # normalize each of the words
        sent_norm = []
        for w in sent_splt:
            wn = cls.normalize_word(w, options)
            if not wn and options.getRemoveEmpty():
                continue
            sent_norm.append(wn)
        return sent_norm

    @classmethod
    def normalize_text(cls, text, options = TextNormalizerOptions()):
        text_norm = []
        for sent in text:
            sent_norm = cls.normalize_sentence(sent, options)
            text_norm.append(sent_norm)

        return text_norm

#def normalize_text_th(text, options, threads):

class WordCounter(object):
    """Counts words in already split and normalized sentences"""

    def __init__(self):
        self.word_count = {}
        self.sentence_count = 0

    def clear(self):
        self.word_count = {}
        self.sentence_count = 0

    def fit(self, split_sentences):
        """ Count the words, can be called multiple times to
        create an aggregate """
        for split_sentence in split_sentences:
            self.sentence_count += 1
            for word in split_sentence: 
                if word not in self.word_count:
                    self.word_count[word] = 0
                self.word_count[word] += 1

    def get_counts(self):
        return self.word_count

    def get_sentence_count(self):
        return self.sentence_count

    def get_counts_for_selected(self, selected_words, default_count=0):
        assert isinstance(selected_words, list)

        counts = []
        for w in selected_words:
            if w in self.word_count:
                counts.append(self.word_count[w])
            else:
                counts.append(default_count)

        return counts

    def get_sum_counts_unselected(self, selected_words):
        total_count = np.sum(self.word_count.values())
        unsel_count = total_count - np.sum(self.get_counts_for_selected(selected_words, default_count = 0))
        unsel_count = max(1, unsel_count)
        return unsel_count

    def get_top_k_words(self, k):
        wc = sorted(self.word_count.items(), key=lambda x: x[1], reverse=True)
        return [w for w,c in wc][:k]


class TextTokenizer(object):
    """ Tokenizes already split and normalized sentences """

    def __init__(self, max_words=8000, extra_special_tokens=[]):
        self.word_to_index = {}
        self.index_to_word = []
        self.max_words = max_words
        self.word_counter = WordCounter()

        self.special_tokens = ["", "UNKNOWN"] + extra_special_tokens
        self.special_token_index = dict([(w, i) for i, w in enumerate(self.special_tokens)])

    def get_word_to_index(self, word):
        return self.word_to_index[word]

    def has_word(self, word):
        return word in self.word_to_index

    def translate_token_to(self, token, target_tokenizer):
        word = self.index_to_word[token]
        if target_tokenizer.has_word(word):
            return target_tokenizer.get_word_to_index(word)
        else:
            return target_tokenizer.get_word_to_index("UNKNOWN")

    
    def save_to_object(self, save_object = {}):
        save_object["tokenizer"] = save_object
        return save_object

    def get_vocab_size(self):
        return len(self.index_to_word)

    def get_word_count(self):
        counts = self.word_counter.get_counts_for_selected(self.index_to_word)
        unknown_index = self.special_token_index["UNKNOWN"]
        counts[unknown_index] = self.word_counter.get_sum_counts_unselected(self.index_to_word)
        empty_index = self.special_token_index[""]
        counts[empty_index] = self.word_counter.get_sentence_count()
        for i in self.special_token_index.values():
            if counts[i] == 0:
                counts[i] = 1
        return counts

    def clear(self):
        self.word_counter.clear()
        self.word_to_index = {}
        self.index_to_word = []

    def fit(self, text):
        """ Builds a token dictionary, can be called multiple times to create a
        token set based on the aggeregated counts
        NOTE: requires text to already be split and normalized"""
        self.word_counter.fit(text)

        num_free_words = self.max_words - len(self.special_tokens)
        top_words = self.word_counter.get_top_k_words(num_free_words)

        self.index_to_word = self.special_tokens + top_words
        self.word_to_index = dict([(w, i) for i, w in enumerate(self.index_to_word)])

    def text_to_seqs(self, text):
        seqs = []
        for sentence in text:
            seq = []
            for w in sentence:
                if w in self.word_to_index:
                    seq.append(self.word_to_index[w])
                else:
                    seq.append(self.special_token_index["UNKNOWN"])
            seqs.append(seq)
        return seqs

    def seqs_to_text(self, seqs):
        text = []
        for seq in seqs:
            sentence = []
            for tok in seq:
                sentence.append(self.index_to_word[tok]) 
            text.append(sentence)
        return text

class TextTokenizerSpacy(TextTokenizer):
    nlp = None

    def __init__(self, nlp=None, max_words=8000, change_words=0):
        super(TextTokenizerSpacy, self).__init__(max_words=max_words)
        self.change_words = change_words
        self.spacy_offset = len(self.special_tokens) + self.change_words
        if nlp is None and TextTokenizerSpacy.nlp is None:
            TextTokenizerSpacy.nlp = spacy.load('en')
        elif nlp is not None:
            TextTokenizerSpacy.nlp = nlp

    def fit(self, text):
        """ Builds a token dictionary, can be called multiple times to create a
        token set based on the aggeregated counts
        NOTE: requires text to already be split and normalized"""
        self.word_counter.fit(text)

        num_free_words = self.max_words - len(self.special_tokens) - self.change_words

        top_words = self.word_counter.get_top_k_words(self.change_words)
        added_words = self.special_tokens + top_words

        #spacy_words = dict([(tok.orth + self.spacy_offset, tok.orth_.encode('ascii', 'ignore')) 
        #     if tok.has_vector])

        # get index_to_word for spacy
        spacy_words = {}
        spacy_words_set = set()
        added_words_set = set(added_words)
        for tok in TextTokenizerSpacy.nlp.vocab:
            if not tok.has_vector:
                continue
            new_word = tok.orth_.encode('ascii', 'ignore')
            
            # add the word if it is not a duplicate
            if new_word not in spacy_words_set and new_word not in added_words_set:
                spacy_words_set.add(new_word)
                spacy_words[tok.orth + self.spacy_offset] = new_word

        # add the top_k_words and the special tokens to the spacy index_to_words
        spacy_words.update(dict([(i, str(v)) for i, v in enumerate(added_words)]))

        self.index_to_word = spacy_words

        for i in xrange(10):
            if i in self.index_to_word:
                print i, self.index_to_word[i]

        self.word_to_index = dict([(w, i) for i, w in self.index_to_word.items()])
        for i, v in enumerate(added_words):
            self.word_to_index[v] = i
        print "Blank word index:", self.word_to_index[""]
        #sys.exit(0)

        self.vocab_size = np.amax(self.index_to_word.keys())+1
        self.added_words = added_words

    def get_vectors(self):
        if TextTokenizerSpacy.nlp is None:
            TextTokenizerSpacy.nlp = spacy.load('en')
        max_has_vec = 0
        for v in self.nlp.vocab:
            if v.has_vector:
                max_has_vec = max(max_has_vec, v.orth)
        print max_has_vec

        emb_num = int(max_has_vec + self.spacy_offset + 1)
        emb_size = TextTokenizerSpacy.nlp.vocab.vectors_length
        
        emb = np.zeros((emb_num, emb_size), dtype=theano.config.floatX)
        for v in TextTokenizerSpacy.nlp.vocab:
            if v.has_vector:
                emb[v.orth + self.spacy_offset] = v.vector

        return emb

    def get_vocab_size(self):
        return self.vocab_size

class SequencePadder(object):
    def __init__(self):
        raise NotImplemented("SequencePadder can't be instantiated")

    @staticmethod
    def remove_padding(X, reverse=False, start_pad=False):
        X_out = []
        for x in X:

            #remove the 0 at the start
            if start_pad:
                assert x[0] == 0
                x = x[1:]

            x_res = []
            for v in x:
                if v == 0:
                    break
                x_res.append(v)

            if reverse:
                x_res = x_res[::-1]

            X_out.append(x_res)
        return X_out

    @staticmethod
    def get_padded(X, max_len, reverse=False, start_pad=True):
        """
        Makes sentences all the same length by:
            - Adding zeros to the end of the sentences.
            - Enforcing a maximum text len by cropping.
        Can also add a 0 at the first position to offset the input words by one space.
        """
        #leave a gap at the start or end of the seqeuence
        start_space = 0
        if start_pad:
            start_space = 1

        X_ret = []
        for x in X:
            x = np.array(x)
            nlen = min(max_len-1, x.shape[0])

            xn = np.zeros((max_len,), dtype=np.int32)
            if not reverse:
                xn[start_space:nlen+start_space] = x[:max_len-1] #normal
            else:
                xn[start_space:nlen+start_space] = x[max_len-2::-1] #reversed
            X_ret.append(xn)

        return np.array(X_ret, dtype=np.int32)

class TokenSequences(object):
    def __init__(self, tokenizer, reverse = False, start_pad = False, seq_len=20):
        self.seqs = None
        self.seqs_pad = None

        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.reverse = reverse
        self.start_pad = start_pad

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def get_word_count(self):
        return np.array(self.tokenizer.get_word_count())

    def set_pad_attributes(self, reverse, start_pad, seq_len):
        self.reverse = reverse
        self.start_pad = start_pad
        self.seq_len = seq_len

        self.seqs_pad = None

    def from_seqs(self, seqs, copy_seqs=True):
        if copy_seqs:
            self.seqs = copy.deepcopy(seqs)
        else:
            self.seqs = seqs

        self.seqs_pad = None

    def extend_seqs(self, seqs, copy_seqs=True):
        if copy_seqs:
            self.seqs.extend(copy.deepcopy(seqs))
        else:
            self.seqs.extend(seqs)

        self.seqs_pad = None


    def shuffle(self, order=None):
        if order is None:
            new_order = np.arange(len(self.seqs))
            np.random.shuffle(new_order)
        else:
            new_order = order
            assert len(order) == len(self.seqs)

        new_seqs = []
        for i in new_order:
            new_seqs.append(self.seqs[i])

        self.from_seqs(new_seqs, copy_seqs=False)

        return new_order

    def from_padded_seqs(self, padded_seqs, is_reversed, is_start_pad):
        seqs = SequencePadder.remove_padding(padded_seqs, reverse=is_reversed, start_pad=is_start_pad)
        self.from_seqs(seqs, copy_seqs=False)

    def from_text(self, text, text_normalizer_options = None):
        if text_normalizer_options is not None:
            tn = TextNormalizer()
            text_norm = tn.normalize_text(text)
        else:
            text_norm = text

        seqs = self.tokenizer.text_to_seqs(text_norm)
        self.from_seqs(seqs, copy_seqs=False)

    def get_num_seqs(self):
        return len(self.seqs)

    def get_seqs(self, start=None, end=None):
        if start is not None and end is not None:
            return self.seqs[start:end]
        else:
            return self.seqs

    def get_padded_seqs(self, start=None, end=None):

        # cache the padded sequeces 
        if self.seqs_pad is None:
            self.seqs_pad = SequencePadder.get_padded(self.seqs, self.seq_len, 
                    reverse=self.reverse, start_pad=self.start_pad)

        if start is not None and end is not None:
            return self.seqs_pad[start:end]
        else:
            return self.seqs_pad

    def get_text(self, start=None, end=None):
        chosen_seqs = self.get_seqs(start, end)
        text = self.tokenizer.seqs_to_text(chosen_seqs)
        return text

    def get_masks(self, start=None, end=None, dtype=np.int32):
        seqs_pad = self.get_padded_seqs(start, end)
        masks = np.zeros(seqs_pad.shape, dtype=dtype)
        # standard mask
        bool_msk = seqs_pad != 0
        # one past the end mask (so that we predict the <eos> token
        bool_msk[:, 1:] = np.logical_or(bool_msk[:, 1:], (seqs_pad[:, :-1] != 0))
        # fill first col because we might have start_pad (regardless first col should always be 1)
        masks[bool_msk] = 1
        masks[:, 0] = 1
        return masks

    def filter(self, chosen):
        seqs_pad = self.get_padded_seqs()
        seqs_pad = seqs_pad[chosen]
        self.from_padded_seqs(seqs_pad, is_reversed=self.reverse, is_start_pad=self.start_pad)
