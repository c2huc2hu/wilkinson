import nltk
from collections import Counter
from language_models.beam_search import LanguageModel
from nltk.lm import Vocabulary, Lidstone
from nltk.corpus import gutenberg
from nltk.util import ngrams

class UnigramLanguageModel(LanguageModel):
    '''Unigram probabilites from the Gutenberg corpus from NLTK'''

    def __init__(self, vocab):
        self.vocab = vocab.words
        self.vocab_indices = vocab._vocab

        nltk.download('gutenberg')
        counter = Counter(nltk.corpus.gutenberg.words())

        self.unigram_probs = [counter[word] for word in self.vocab]

    def initialize(self, source):
        pass

    def next_token(self, s):
        '''
        return unigram probabilites for the last token
        '''
        return self.unigram_probs

class BigramLanguageModel(LanguageModel):
    '''
    Bigram probabilities from the Gutenberg corpus from NLTK
    This is so slow it's unusable
    '''

    def __init__(self, vocab):
        N = 2
        sents = gutenberg.sents()

        train_data = [ngrams(sent, N) for sent in sents]
        self.vocab = [word for sent in sents for word in sent]
        self.vocab_indices = {self.vocab[i]: i for i in range(len(self.vocab))}

        v = Vocabulary(self.vocab)

        self.lm = Lidstone(0.01, N, v) # add 0.01 to counts
        self.lm.fit(train_data)

    def next_token(self, s):
        '''This is super slow, but works'''
        return [self.lm.score(word, s[-2:]) for word in self.vocab]



from pytorch_transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import torch.nn.functional as F

class GPTLanguageModel(LanguageModel):
    def __init__(self):
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        self.model.eval() # set model to eval mode because we're not fine tuning

        self.vocab = tokenizer.encode

        self.unk = self.tokenizer.unk_token

        self.score_calls = 0

    def score(sentence):
        self.score_calls += 1
        if self.score_calls % 100000 == 0:
            print('calls to score', self.score_calls)

        with torch.no_grad():
            tokenized_input = self.tokenizer.tokenize(sentence)
            if len(tokenized_input) <= 1:
                tokenized_input = [220] * (2 - len(tokenized_input)) + tokenized_input # length cannot be <= 1, so prepend ' ' to it
            tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenized_input)])
            loss=model(tensor_input, lm_labels=tensor_input)
            return np.exp(loss.item())

    def next_token(self, s):
        '''Get the probabilities of the next wordpiece'''
        input_ids = self.tokenizer.encode(' '.join(s)) # todo: strip out newlines
        input_tensor = torch.tensor(input_ids).unsqueeze(0)
        out_logits, past = self.model(input_tensor)
        out_tensor = out_logits[0][0,-1,:] # get the logits from the last word
        out_probs = F.softmax(out_tensor, dim=-1)

        return out_probs


