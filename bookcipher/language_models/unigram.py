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
        self.unk = -1

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

class SlowGPTLanguageModel(LanguageModel):
    def __init__(self, vocab):
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        self.model.eval() # set model to eval mode because we're not fine tuning

        self.vocab = vocab
        self.vocab_indices = {word: i for i, word in enumerate(self.vocab.words)}

        self.unk = -1

        self.score_calls = 0

        # pretokenize each word in vocab
        self.inflected_vocab = [self.vocab.inflect(word) for word in vocab]
        self.tokenized_inflected_vocab = map(self.tokenizer.encode, itertools.chain(*self.inflected_vocab)) # list of ints
        print('TIV', self.tokenized_inflected_vocab[:10])

        # mapping vocab -> tokenized_inflected_vocab
        self.vocab_to_inflected_index = list(itertools.accumulate(map(len, self.inflected_vocab)))
        self.vocab_to_inflected_index.insert(0, 0)
        self.vocab_to_inflected_index.pop()

    def score(self, sentence):
        self.score_calls += 1
        if self.score_calls % 1000 == 0:
            print('calls to score', self.score_calls)

        with torch.no_grad():
            tokenized_input = self.tokenizer.tokenize(sentence)
            if len(tokenized_input) <= 1:
                tokenized_input = [220] * (2 - len(tokenized_input)) + tokenized_input # length cannot be <= 1, so prepend ' ' to it
            tensor_input = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokenized_input)])
            loss=self.model(tensor_input, labels=tensor_input)
            return -loss[0]

    def next_token_batch(self, sentence, start, end):
        '''
        score inflected forms from start to end. start and end are indices in vocab

        return zip(probabilites, inflected words)
        '''

        # Use cached vocab
        inflected_vocab = self.inflected_vocab[self.vocab_to_inflected_index[start]:self.vocab_to_inflected_index[end]]
        tokenized_input = self.tokenized_inflected_vocab[self.vocab_to_inflected_index[start]:self.vocab_to_inflected_index[end]]
        sentence_length = [len(arr) for arr in tokenized_input]

        # Can't run GPT on sentences with length 1
        if len(sentence) <= 1:
            return zip(inflected_vocab, itertools.repeat(0))

        # Run the model once to get the history
        old_logits, past = model(sentence)
        
        # pad sentences with zeros
        np_input = np.zeros((len(sentence_length), max(sentence_length)), dtype=np.int32)
        for i, sent in enumerate(tokenized_input):
            np_input[i,:len(tokenized_input[i])] = tokenized_input[i]
        tensor_input = torch.tensor(np_input, dtype=torch.long)


        # Run the model with the new stuff
        model(tokenized_input, past=past)

        loss_fcn = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
        losses = loss_fcn(
            torch.cat(torch.unsqueeze(logits[:,-1,:], 1), logits[:,:-1,:]).transpose(1,2), # concat old loss to new loss
            tensor_input
        )

        # compute mean loss
        with torch.no_grad():
            mask = (sentence_length < np.arange(max(sentence_length) - 1)[:,None]).T
            np_losses = losses.detach().numpy()
            cross_entropies = (((~mask) * np_losses).sum(axis=1) / sentence_length)
        return zip(tokenized_input, cross_entropies)

    def next_token(self, s):
        '''Get the probabilities of the next wordpiece'''

        result = [self.score(' '.join(s) + ' ' + word) for word in self.vocab.words]
        return result


if __name__ == '__main__':
    v = Vocab('wordbanks/vocab.toy')
    lm = SlowGPTLanguageModel(v)
    lm.next_token_batch([1,2,3])
