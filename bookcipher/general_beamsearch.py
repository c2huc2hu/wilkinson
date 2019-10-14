from collections import namedtuple, defaultdict, Counter
from datetime import datetime
import math
import re
import os

import numpy as np
from tqdm import tqdm, trange

LatticeEdge = namedtuple('LatticeEdge', ['label', 'prob']) # prob should be log probability

class Lattice():
    def __init__(self, lattice=None, start_state=None, final_state=None):
        '''
        Construct a lattice

        Lattice is a dict of dict of list of LatticeEdges
        lattice[from_state][to_state] = (label='word', prob=0.123)
            {from_state: {to_state: [(label=lattice_label, prob=probability), ...]}, {...}, ...}
        '''

        self.lattice = lattice
        self.start_state = start_state
        self.final_state = final_state

    def possible_to_states(self, from_state):
        return self.lattice[from_state].keys()

    def possible_edges(self, from_state, to_state):
        '''
        Return edge labels and probs (named tuple) to go from `from_state` to `to_state`
        '''
        return self.lattice[from_state][to_state]

    def prob(self, from_state):
        '''Return probabilities for each element in i. Batching computation lets you vectorize things and apply smoothing if necessary'''
        return self.lattice[from_state].items() # iterable of (target_state, (label, probability))

    def from_carmel_lattice(self, filename):
        '''Alternate constructor from carmel file'''
        with open(filename) as f:
            self.final_state = next(f).strip()

            self.lattice = defaultdict(lambda: defaultdict(list))
            for line in f:
                if line.strip():
                    m = re.match(r'\((\w+)\s*\((\w+)\s+([\w\'"]+)\s+([\d.]+)\)\)', line)

                    if not m:
                        print('couldnt parse line', line)
                    else:
                        from_state, to_state, label, prob = m.groups()
                        label = label.strip('\'"') # strip quotes
                        prob = np.log(float(prob)) # turn into log probs

                        self.lattice[from_state][to_state].append(LatticeEdge(label, prob))

                        # start state is the first state by default
                        if self.start_state is None:
                            self.start_state = from_state

    def to_carmel_lattice(self, filename):
        dirname = os.path.dirname(filename)
        os.makedirs(dirname, exist_ok=True)
        with open(filename, 'w') as fh:
            print(self.final_state, file=fh)

            for from_state in sorted(self.lattice, key=lambda from_state: from_state != self.start_state): # put the start state first
                for to_state in self.possible_to_states(from_state):
                    for (label, prob) in self.possible_edges(from_state, to_state):
                        print('({} ({} "{}" {}))'.format(from_state, to_state, label, math.exp(prob)), file=fh)

    @property
    def n_states(self):
        return len(self.lattice)

LMScore = namedtuple('LMScore', ['tokens', 'score']) # score should be log probability

class LanguageModel():
    def __init__(self):
        pass
    def encode(self, word):
        '''Tokenize and encode a word. Should return a list'''
        raise NotImplemented
    def decode(self, tokens):
        '''Decode a series of tokens. Return a string'''
        raise NotImplemented
    def score(self, context, words):
        '''Take a list of tokens as context and list of words. Return a list of LMScore for each word'''
        raise NotImplemented

class LengthLanguageModel(LanguageModel):
    '''Super simple, cheap language model for testing. Penalizes longer words'''
    def encode(self, word):
        return [word]
    def decode(self, tokens):
        return ' '.join(tokens)
    def score(self, context, words):
        return [LMScore(tokens=[word], score=-len(word)) for word in words]

class UnigramLanguageModel(LanguageModel):
    '''Unigram probabilites from the Gutenberg corpus from NLTK'''

    def __init__(self):
        import nltk
        nltk.download('gutenberg')
        self.counter = Counter(nltk.corpus.gutenberg.words())

    def encode(self, word):
        return [word]

    def decode(self, tokens):
        return ' '.join(tokens)

    def score(self, context, words):
        probabilities = np.array([-self.counter[word] for word in words])
        probabilities = np.log(probabilities /probabilities.sum())
        return [LMScore(tokens=[word], score=prob) for word, prob in zip(words, probabilities)]

from pytorch_transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import torch.nn.functional as F

class GPTLanguageModel(LanguageModel):
    def __init__(self, model_path='gpt2', tokenizer_path='gpt2'):
        self.device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
        self.model = GPT2LMHeadModel.from_pretrained(model_path) # default loads from huggingface AWS servers. can cache if behind a firewall
        self.model.eval() # set model to eval mode because we're not fine tuning
        self.model.to(self.device)
        print('loaded model')

        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
        Beam.tokenizer = self.tokenizer
        print('loaded tokenizer')
    def encode(self, word):
        return self.tokenizer.encode(word)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def score(self, context, words):
        '''context should be a list of GPT tokens in history. score `words` and return scores in the same order'''

        if len(words) == 0:
            raise ValueError('No extensions proposed')
        elif len(words) == 1:
            return 

        # Can't run GPT on sentences with length 1. Make up a super-generic history
        if len(context) <= 1:
            # context = self.tokenizer.encode(self.tokenizer.unk_token) * (2 - len(context)) + context
            context = self.tokenizer.encode('Yes.') + context

        tokenized_input = [self.encode(word) for word in words]

        # Pad tokenized_input with zeros
        word_lengths = np.array([len(word) for word in tokenized_input]) # number of wordpieces in each word
        np_input = np.zeros((len(word_lengths), max(word_lengths)), dtype=np.int32)
        for i, sent in enumerate(tokenized_input):
            np_input[i,:len(tokenized_input[i])] = tokenized_input[i]

        sentence_lengths = word_lengths + len(context)

        if max(sentence_lengths) >= 512:
            start_overflow = max(sentence_lengths) - 512 # drop everything but the last 512 tokens because GPT can't handle that
            np_input = np_input[:,start_overflow:]
            sentence_lengths -= start_overflow

        batch_size = 128 # batch up compute so we don't run out of GPU memory
        cross_entropies = np.zeros(np_input.shape[0])

        for batch_start in range(0, np_input.shape[0], batch_size):
            batch_lengths = sentence_lengths[batch_start:batch_start+batch_size]
            batch_inputs = np_input[batch_start:batch_start+batch_size]

            with torch.no_grad():
                # prepend past to each context. can't figure out how to cache past and expand dimensions properly
                tensor_input = torch.tensor(batch_inputs, dtype=torch.long, device=self.device)
                past_tensor = torch.tensor([context], device=self.device)
                past_tensor = past_tensor.expand(batch_lengths.shape[0], -1)
                tensor_input = torch.cat((past_tensor, tensor_input), dim=1)

                # Run the model with each new possibility (batched)
                logits, present = self.model(tensor_input)

                # Manually calculate loss for each thing
                loss_fcn = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
                losses = loss_fcn(
                    logits[:,:-1,:].transpose(1,2),
                    tensor_input[:,1:]
                )

                # compute mean loss
                # TODO: possible optimization: move this onto GPU. torch doesn't support newaxis broadcasting, so not sure if it would help
                mask = ~(batch_lengths <= np.arange(1,max(sentence_lengths))[:,None]).T
                np_losses = losses.detach().cpu().numpy()
                cross_entropies[batch_start:batch_start+batch_size] = ((mask) * np_losses).sum(axis=1)

                # for flat_i, (original_word, tokens, prob) in enumerate(zip(original_words, tokenized_input, cross_entropies)):
                #     print(f'Language model gives {prob} probability to word {original_word} (raw: {tokens}) with context {sentence}')

        return [LMScore(tokens=tokens, score=lm_prob) for tokens, lm_prob in zip(tokenized_input, -cross_entropies)]

class Beam():
    tokenizer = None

    def __init__(self, prediction='', lm_prob=0, lattice_prob=0):
        self.prediction = prediction
        self.lm_prob = lm_prob
        self.lattice_prob = lattice_prob
        self.log_prob = lm_prob + lattice_prob

    def __lt__(self, other):
        return self.log_prob < other.log_prob

    def __str__(self):
        if Beam.decode is None:
            return "{}:\n    {} + {} = {}".format(self.prediction, self.lm_prob, self.lattice_prob, self.log_prob)
        else:
            return "{}/{}:\n    {} + {} = {}".format(self.prediction, Beam.decode(self.prediction), self.lm_prob, self.lattice_prob, self.log_prob)


def beam_search(lm, lattice, beam_width=8):
    '''
    Do a beam search on a sausage lattice. LM should be an instance of LanguageModel, Lattice should be an instance of Lattice
    
    Return beams sorted best to worst
    '''
    print('Starting beam search')
    beams = [Beam([])]

    # set up decoder for nicely debugging
    try:
        Beam.decode = lm.decode
    except AttributeError:
        pass

    state = lattice.start_state

    for i in trange(lattice.n_states):
        print('Beam iteration {}/{} @ time {}, label {}'.format(i, lattice.n_states, datetime.now().time(), state))

        new_beams = []
        to_states = lattice.possible_to_states(state)
        if len(to_states) != 1:
            raise NotImplemented('Beam search only supports sausage lattices')
        else:
            next_state = list(to_states)[0]

        lattice_edges = lattice.possible_edges(state, next_state)

        # if there's only one possibility, don't run the language model on it
        if len(lattice_edges) == 1:
            label, _prob = lattice_edges[0]
            for beam in beams:
                beam.prediction.extend(lm.encode(label))
            state = next_state
            continue

        # error out if there are no possibilites
        elif len(lattice_edges) == 0:
            raise IndexError('State {} has no successors'.format(i))

        else:
            for beam in tqdm(beams):
                lm_scores = lm.score(beam.prediction, [edge.label for edge in lattice_edges])

                if len(lattice_edges) != len(lm_scores):
                    raise IndexError('Number of lm probabilities ({}) doesn\'t match number of labels ({})'.format(len(lm_scores), len(lattice_edges)))

                for ((lattice_edge, lattice_prob), (lm_tokens, lm_score)) in zip(lattice_edges, lm_scores):
                    new_beam = Beam(beam.prediction + lm_tokens, lm_score, beam.lattice_prob + lattice_prob)
                    # print(f'Proposing new word {word} with probability {new_beam.lm_prob} + {new_beam.lattice_prob} = {new_beam.log_prob}')
                    new_beams.append(new_beam)

        state = next_state

        beams = sorted(new_beams, key=lambda beam: beam.log_prob, reverse=True)
        beams = beams[:beam_width]

        # # for debugging long running things
        print('Best Beams:')
        for beam in beams:
            print(str(beam))
    # print('=========================')
    return beams

def test():
    # Simplest lattice
    print('====== super simple test ========')
    l = Lattice({0: {1: [LatticeEdge('the', 0)]}}, 0, 1)
    lm = LengthLanguageModel()
    l.to_carmel_lattice('lattices/simple.lattice')

    beams = beam_search(lm, l)
    for b in beams:
        print(str(b))

    print('====== testing lattice from file ========')
    l2 = Lattice()
    l2.from_carmel_lattice('lattices/test.lattice')
    beams = beam_search(lm, l2)
    for b in beams:
        print(str(b))

    print('====== testing GPT lm ========')
    l2 = Lattice()
    l2.from_carmel_lattice('lattices/test.lattice')
    gpt = GPTLanguageModel()
    beams = beam_search(gpt, l2, beam_width=2)
    for b in beams:
        print(str(b))


def main():
    pass

if __name__ == '__main__':
    test()

    lattice = Lattice()
    lm = LengthLanguageModel()


    
