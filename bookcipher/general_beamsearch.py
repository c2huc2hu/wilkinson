from collections import namedtuple, defaultdict, Counter
from datetime import datetime
import math
import re
import os
import copy

import numpy as np
from tqdm import tqdm, trange

class LatticeEdge():
    def __init__(self, label, log_prob):
        self.label = label
        self.log_prob = log_prob # should be a log prob

class Lattice():
    def __init__(self, lattice=None, start_state=None, final_state=None):
        '''
        Construct a lattice

        Lattice is a dict of dict of list of LatticeEdges
        lattice[from_state][to_state] = (label='word', log_prob=0.123)
            {from_state: {to_state: [(label=lattice_label, log_prob=probability), ...]}, {...}, ...}
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
        '''Return log probabilities for each element in i. Batching computation lets you vectorize things and apply smoothing if necessary'''
        return self.lattice[from_state].items() # iterable of (target_state, (label, probability))

    def from_carmel_lattice(self, filename):
        '''Alternate constructor from carmel file'''
        with open(filename) as f:
            self.final_state = next(f).strip()

            self.lattice = defaultdict(lambda: defaultdict(list))
            for line in f:
                if line.strip():
                    m = re.match(r'\((\w+)\s*\((\w+)\s+([\w\'" .-]+)\s+([\d.e-]+)\)\)', line)

                    if not m:
                        print('couldnt parse line', line)
                    else:
                        from_state, to_state, label, prob = m.groups()
                        label = label.strip('\'"') # strip quotes
                        log_prob = np.log(float(prob)) # turn into log probs

                        self.lattice[from_state][to_state].append(LatticeEdge(label, log_prob))

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
                    for edge in self.possible_edges(from_state, to_state):
                        print('({} ({} "{}" {}))'.format(from_state, to_state, edge.label, math.exp(edge.log_prob)), file=fh)

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
        return [LMScore(tokens=[word], score=log_prob) for word, log_prob in zip(words, probabilities)]

class Beam():
    tokenizer = None

    def __init__(self, prediction='', lm_prob=0, lattice_prob=0, prev=None, lattice_edge=None):
        self.prediction = prediction
        self.lm_prob = lm_prob
        self.lattice_prob = lattice_prob
        self.log_prob = lm_prob + lattice_prob
        self.prev = prev
        self.lattice_edge = lattice_edge

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
    state = lattice.start_state
    beams = [Beam([])]
    history = []

    # set up decoder for nicely debugging
    try:
        Beam.decode = lm.decode
    except AttributeError:
        print('LM Decoder not defined, raw tokens will be printed')


    for i in trange(lattice.n_states):
        print('Beam iteration {}/{} @ time {}, label {}'.format(i, lattice.n_states, datetime.now().time(), state))

        # print('Allocated memory', torch.cuda.memory_allocated(lm.device), 'max allocated', torch.cuda.max_memory_allocated(lm.device))
        # import gc
        # gc.collect() # torch leaks memory used in the tensors, so free it manually
        # allocated_count = 0
        # for obj in gc.get_objects():
        #     try:
        #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
        #             allocated_count += 1
        #     except:
        #         pass
        # print('Allocated', allocated_count)

        new_beams = []
        to_states = lattice.possible_to_states(state)
        if len(to_states) != 1:
            raise NotImplemented('Beam search only supports sausage lattices')
        else:
            next_state = list(to_states)[0]

        lattice_edges = lattice.possible_edges(state, next_state)

        # error out if there are no possibilites
        if len(lattice_edges) == 0:
            raise IndexError('State {} has no successors'.format(i))
        else:
            for beam in tqdm(beams):
                lm_scores = lm.score(beam.prediction, [edge.label for edge in lattice_edges])

                if len(lattice_edges) != len(lm_scores):
                    raise IndexError('Number of lm probabilities ({}) doesn\'t match number of labels ({})'.format(len(lm_scores), len(lattice_edges)))

                for (lattice_edge, (lm_tokens, lm_score)) in zip(lattice_edges, lm_scores):
                    new_beam = Beam(beam.prediction + lm_tokens, beam.lm_prob + lm_score, beam.lattice_prob + lattice_edge.log_prob, beam, lattice_edge)
                    new_beams.append(new_beam)
                    # print(f'Proposing new word {word} with probability {new_beam.lm_prob} + {new_beam.lattice_prob} = {new_beam.log_prob}')

        state = next_state

        # Dedupe beams. Can get duplicates if two words in the dictionary are inflected the same way
        new_beams = {tuple(beam.prediction): beam for beam in new_beams}.values()
        history.append(new_beams)
        print('History', len(history))

        # Sort beams by probability
        beams = sorted(new_beams, key=lambda beam: beam.log_prob, reverse=True)
        beams = beams[:beam_width]

        # for debugging long running things
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
    
if __name__ == '__main__':
    test()