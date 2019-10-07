from datetime import datetime

from tqdm import tqdm
import numpy as np
from scipy.stats import beta

from pytorch_transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import torch.nn.functional as F

from vocab import Vocab
from wordbank import Wordbank, Token
from passes import tokenize_ciphertext

class TokenLattice():
    def __init__(self, wordbank):
        self.wordbank = wordbank
        self.vocab = wordbank.vocab

    def possible_words(self, source_token):
        start, end, _mean = self.wordbank.get_anchors(source_token)
        return start, end
    
    def batch_prob(self, start, end, source_token):
        '''
        Return a list of pairs (word, lattice_probability)
        start and end index vocab.words

        The total amount is vocab.flat_index[end] - vocab.flat_index[start]
        '''

        # unknown words aren't decipherable
        if source_token.ciphertype is None:
            return [0]
        # no point in assigning probabilities to known words
        if not source_token.is_unk(self.wordbank):
            return [0] * len(self.vocab._inflected_inv_vocab[start]) # assign probabilities over possible inflections

        # Positions in a modern dictionary
        # Can scale things to (x - anchor_left) / scale to scale it from 0 to 1
        anchor_left, anchor_right, mean = self.wordbank.get_anchors(source_token)
        scale = anchor_right - anchor_left
        if scale == 0:
            raise ValueError('Found a token that can be deduced from wordbank. Did you forget to call wordbank.apply?')

        # This calculation is redundant, but cheap
        assert start == anchor_left
        assert end == anchor_right

        # Parameterize distribution
        b = 1 # parameterizes the sharpness of the distribution.
              # TODO: actually fit data to figure out what this should be
        a = (mean * b) / (1 - mean) # controls where the peak is

        # distribute probabilities to `scale` probability buckets, corresponding to tokens in vocab
        cdf = beta.cdf(np.arange(scale + 1) / scale, a=a, b=b)
        probability_buckets = np.log(cdf[1:] - cdf[:-1])

        # vocab._inflected_inv_vocab, probability_buckets

        result = []
        for raw_word, inflected_forms, prob in zip(self.vocab.words[start:end], self.vocab._inflected_inv_vocab[start:end], probability_buckets):
            productive_penalty = np.log(len(inflected_forms)) # distribute probability over all inflected forms
            # print(f'Distributing {prob} probability mass from word {raw_word} to {len(inflected_forms)} buckets ({prob - productive_penalty})')
            # print('Forms: ')
            for form in inflected_forms:
                # print(form, end=', ')
                result.append(prob - productive_penalty)
            # print('')

        return result

    def to_carmel_lattice(self, source, filename):
        with open(filename, 'w') as fh:
            print(len(source), file=fh)

            for i, token in enumerate(source):
                if not token.plaintext.strip():
                    continue  # skip blank tokens
                elif token.ciphertype is None:
                    words = [token.plaintext]
                    start = end = 0 # these don't matter
                else:
                    start, end = self.possible_words(token)
                    words = self.vocab.inflected_words[self.vocab.to_flat_idx(start):self.vocab.to_flat_idx(end)]

                for next_word, lattice_prob in zip(words, self.batch_prob(start, end, token)):
                    print('({} ({} "{}" {}))'.format(i, i + 1, next_word, np.exp(lattice_prob)), file=fh)

class GPTLanguageModel():
    def __init__(self, vocab):
        self.vocab = vocab

        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        Beam.tokenizer = self.tokenizer

        self.model.eval() # set model to eval mode because we're not fine tuning

        # pretokenize inflected vocab
        self.tokenized_inflected_words = [self.tokenizer.encode(word) for word in vocab.inflected_words]

    def batch_score(self, sentence, start, end):
        '''Score adding each word in vocab.words to the language model'''

        # Can't run GPT on sentences with length 1. Make up a super-generic history
        if len(sentence) <= 1:
            # sentence = self.tokenizer.encode(self.tokenizer.unk_token) * (2 - len(sentence)) + sentence
            sentence = self.tokenizer.encode('Yes.') + sentence

        if start == end:
            # Word is known, only try variations of it
            original_words = self.vocab._inflected_inv_vocab[start]
            tokenized_input = self.tokenized_inflected_words[start:start+len(original_words)]
        else:
            flat_start = self.vocab.to_flat_idx(start)
            flat_end = self.vocab.to_flat_idx(end)

            original_words = self.vocab.inflected_words[flat_start:flat_end] # Flattened array of inflected forms, corresponding one-to-one with tokenized_input;
            tokenized_input = self.tokenized_inflected_words[flat_start:flat_end] # flattened array of token IDs for GPT. not necessarily rectangular

        # Pad tokenized_input with zeros
        word_lengths = np.array([len(word) for word in tokenized_input]) # number of wordpieces in each word
        np_input = np.zeros((len(word_lengths), max(word_lengths)), dtype=np.int32)
        for i, sent in enumerate(tokenized_input):
            np_input[i,:len(tokenized_input[i])] = tokenized_input[i]
        tensor_input = torch.tensor(np_input, dtype=torch.long)

        sentence_lengths = word_lengths + len(sentence)

        if max(sentence_lengths) >= 512:
            start_overflow = max(sentence_lengths) - 512 # drop everything but the last 512 tokens because GPT can't handle that
            tensor_input = tensor_input[:,start_overflow:]
            sentence_lengths -= start_overflow

        # prepend past to each context. can't figure out how to cache past and expand dimensions properly
        past_tensor = torch.tensor([sentence])
        past_tensor = past_tensor.expand(tensor_input.shape[0], -1)
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
        with torch.no_grad():
            mask = ~(sentence_lengths <= np.arange(1,max(sentence_lengths))[:,None]).T
            np_losses = losses.detach().numpy()
            cross_entropies = ((mask) * np_losses).sum(axis=1)

        # for flat_i, (original_word, tokens, prob) in enumerate(zip(original_words, tokenized_input, cross_entropies)):
        #     print(f'Language model gives {prob} probability to word {original_word} (raw: {tokens}) with context {sentence}')

        return -cross_entropies, tokenized_input, original_words


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
        if Beam.tokenizer is None:
            return "{}:\n    {} + {} = {}".format(self.prediction, self.lm_prob, self.lattice_prob, self.log_prob)
        else:
            return "{}/{}:\n    {} + {} = {}".format(self.prediction, Beam.tokenizer.decode(self.prediction), self.lm_prob, self.lattice_prob, self.log_prob)


def token_beam_search(source, lm, lattice, beam_width=8):
    '''Return beams sorted best to worst'''
    beams = [Beam([])]

    for i in range(len(source)):
        print('Beam iteration {}/{} @ {}'.format(i, len(source), datetime.now().time()))

        # unknown word, isn't decipherable
        if source[i].ciphertype is None:
            if source[i].plaintext.isalnum():
                # add numbers and letters to the beam, but not tokens
                for beam in beams:
                    beam.prediction.extend(lm.tokenizer.encode(source[i].plaintext))
            elif source[i].plaintext.isspace():
                pass # skip whitespace
            else:
                # add unk to the beam
                for beam in beams:
                    beam.prediction.extend(lm.tokenizer.encode(lm.tokenizer.unk_token))

            # Doesn't affect the score
            continue

        new_beams = []
        start, end = lattice.possible_words(source[i])
        lattice_probs = lattice.batch_prob(start, end, source[i])

        import pdb
        pdb.set_trace()

        for beam in tqdm(beams):
            # expand beam
            lm_probs, token_lists, inflected_words = lm.batch_score(beam.prediction, start, end)

            for word, tokens, lattice_prob, lm_prob in zip(inflected_words, token_lists, lattice_probs, lm_probs):
                new_beam = Beam(beam.prediction + tokens, lm_prob, beam.lattice_prob + lattice_prob)
                new_beams.append(new_beam)
                # print(f'Proposing new word {word} with probability {new_beam.lm_prob} + {new_beam.lattice_prob} = {new_beam.log_prob}')

        beams = sorted(new_beams, key=lambda beam: beam.log_prob, reverse=True)
        print('Best Beams:')
        beams = beams[:beam_width]
        for beam in beams:
            print(str(beam))
    return beams

def test(lm, lattice):

    token1 = Token('[160]^')
    token2 = Token('[960]^')

    print('=== Possible words ===')
    print(lattice.possible_words(token1)) # known, a
    start, end = lattice.possible_words(token2)
    print(start, end)

    print('=== Lattice score ===')
    lattice_score = lattice.batch_prob(start, end, token2)
    print(lattice_score)

    print('=== LM score ===')
    lm_score = lm.batch_score([123, 456], start, end)
    print(lm_score)

if __name__ == '__main__':
    v = Vocab('wordbanks/vocab.toy')
    wb = Wordbank(v)
    wb.load('wordbanks/wordbank.toy')

    lm = GPTLanguageModel(v)
    lattice = TokenLattice(wb)

    # test(lm, lattice)
    # quit()

    # lm.batch_score(lm.tokenizer.encode('A cat sees the'), 0,6)
    # quit()

    # ct = tokenize_ciphertext('[1078]^ [330]^ [960]^ [160]^ [490]^') # given the toy vocab, should give "the cat sees the dog"
    # ct = tokenize_ciphertext('the [330]^ [960]^ [160]^ [490]^\n with the')

    ct = tokenize_ciphertext('''[556]^ Ferdinand 2 [678]^ 95 [1235]^ [433]^ [79]^ [664]^ 
  [1218]^  [807]^ [313]^ [1078]^ [804]^ ''')
    ct = [wb.apply(tok) for tok in ct]


    lattice.to_carmel_lattice(ct, 'output/lattice.toy')

    beams = token_beam_search(ct, lm, lattice, beam_width=64)
    print('\n\n================ DONE ===============\n\n\n')
    for beam in beams:
        print(str(beam))
