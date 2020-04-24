import numpy as np
from pytorch_transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import torch.nn.functional as F

from general_beamsearch import LanguageModel, LMScore # for writing a general LM

class GPTLanguageModel(LanguageModel):
    def __init__(self, model_path='gpt2', tokenizer_path='gpt2'):
        self.device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
        self.model = GPT2LMHeadModel.from_pretrained(model_path) # default loads from huggingface AWS servers. can cache if behind a firewall
        self.model.eval() # set model to eval mode because we're not fine tuning
        self.model.to(self.device)
        print('loaded model')

        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
        print('loaded tokenizer')

        self.num_calls = 0

    def encode(self, word):
        return self.tokenizer.encode(word)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def score(self, context, words, past=None):
        '''
        Return log probabilities for extending context

        context - a list of GPT tokens that comprise the prior history
        words - a list of possible words.
        past - previous logits of GPT for that beam for efficiency. Currently not used, could be an efficiency gain

        return a list of LMScores in the same order as words
        '''

        if len(words) == 0:
            raise ValueError('No extensions proposed')
        # elif len(words) == 1:
        #   in theory, if there's only one possibility, we could skip the LM since the ranking won't change,
        #   but with the current implementation, we still need to run it because the score won't get accumulated otherwise,
        #   which could make a difference for comparing e.g. "saw" and "stared" in the sentence "I ___ it".
        #   This implementation could be improved by queuing up words with only one possibility and passing them to `words` at once

        print('Calling GPT with', context, words)

        # Can't run GPT on sentences with length 1. Prepend EOS
        if not context:
            context = [self.tokenizer.vocab_size - 1] + context
        # Limit the size of context because GPT can only handle 512 wordpieces
        context = context[-64:]

        tokenized_input = [self.encode(word) for word in words]

        # Pad tokenized_input with zeros to make it rectangular
        word_lengths = np.array([len(word) for word in tokenized_input]) # number of wordpieces in each candidate word
        np_input = np.zeros((len(word_lengths), max(word_lengths)), dtype=np.int32)
        for i, sent in enumerate(tokenized_input):
            np_input[i,:len(tokenized_input[i])] = tokenized_input[i]

        batch_size = 128 # batch up compute so we don't run out of GPU memory
        log_probs = np.zeros(np_input.shape[0]) # accumulate results in this

        with torch.no_grad():
            # Run context to get and cache past. This prevents runtime from growing as the context gets longer.
            # This could be saved from previous rounds, but that's probably unnecessary
            past_input = torch.tensor([context], device=self.device)
            past_logits, past = self.model(past_input)
            
            # Sort data by length
            order = np.argsort(-word_lengths)
                
            for batch_start in range(0, np_input.shape[0], batch_size):
                # print('Allocated memory inner loop', torch.cuda.memory_allocated(self.device), 'max allocated', torch.cuda.max_memory_allocated(self.device))
                batch_indices = order[batch_start:batch_start+batch_size]
                batch_lengths = word_lengths[batch_indices]
                batch_inputs = np_input[batch_indices,:max(batch_lengths)]
                torch_batch_lengths = torch.tensor(batch_lengths, dtype=torch.int32, device=self.device)
                
                true_batch_size = len(batch_lengths) # the actual size of the batch, accounting for the final batch being smaller

                # Evaluate on present state
                present_input = torch.tensor(batch_inputs, dtype=torch.long, device=self.device)

                # actually do GPT compute only if necessary. otherwise can just take it from final logits of previous step
                if max(batch_lengths) > 1:
                    # Run the model with each new possibility (batched). We do a lot of unnecessary computation here
                    # Every possibility is batched, even single word ones for which we know the answer
                    past_ = torch.stack(past)
                    past_ = past_.expand(-1,-1,true_batch_size,-1,-1,-1)
                    present_logits, _present = self.model(present_input, past=past_)
                    
                    # Calculate loss for each possible completion.
                    logits = torch.cat((past_logits[:, -1, :].expand(true_batch_size, 1, -1), present_logits[:, :-1]), dim=1).transpose(1,2)
                    targets = present_input

                    print(logits.shape, targets.shape)

                    loss_fcn = torch.nn.CrossEntropyLoss(reduction='none')
                    losses = loss_fcn(logits, targets)
                    print(losses)
                    for b in range(true_batch_size):
                        losses[b,batch_lengths[b]:] = 0
                    log_probs[batch_indices] = losses.sum(axis=1).detach().cpu().numpy()

                # skip GPT call for batches of words of length 1
                else:
                    logits = past_logits[:, -1, :].expand(true_batch_size, -1)
                    targets = present_input[:,0]

                    loss_fcn = torch.nn.CrossEntropyLoss()
                    losses = loss_fcn(logits, targets)
                    log_probs[batch_indices] = losses.detach().cpu().numpy()

        result = [LMScore(tokens=tokens, score=lm_prob) for tokens, lm_prob in zip(tokenized_input, -log_probs)]
        return result

def test():
    print('====== testing GPT lm ========')
    from gpt_lm import GPTLanguageModel
    from general_beamsearch import Lattice, Beam, beam_search

    l = Lattice()
    l.from_carmel_lattice('lattices/test2.lattice')
    gpt = GPTLanguageModel()
    Beam.tokenizer = gpt.tokenizer # for debugging
    beams = beam_search(gpt, l, beam_width=16)
    for b in beams:
        print(str(b))

    print('num gpt calls', gpt.num_calls)

if __name__ == '__main__':
    test()