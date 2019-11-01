import numpy as np
from pytorch_transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import torch.nn.functional as F

from general_beamsearch import LanguageModel, LMScore # for writing a general LM
from general_beamsearch import Lattice, Beam, beam_search # for running test cases

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
            context = self.encode('Yes.') + context
        context = context[-64:] # limit the size of context because GPT can only handle 512

        tokenized_input = [self.encode(word) for word in words]

        # Pad tokenized_input with zeros to make it rectangular
        word_lengths = np.array([len(word) for word in tokenized_input]) # number of wordpieces in each candidate word
        np_input = np.zeros((len(word_lengths), max(word_lengths)), dtype=np.int32)
        for i, sent in enumerate(tokenized_input):
            np_input[i,:len(tokenized_input[i])] = tokenized_input[i]

        batch_size = 128 # batch up compute so we don't run out of GPU memory
        cross_entropies = np.zeros(np_input.shape[0]) # accumulate results in this

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
                
                true_batch_size = len(batch_lengths) # the actual size of the batch, accounting for the final batch being smaller

                # actually do GPT compute only if necessary. otherwise can just take it from final logits of previous step
                if max(batch_lengths) > 1:

                    # Evaluate on present state
                    present_input = torch.tensor(batch_inputs, dtype=torch.long, device=self.device)

                    # Run the model with each new possibility (batched). We do a lot of unnecessary computation here
                    # Every possibility is batched, even single word ones for which we know the answer
                    past_ = torch.stack(past)
                    past_ = past_.expand(-1,-1,true_batch_size,-1,-1,-1)
                    present_logits, _present = self.model(present_input, past=past_)
                    
                    # Calculate loss for each possible completion. Can't use huggingface's summary because it summarizes over words, not sentences
                    # Could cache the loss for the past, but I don't think that's the bottleneck
                    logits = torch.cat((past_logits.expand(true_batch_size,-1,-1), present_logits), dim=1).transpose(1,2)[:,:,:-1]
                    targets = torch.cat((past_input.expand(true_batch_size,-1), present_input), dim=1)[:,1:]

                # skip GPT call for words of length 1
                else:
                    logits = past_logits.expand(true_batch_size,-1,-1).transpose(1,2)
                    present_input = torch.tensor(batch_inputs, dtype=torch.long, device=self.device)
                    targets = torch.cat((past_input.expand(true_batch_size,-1), present_input), dim=1)[:,1:]

                loss_fcn = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='none')        
                losses = loss_fcn(
                    logits,
                    targets
                )

                # compute mean loss
                # possible optimization: move this onto GPU? 
                # torch doesn't support newaxis broadcasting, and it requires an extra copy, so probably no impact
                mask = ~(len(context) + batch_lengths <= np.arange(1,len(context) + max(batch_lengths))[:,None]).T
                np_losses = losses.detach().cpu().numpy()
                cross_entropies[batch_indices] = (mask * np_losses).sum(axis=1)

        return [LMScore(tokens=tokens, score=lm_prob) for tokens, lm_prob in zip(tokenized_input, -cross_entropies)]

def test():
    print('====== testing GPT lm ========')
    from gpt_lm import GPTLanguageModel
    l = Lattice()
    l.from_carmel_lattice('lattices/test.lattice')
    gpt = GPTLanguageModel()
    beams = beam_search(gpt, l, beam_width=2)
    for b in beams:
        print(str(b))


if __name__ == '__main__':
    test()