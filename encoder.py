from torch import nn
from transformers import DistilBertModel, DistilBertTokenizerFast


class RobertaSentenceEncoder(nn.Module):
    def __init__(self, max_length,):
        nn.Module.__init__(self)
        self.max_length = max_length
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')

    def encode(self, sample):
        tokens = self.tokenize(sample['tokens'])
        pos1 = sample['h'][2]
        pos2 = sample['t'][2]
        return tokens, pos1, pos2
