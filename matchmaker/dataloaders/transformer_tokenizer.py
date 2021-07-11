from allennlp.data.fields import TransformerTextField
from transformers import PreTrainedTokenizerFast
from transformers import AutoTokenizer


class PatchedTransformerTextField(TransformerTextField):
    def __len__(self):
        return self.input_ids.shape[-1]


class FastTransformerTokenizer():
    """
    basic wrapper for an HuggingFace AutoTokenizer
    """

    def __init__(self, model):

        self._tokenizer = AutoTokenizer.from_pretrained(model)

    def tokenize(self, sentence: str, sentence2: str = None, max_length: int = 512):
        if sentence2 != None:
            seq_tokenized = self._tokenizer(sentence, sentence2,
                                            max_length=max_length,
                                            truncation=True,
                                            return_tensors="pt",
                                            return_attention_mask=True)

        else:
            seq_tokenized = self._tokenizer(sentence,
                                            max_length=max_length,
                                            truncation=True,
                                            return_tensors="pt",
                                            return_attention_mask=True)

        for k, d in seq_tokenized.data.items():
            d.squeeze_(0)
        return seq_tokenized.data
