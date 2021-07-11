
import torch
from torch import nn

class BertNoPosEmbeddings(nn.Module):
    """wrapper to remove postion embeddings from bert
    """
    def __init__(self, word_embeddings,token_type_embeddings,layerNorm,dropout):
        super(BertNoPosEmbeddings, self).__init__()
        self.word_embeddings = word_embeddings
        self.token_type_embeddings = token_type_embeddings

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = layerNorm
        self.dropout = dropout

    def forward(self, input_ids, token_type_ids=None):
        #seq_length = input_ids.size(1)
        #position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        #position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        #if token_type_ids is None:
        #    token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        #position_embeddings = self.position_embeddings(position_ids)
        #token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings #+ token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertEncoderReduced(nn.Module):
    def __init__(self, layers):
        super(BertEncoderReduced, self).__init__()
        self.layer = layers

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers
