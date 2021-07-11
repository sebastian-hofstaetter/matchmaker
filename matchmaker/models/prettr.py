from transformers import *
from transformers.models.distilbert.modeling_distilbert import *
import math
import torch
from torch import nn as nn

class PreTTRConfig(DistilBertConfig):
    join_layer_idx = 3

class PreTTR(DistilBertModel):
    '''
    PreTTR changes the distilbert model from huggingface to be able to split query and document until a set layer,
    we skipped compression present in the original

    from: Efficient Document Re-Ranking for Transformers by Precomputing Term Representations
          MacAvaney, et al. https://arxiv.org/abs/2004.14255
    '''
    config_class = PreTTRConfig

    def __init__(self, config):
        super().__init__(config)
        self.transformer = SplitTransformer(config)  # Encoder, we override the classes, but the names stay the same -> so it gets properly initialized
        self.embeddings = PosOffsetEmbeddings(config)  # Embeddings
        self._classification_layer = torch.nn.Linear(self.config.hidden_size, 1, bias=False)

        self.join_layer_idx = config.join_layer_idx

    def forward(
            self,
            query,
            document,
            use_fp16: bool = True,
            output_secondary_output: bool = False) -> torch.Tensor:

        with torch.cuda.amp.autocast(enabled=use_fp16):

            query_input_ids = query["input_ids"]
            query_attention_mask = query["attention_mask"]

            document_input_ids = document["input_ids"][:, 1:]
            document_attention_mask = document["attention_mask"][:, 1:]

            query_embs = self.embeddings(query_input_ids)  # (bs, seq_length, dim)
            document_embs = self.embeddings(document_input_ids, query_input_ids.shape[-1])  # (bs, seq_length, dim)

            tfmr_output = self.transformer(
                query_embs=query_embs,
                query_mask=query_attention_mask,
                doc_embs=document_embs,
                doc_mask=document_attention_mask,
                join_layer_idx=self.join_layer_idx
            )
            hidden_state = tfmr_output[0]

            score = self._classification_layer(hidden_state[:, 0, :]).squeeze()

            if output_secondary_output:
                return score, {}
            return score

    def get_param_stats(self):
        return "PreTTR: / "

    def get_param_secondary(self):
        return {}


class PosOffsetEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.dim, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.dim)
        if config.sinusoidal_pos_embds:
            create_sinusoidal_embeddings(
                n_pos=config.max_position_embeddings, dim=config.dim, out=self.position_embeddings.weight
            )

        self.LayerNorm = nn.LayerNorm(config.dim, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids, pos_offset=0):
        """
        Parameters
        ----------
        input_ids: torch.tensor(bs, max_seq_length)
            The token ids to embed.

        Outputs
        -------
        embeddings: torch.tensor(bs, max_seq_length, dim)
            The embedded tokens (plus position embeddings, no token_type embeddings)
        """
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)  # (max_seq_length)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids) + pos_offset  # (bs, max_seq_length)

        word_embeddings = self.word_embeddings(input_ids)  # (bs, max_seq_length, dim)
        position_embeddings = self.position_embeddings(position_ids)  # (bs, max_seq_length, dim)

        embeddings = word_embeddings + position_embeddings  # (bs, max_seq_length, dim)
        embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
        embeddings = self.dropout(embeddings)  # (bs, max_seq_length, dim)
        return embeddings


class SplitTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_layers = config.n_layers

        layer = TransformerBlock(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.n_layers)])

    def forward(self, query_embs, query_mask, doc_embs, doc_mask, join_layer_idx, output_attentions=False, output_hidden_states=False):
        """
        Parameters
        ----------
        x: torch.tensor(bs, seq_length, dim)
            Input sequence embedded.
        attn_mask: torch.tensor(bs, seq_length)
            Attention mask on the sequence.

        Outputs
        -------
        hidden_state: torch.tensor(bs, seq_length, dim)
            Sequence of hiddens states in the last (top) layer
        all_hidden_states: Tuple[torch.tensor(bs, seq_length, dim)]
            Tuple of length n_layers with the hidden states from each layer.
            Optional: only if output_hidden_states=True
        all_attentions: Tuple[torch.tensor(bs, n_heads, seq_length, seq_length)]
            Tuple of length n_layers with the attention weights from each layer
            Optional: only if output_attentions=True
        """
        all_hidden_states = ()
        all_attentions = ()

        #
        # query / doc sep.
        #
        hidden_state_q = query_embs
        hidden_state_d = doc_embs
        for layer_module in self.layer[:join_layer_idx]:

            layer_outputs_q = layer_module(
                x=hidden_state_q, attn_mask=query_mask, head_mask=None, output_attentions=output_attentions
            )
            hidden_state_q = layer_outputs_q[-1]

            layer_outputs_d = layer_module(
                x=hidden_state_d, attn_mask=doc_mask, head_mask=None, output_attentions=output_attentions
            )
            hidden_state_d = layer_outputs_d[-1]

        #
        # combine
        #
        x = torch.cat([hidden_state_q, hidden_state_d], dim=1)
        attn_mask = torch.cat([query_mask, doc_mask], dim=1)

        #
        # combined
        #
        hidden_state = x
        for layer_module in self.layer[join_layer_idx:]:
            layer_outputs = layer_module(
                x=hidden_state, attn_mask=attn_mask, head_mask=None, output_attentions=output_attentions
            )
            hidden_state = layer_outputs[-1]

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)

        outputs = (hidden_state,)
        if output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)
