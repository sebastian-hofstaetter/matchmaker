from typing import Dict, Union
import torch

from transformers import AutoModel
from transformers import PreTrainedModel,PretrainedConfig

class BERT_Dot_Config(PretrainedConfig):
    model_type = "BERT_Dot"
    bert_model: str
    trainable: bool = True
    compress_dim: int = -1 # if -1 add no compression layer, otherwise add 1 single linear layer (from bert_out_dim to compress_dim)
    return_vecs: bool = False # whether to return the vectors in the training forward pass (for in-batch negative loss)

class BERT_Dot(PreTrainedModel):    
    """
    The main dense retrieval model;
    this model does not concat query and document, rather it encodes them sep. and uses a dot-product between the two cls vectors
    """

    config_class = BERT_Dot_Config
    base_model_prefix = "bert_model"

    @staticmethod
    def from_config(config):
        cfg = BERT_Dot_Config()
        cfg.bert_model          = config["bert_pretrained_model"]
        cfg.trainable           = config["bert_trainable"]
        cfg.return_vecs         = config.get("in_batch_negatives",False)
        cfg.compress_dim        = config.get("bert_dot_compress_dim",-1)
        return BERT_Dot(cfg)

    def __init__(self,
                 cfg) -> None:

        super().__init__(cfg)

        self.bert_model = AutoModel.from_pretrained(cfg.bert_model)

        for p in self.bert_model.parameters():
            p.requires_grad = cfg.trainable

        self.use_compressor = cfg.compress_dim > -1
        if self.use_compressor:
            self.compressor = torch.nn.Linear(self.bert_model.config.hidden_size, cfg.compress_dim)

        self.return_vecs = cfg.return_vecs

    #def reanimate(self,added_bias,layers):
    #    self.bert_model.reanimate(added_bias,layers)

    def forward(self,
                query: Dict[str, torch.LongTensor],
                document: Dict[str, torch.LongTensor],
                use_fp16:bool = True,
                output_secondary_output: bool = False) -> Dict[str, torch.Tensor]:
        
        with torch.cuda.amp.autocast(enabled=use_fp16):

            query_vecs = self.forward_representation(query)
            document_vecs = self.forward_representation(document)

            score = torch.bmm(query_vecs.unsqueeze(dim=1), document_vecs.unsqueeze(dim=2)).squeeze(-1).squeeze(-1)

            # used for in-batch negatives, we return them for multi-gpu sync -> out of the forward() method
            if self.training and self.return_vecs:
                score = (score, query_vecs, document_vecs)

            if output_secondary_output:
                return score, {}
            return score

    def forward_representation(self,
                               tokens: Dict[str, torch.LongTensor],
                               sequence_type="n/a") -> torch.Tensor:
        
        vectors = self.bert_model(**tokens)[0][:,0,:]

        if self.use_compressor:
            vectors = self.compressor(vectors)

        return vectors

    # override loading
    def from_pretrained(self, name:str):
        self.bert_model = AutoModel.from_pretrained(name)

    def get_param_stats(self):
        return "BERT_dot: / "
    def get_param_secondary(self):
        return {}