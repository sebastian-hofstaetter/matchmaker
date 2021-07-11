from matchmaker.models.bert_dot import BERT_Dot
from matchmaker.models.published.sigir21_idcm import IDCM, IDCM_InferenceOnly

from matchmaker.models.knrm import KNRM
from matchmaker.models.conv_knrm import Conv_KNRM

from matchmaker.models.matchpyramid import MatchPyramid

from matchmaker.models.pacrr import PACRR
from matchmaker.models.co_pacrr import CO_PACRR

from matchmaker.models.duet import Duet
from matchmaker.models.drmm import DRMM

from matchmaker.models.bert_cat import *

from matchmaker.models.colbert import ColBERT

from matchmaker.models.prettr import PreTTR

from matchmaker.models.bert_dot_dualencoder import *

from matchmaker.models.parade import *

from matchmaker.models.max_p_adapter import *
from matchmaker.models.mean_p_adapter import *

from matchmaker.models.published.cikm20_tk_sparse import *
from matchmaker.models.published.sigir20_tkl import *
from matchmaker.models.published.ecai20_tk import *

from matchmaker.modules.neuralIR_encoder import *

from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder

from allennlp.modules.token_embedders import Embedding
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from matchmaker.modules.bert_embedding_token_embedder import BertEmbeddingTokenEmbedder
from matchmaker.modules.neuralIR_encoder import *
from matchmaker.utils.input_pipeline import *

#from matchmaker.models.private.qa_bert_cat import *
#from matchmaker.models.private.bert_dot_qa import *


def get_word_embedder(config):

    padding_idx = 0
    word_embedder = None
    # embedding layer (use pre-trained, but make it trainable as well)
    if config["token_embedder_type"] == "embedding":
        vocab = Vocabulary.from_files(config["vocab_directory"])
        tokens_embedder = Embedding(vocab=vocab,
                                    pretrained_file= config["pre_trained_embedding"],
                                    embedding_dim=config["pre_trained_embedding_dim"],
                                    trainable=config["train_embedding"],
                                    padding_index=0,
                                    sparse=config["sparse_gradient_embedding"])
        word_embedder = BasicTextFieldEmbedder({"tokens": tokens_embedder})
        
    elif config["token_embedder_type"] == "bert_embedding":
        vocab = None
        bert_embedding = BertEmbeddingTokenEmbedder(config["bert_pretrained_model"],pos_embeddings=config["bert_emb_pos"],keep_layers=config["bert_emb_keep_layers"])
        bert_embedding.bert_embeddings.word_embeddings.sparse = config["sparse_gradient_embedding"]
        bert_embedding.bert_embeddings.token_type_embeddings.sparse = config["sparse_gradient_embedding"]
        word_embedder = BasicTextFieldEmbedder({"tokens":bert_embedding},
                                                allow_unmatched_keys = True,
                                                embedder_to_indexer_map={"tokens":{"tokens":"tokens","offsets":"tokens-offsets","token_type_ids":"tokens-type-ids"}})
    elif config["token_embedder_type"] == "bert_vectors":
        vocab = None
        bert_embedding = PretrainedTransformerEmbedder(config["bert_pretrained_model"],requires_grad=config["train_embedding"])#,top_layer_only=True)

        #if config["bert_emb_layers"] > -1:
        #    bert_embedding.bert_model.encoder.layer = bert_embedding.bert_model.encoder.layer[:config["bert_emb_layers"]]

        word_embedder = BasicTextFieldEmbedder({"tokens":bert_embedding},
                                                allow_unmatched_keys = True,
                                                embedder_to_indexer_map={"tokens":{"input_ids":"tokens","offsets":"tokens-offsets","token_type_ids":"tokens-type-ids"}})
    elif config["token_embedder_type"] == "huggingface_bpe":
        files = config["bpe_vocab_files"].split(";")
        tok = CharBPETokenizer(files[0],files[1])
        padding_idx = tok.token_to_id("<pad>")
        tokens_embedder = Embedding(num_embeddings=tok.get_vocab_size(),
                                    embedding_dim= config["pre_trained_embedding_dim"],
                                    trainable= config["train_embedding"],
                                    padding_index=padding_idx,
                                    sparse=config["sparse_gradient_embedding"])
        word_embedder = BasicTextFieldEmbedder({"tokens": tokens_embedder})

    elif config["token_embedder_type"] in ["bert_cat","bert_cls","bert_dot","bert_tower"]:
        model = config["bert_pretrained_model"]
        if "facebook/dpr" in config["bert_pretrained_model"]: # ugh .. 
            model= "bert-base-uncased"                        # should be identical (judging from paper + huggingface doc)
        padding_idx = PretrainedTransformerIndexer(model_name=model)._tokenizer.pad_token_id
    else:
        logger.error("token_embedder_type %s not known",config["token_embedder_type"])
        exit(1)

    return word_embedder,padding_idx

def build_model(model,encoder_type,word_embedder,config):
    if encoder_type == None:
        pass
    elif encoder_type == NeuralIR_Encoder_WithIdfs or encoder_type == NeuralIR_Encoder_WithIdfs_PassThrough:
        idf_embedder = None
        if config["token_embedder_type"] == "embedding":
            idf_embedder = Embedding(vocab=vocab,
                                    pretrained_file= config["idf_path"],
                                    embedding_dim=1,
                                    trainable=config["idf_trainable"],
                                    padding_index=0,
                                    sparse=config["sparse_gradient_embedding"])
            idf_embedder = BasicTextFieldEmbedder({"tokens":idf_embedder})#, 
                                                  #allow_unmatched_keys = True, 
                                                  #embedder_to_indexer_map={"tokens":{"tokens":"tokens"}})
        model = encoder_type(word_embedder, idf_embedder, model)    
    else:
        model = encoder_type(word_embedder, model)

    return model

def get_model(config,word_embedder,padding_idx):
    encoder_type = NeuralIR_Encoder

    model_conf = config["model"]

    wrap_max_p = False
    if model_conf.startswith("maxP->"):
        wrap_max_p = True
        model_conf=model_conf.replace("maxP->","")

    wrap_mean_p = False
    if model_conf.startswith("meanP->"):
        wrap_mean_p = True
        model_conf=model_conf.replace("meanP->","")

    #
    # pour published models
    #
    if model_conf == "TK": model = ECAI20_TK.from_config(config,word_embedder.get_output_dim())
    elif model_conf == "TKL": model = TKL_sigir20.from_config(config,word_embedder.get_output_dim())
    elif model_conf == "TK_Sparse": model = CIKM20_TK_Sparse.from_config(config,word_embedder.get_output_dim())
    elif model_conf == "Bert_patch" or model_conf == "IDCM":
        model = IDCM.from_config(config,padding_idx=padding_idx)
        encoder_type = None

    #
    # baselines with text only
    #
    elif model_conf == "knrm": model = KNRM.from_config(config,word_embedder.get_output_dim())
    elif model_conf == "conv_knrm": model = Conv_KNRM.from_config(config,word_embedder.get_output_dim())
    elif model_conf == "match_pyramid": model = MatchPyramid.from_config(config,word_embedder.get_output_dim())
    elif model_conf == "drmm": model = DRMM(word_embedder,10)

    #
    # baseline models with idf use
    #
    elif model_conf == "pacrr":
        model = PACRR.from_config(config,word_embedder.get_output_dim())
        encoder_type = NeuralIR_Encoder_WithIdfs
    elif model_conf == "co_pacrr":
        model = CO_PACRR.from_config(config,word_embedder.get_output_dim())
        encoder_type = NeuralIR_Encoder_WithIdfs
    elif model_conf == "duet":
        model = Duet.from_config(config,word_embedder.get_output_dim())
        encoder_type = NeuralIR_Encoder_WithIdfs

    #
    # bert models
    #
    else:
        encoder_type = None
    
        if model_conf == "bert_cls" or model_conf == "bert_cat": model = BERT_Cat.from_config(config)
        elif model_conf == "bert_tower" or model_conf == "bert_dot":  model = BERT_Dot.from_config(config)
        #elif model_conf == "QA_Bert_cat": model = QA_Bert_cat(bert_model = config["bert_pretrained_model"],trainable=config["bert_trainable"])
        #elif model_conf == "bert_dot_qa":
        #    model = Bert_dot_qa(bert_model = config["bert_pretrained_model"],return_vecs=config.get("in_batch_negatives",False),trainable=config["bert_trainable"])

        elif model_conf == "bert_dot_dualencoder":
            model = Bert_dot_dualencoder(bert_model_document= config["bert_pretrained_model"],bert_model_query=config["bert_pretrained_model_secondary"],return_vecs=config["in_batch_negatives"],trainable=config["bert_trainable"])

        elif model_conf == "ColBERT":
            model = ColBERT.from_config(config)

        elif model_conf == "PreTTR" or model_conf == "Bert_Split":
            model = PreTTR.from_pretrained(config["bert_pretrained_model"])

        elif model_conf == "Parade":
            model = Parade.from_config(config,padding_idx=padding_idx)

        else:
            print("Model %s not known",config["model"])
            exit(1)

    if wrap_max_p or wrap_mean_p:
        if "inner_model_path" in config:
            load_result = model.load_state_dict(torch.load(config["inner_model_path"],map_location="cpu"),strict=False)
            logger.info('Warmstart inner model from:  %s', config["inner_model_path"])
            logger.info(load_result)
            print("Inner-Warmstart Result:",load_result)
        if wrap_max_p:
            model = MaxPAdapter.from_config(config,inner_model=model,padding_idx=padding_idx)
        if wrap_mean_p:
            model = MeanPAdapter.from_config(config,inner_model=model,padding_idx=padding_idx)

    return model, encoder_type