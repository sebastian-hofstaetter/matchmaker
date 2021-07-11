import yaml
from typing import Dict, List
import os


def get_config(config_path: List[str], overwrites: str = None) ->Dict[str, any]:
    cfg = {}
    for path in config_path:
        with open(os.path.join(os.getcwd(), path), 'r') as ymlfile:
            cfg.update(yaml.load(ymlfile, Loader=yaml.FullLoader))

    if overwrites is not None and overwrites != "":
        over_parts = [yaml.load(x.replace("\\n", "\n"), Loader=yaml.FullLoader) for x in overwrites.split(",")]

        for d in over_parts:
            for key, value in d.items():
                cfg[key] = value

    _auto_config_filler(cfg)

    return cfg


def get_config_single(config_path: str, overwrites: str = None) -> Dict[str, any]:

    config_path_yaml = config_path
    if not config_path.endswith("config.yaml"):
        config_path_yaml = os.path.join(config_path, "config.yaml")

    # assume huggingface looks like: sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco
    if not os.path.exists(config_path_yaml) and not os.path.isabs(config_path):
        local_hf_config = os.path.join(os.getcwd(), "config", "huggingface_modelhub", config_path + ".yaml")
        if os.path.exists(local_hf_config):
            config_path_yaml = local_hf_config
        else:
            raise Exception(config_path + " does not exist locally & is not a known huggingface config (if using hf you need to create a local config file in config/huggingface_modelhub)")

    with open(config_path_yaml, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    if overwrites is not None and overwrites != "":
        over_parts = [yaml.load(x.replace("\\n", "\n"), Loader=yaml.FullLoader) for x in overwrites.split(",")]

        for d in over_parts:
            for key, value in d.items():
                cfg[key] = value

    return cfg


def save_config(config_path: str, config: Dict[str, any]):
    with open(config_path, 'w') as ymlfile:
        yaml.safe_dump(config, ymlfile, default_flow_style=False)


_auto_config_info = {
    ("model_input_type", "model"): [
        (["bert_cat","bert_cls"], "concatenated"),
        (["bert_tower","bert_dot", "TK", "TKL", "ColBERT","PreTTR","IDCM"], "independent")
    ],
    ("token_embedder_type", "model"): [
        (["bert_cat","bert_cls"], "bert_cat"),
        (["bert_tower","bert_dot", "ColBERT","PreTTR","IDCM"], "bert_dot"),
        (["TK", "TKL"], "embedding")
    ]
}


def _auto_config_filler(config: Dict[str, any]):
    for (auto_set, auto_switch), cases in _auto_config_info.items():
        if config.get(auto_set, "") == "auto":
            success = False
            switch_target = config.get(auto_switch, "")
            for case, value in cases:
                if switch_target in case:
                    config[auto_set] = value
                    success = True
                    break
            if not success:
                raise Exception("Could not fill in auto config for: " + str(auto_set)+" where switch is based on: " + str(switch_target))
