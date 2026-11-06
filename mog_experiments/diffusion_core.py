# This file specifies an interface that the generative model must implement
import json
import os
from datetime import datetime

import numpy as np

class ModelConfig:
    model_name: str =  None
    state_shape = None
    index_set: list = []
    deets = {
        
    }
    def to_dict(self):
        return {
            "model_name": self.model_name,
            "state_shape": self.state_shape,
            "index_set": self.index_set,
            "deets": self.deets,
        }


class Core: # Abstract class defining interaction protocol for a diffusion core
    def __init__(self, config):
        self.state_shape = None
        self.model_config = config
        self.t = None
        # self.X # current state for the model, has shape self.state_shape
        # self.tentative_states # list of proposed states from last call to propose function, has shape [M, self.state_shape] for some M
        # self.model_details = ModelConfig()

    def load_from_config(self, config_path):
        pass

    def propose(self, S = 1): # generates proposal or list thereof, returns np.array([S, self.state_shape])
        pass

    def propose_external(self, init_state, index): # propose state from init_state, index
        pass

    def commit(self, state): # accepts state of shape self.state_shape and saves it as self.state
        pass
        # self.state

    def reset(self):
        pass

    def collect_run(self):
        pass

    def save(self, states, data_dir: str, run_id: str) -> dict:
        # This must be defined in subclasses because the saving process depends on the datatype of the state
        pass

# def load_core_from_config(config) -> Core: # accepts a config and uses model_id to load appropriate core
#     """Load a diffusion core from a saved config file."""
#     cfg = config
#     model_name = cfg.get("model_name", None)
#     if model_name is None:
#         raise ValueError("Config file does not specify model_name")
#     if model_name == "mixture_model":
#         from toy_data.mixture_model import MixtureModel
#         core = MixtureModel(None, None, cfg, load_from_config=True)
#     else:
#         raise ValueError(f"Unknown model_name {model_name} in config")
#     return core

def load_core_from_config(config) -> Core:  # accepts a config and uses model_id to load appropriate core
    """Load a diffusion core from a saved config file."""
    cfg = config
    model_name = cfg.get("model_name", None)
    if model_name is None:
        raise ValueError("Config file does not specify model_name")

    if model_name == "mixture_model":
        from mixture_model import MixtureModel
        core = MixtureModel(None, None, cfg, load_from_config=True)

    elif model_name == "masked_lm":
        # Load a MaskedLMCore from config (recreates tokenizer + MLM)
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        from masked_lm_core import MaskedLMCore

        deets = cfg.get("deets", {})
        hf_model_id = deets["hf_model_id"]

        tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
        mlm_model = AutoModelForMaskedLM.from_pretrained(hf_model_id)
        core = MaskedLMCore(cfg, tokenizer, mlm_model, load_from_config=True)
    
    elif model_name == "llada":
        # Lazy import to avoid circular dependency if in same package
        from llada_core import LLaDACore
        
        hf_id = deets.get("hf_model_id", "GSAI-ML/LLaDA-8B-Instruct")
        quant_mode = deets.get("quantization", None)
        dummy_mode = deets.get("dummy", False)
        
        # 1. Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        
        # 2. Load Model (Handle Quantization)
        if dummy_mode:
            print("Loading LLaDACore in DUMMY mode (No Model).")
            model = None
        else:
            print(f"Loading LLaDA Model: {hf_id}...")
            if quant_mode == "4bit":
                print(" -> Using 4-bit Quantization (BitsAndBytes)")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                model = AutoModel.from_pretrained(
                    hf_id, 
                    trust_remote_code=True, 
                    quantization_config=bnb_config, 
                    device_map="auto"
                )

        core = LLaDACore(cfg, tokenizer, model, load_from_config=True)

    else:
        raise ValueError(f"Unknown model_name {model_name} in config")

    return core


