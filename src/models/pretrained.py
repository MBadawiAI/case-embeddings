from data_utils.config import ModelConfig
import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM, AutoTokenizer, BitsAndBytesConfig

class PretrainedModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        ## COPIED EXAMPLE CODE, need to generify. Hyperparameters would be stored in the config
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",  # or "fp4" for float 4-bit quantization
            bnb_4bit_use_double_quant=True  # use double quantization for better performance
        )
        config = AutoTokenizer.from_pretrained(
            config.tokenizer,
            trust_remote_code=True
        )

        model = AutoModelForMaskedLM.from_pretrained(
            "facebook/xlm-roberta-large",
            dtype=torch.float16,
            device_map="auto",
            attn_implementation="flash_attention_2",
            quantization_config=quantization_config
        )

        return model

    def forward(self, x):
        pass