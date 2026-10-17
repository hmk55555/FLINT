import importlib
from unicodedata import name
import torch
import torch.nn as nn
import transformers
from transformers import BertPreTrainedModel, BertModel, AutoTokenizer, AutoModel, AutoConfig
from transformers import RobertaModel, RobertaPreTrainedModel
from transformers import XLMRobertaModel, XLMRobertaConfig
from transformers import ElectraModel, ElectraPreTrainedModel
from transformers import DebertaV2Model, DebertaV2PreTrainedModel
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from colbert.utils.utils import torch_load_dnn
from colbert.modeling.flint_perceiver import SharedPerceiverBottleneck

class XLMRobertaPreTrainedModel(RobertaPreTrainedModel):
    """
    This class overrides [`RobertaModel`]. Please check the superclass for the appropriate documentation alongside
    usage examples.
    """

    config_class = XLMRobertaConfig


base_class_mapping={
    "roberta-base": RobertaPreTrainedModel,
    "google/electra-base-discriminator": ElectraPreTrainedModel,
    "xlm-roberta-base": XLMRobertaPreTrainedModel,
    "xlm-roberta-large": XLMRobertaPreTrainedModel,
    "bert-base-uncased": BertPreTrainedModel,
    "bert-large-uncased": BertPreTrainedModel,
    "microsoft/mdeberta-v3-base": DebertaV2PreTrainedModel,
    "bert-base-multilingual-uncased": BertPreTrainedModel


}

model_object_mapping = {
    "roberta-base": RobertaModel,
    "google/electra-base-discriminator": ElectraModel,
    "xlm-roberta-base": XLMRobertaModel,
    "xlm-roberta-large": XLMRobertaModel,
    "bert-base-uncased": BertModel,
    "bert-large-uncased": BertModel,
    "microsoft/mdeberta-v3-base": DebertaV2Model,
    "bert-base-multilingual-uncased": BertModel

}


transformers_module = dir(transformers)

def find_class_names(model_type, class_type):
    model_type = model_type.replace("-", "").lower()
    for item in transformers_module:
        if model_type + class_type == item.lower():
            return item

    return None


def is_modernbert(config):
    return getattr(config, "model_type", "") == "modernbert"


def class_factory(name_or_path):
    loadedConfig  = AutoConfig.from_pretrained(name_or_path, trust_remote_code=True)

    if getattr(loadedConfig, "auto_map", None) is None:
        model_type = loadedConfig.model_type
        pretrained_class = find_class_names(model_type, 'pretrainedmodel')
        model_class = find_class_names(model_type, 'model')
        
        if pretrained_class is not None:
            pretrained_class_object = getattr(transformers, pretrained_class)
        elif model_type == 'xlm-roberta':
            pretrained_class_object = XLMRobertaPreTrainedModel
        elif base_class_mapping.get(name_or_path) is not None:
            pretrained_class_object = base_class_mapping.get(name_or_path)
        else:
            raise ValueError("Could not find correct pretrained class for the model type {model_type} in transformers library")

        if model_class != None:
            model_class_object = getattr(transformers, model_class)
        elif model_object_mapping.get(name_or_path) is not None:
            model_class_object = model_object_mapping.get(name_or_path)
        else:
            raise ValueError("Could not find correct model class for the model type {model_type} in transformers library")
    else:
        assert "AutoModel" in loadedConfig.auto_map, "The custom model should have AutoModel class in the config.automap"
        model_class = loadedConfig.auto_map["AutoModel"]
        assert model_class.endswith("Model")
        pretrained_class = model_class.replace("Model", "PreTrainedModel")
        model_class_object = get_class_from_dynamic_module(model_class, name_or_path)
        pretrained_class_object = get_class_from_dynamic_module(pretrained_class, name_or_path)


    class HF_ColBERT(pretrained_class_object):
        """
            Shallow wrapper around HuggingFace transformers. All new parameters should be defined at this level.

            This makes sure `{from,save}_pretrained` and `init_weights` are applied to new parameters correctly.
        """
        _keys_to_ignore_on_load_unexpected = [r"cls"]

        def __init__(self, config, colbert_config):
            super().__init__(config)

            self.config = config
            self.dim = colbert_config.dim
            self.model_variant = colbert_config.model_variant
            assert self.model_variant in ("flint", "constbert", "colbert"), self.model_variant

            if self.model_variant == "flint":
                is_mb = is_modernbert(config)
                self.perceiver = SharedPerceiverBottleneck(
                    num_query_latents=colbert_config.num_query_latents,
                    num_doc_latents=colbert_config.num_doc_latents,
                    dim=config.hidden_size,
                    dropout=colbert_config.perceiver_dropout,
                    ff_ratio=(2 if is_mb else 4),
                    attn_impl=("sdpa" if is_mb else "mha"),
                )

            self.linear = nn.Linear(config.hidden_size, colbert_config.dim, bias=False)
            if self.model_variant == "flint" and not is_mb:
                self.doc_linear = nn.Linear(config.hidden_size, colbert_config.dim, bias=False)
            else:
                self.doc_linear = self.linear
            if self.model_variant == "constbert":
                self.doc_project = nn.Linear(colbert_config.doc_maxlen, colbert_config.num_doc_latents, bias=False)
            setattr(self,self.base_model_prefix, model_class_object(config))

            # if colbert_config.relu:
            #     self.score_scaler = nn.Linear(1, 1)

            self.init_weights()

            # if colbert_config.relu:
            #     self.score_scaler.weight.data.fill_(1.0)
            #     self.score_scaler.bias.data.fill_(-8.0)

        def _init_weights(self, module):
            super()._init_weights(module)

            if self.model_variant != "flint":
                if module is getattr(self, 'linear', None) or module is getattr(self, 'doc_project', None):
                    nn.init.trunc_normal_(module.weight, std=self.config.initializer_range)
            elif is_modernbert(self.config):
                if isinstance(module, SharedPerceiverBottleneck):
                    module.reset_parameters()
                elif module is getattr(self, 'linear', None):
                    nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, SharedPerceiverBottleneck):
                nn.init.trunc_normal_(module.query_latents, std=self.config.initializer_range)
                nn.init.trunc_normal_(module.doc_latents, std=self.config.initializer_range)
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)

        @property
        def LM(self):
            base_model_prefix = getattr(self, "base_model_prefix")
            return getattr(self, base_model_prefix)

        @staticmethod
        def _backbone_load_kwargs(name_or_path):
            extra = {}
            try:
                cfg = AutoConfig.from_pretrained(name_or_path, trust_remote_code=True)
            except Exception:
                return extra
            if hasattr(cfg, "reference_compile"):
                extra["reference_compile"] = False
            if is_modernbert(cfg) and torch.cuda.is_available():
                extra["attn_implementation"] = "flash_attention_2"
            return extra

        @classmethod
        def from_pretrained(cls, name_or_path, colbert_config):
            if name_or_path.endswith('.dnn'):
                dnn = torch_load_dnn(name_or_path)
                base = dnn.get('arguments', {}).get('model', 'bert-base-uncased')

                extra = cls._backbone_load_kwargs(base)
                obj = super().from_pretrained(base, state_dict=dnn['model_state_dict'], colbert_config=colbert_config, **extra)
                obj.base = base

                return obj

            extra = cls._backbone_load_kwargs(name_or_path)
            obj = super().from_pretrained(name_or_path, colbert_config=colbert_config, **extra)
            obj.base = name_or_path

            return obj

        @staticmethod
        def raw_tokenizer_from_pretrained(name_or_path):
            if name_or_path.endswith('.dnn'):
                dnn = torch_load_dnn(name_or_path)
                base = dnn.get('arguments', {}).get('model', 'bert-base-uncased')

                obj = AutoTokenizer.from_pretrained(base)
                obj.base = base

                return obj

            obj = AutoTokenizer.from_pretrained(name_or_path)
            obj.base = name_or_path

            return obj

    return HF_ColBERT
