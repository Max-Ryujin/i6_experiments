"""
Config for the base CTC models v4, including specaug start time
"""

from dataclasses import dataclass

import torch
from torch import nn
from typing import Callable, List, Literal, Optional, Type, Union

from i6_models.assemblies.conformer.conformer_rel_pos_v1 import ConformerRelPosBlockV1Config, ConformerRelPosBlockV1
from i6_models.config import ModuleFactoryV1, ModelConfiguration
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1Config
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1Config


@dataclass(kw_only=True)
class VGG4LayerActFrontendV1Config_mod(VGG4LayerActFrontendV1Config):
    activation_str: str = ""
    activation: Optional[Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]]] = None

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        activation_str = d.pop("activation_str")
        if activation_str == "ReLU":
            from torch.nn import ReLU
            activation = ReLU()
        else:
            assert False, "Unsupported activation %s" % d["activation_str"]
        d["activation"] = activation
        return VGG4LayerActFrontendV1Config(**d)


@dataclass
class PredictorConfig(ModelConfiguration):
    symbol_embedding_dim: int
    emebdding_dropout: float
    num_lstm_layers: int
    lstm_hidden_dim: int
    lstm_dropout: float
    
    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        return PredictorConfig(**d)


@dataclass
class SpecaugConfig(ModelConfiguration):
    repeat_per_n_frames: int
    max_dim_time: int
    num_repeat_feat: int
    max_dim_feat: int

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        return SpecaugConfig(**d)


@dataclass
class ConformerPosEmbConfig(ModelConfiguration):
    learnable_pos_emb: bool
    rel_pos_clip: Optional[int]
    with_linear_pos: bool
    with_pos_bias: bool
    separate_pos_emb_per_head: bool
    pos_emb_dropout: float


@dataclass
class ModelConfig():
    feature_extraction_config: LogMelFeatureExtractionV1Config
    frontend_config: VGG4LayerActFrontendV1Config
    predictor_config: PredictorConfig
    specaug_config: SpecaugConfig
    pos_emb_config: ConformerPosEmbConfig
    specauc_start_epoch: int
    label_target_size: int
    conformer_size: int
    num_layers: int
    num_heads: int
    ff_dim: int
    att_weights_dropout: float
    conv_dropout: float
    ff_dropout: float
    mhsa_dropout: float
    mhsa_with_bias: bool
    conv_kernel_size: int
    final_dropout: float
    joiner_dim: int
    joiner_activation: str
    joiner_dropout: float
    dropout_broadcast_axes: Optional[Literal["B", "T", "BT"]]
    module_list: List[str]
    module_scales: List[float]
    aux_ctc_loss_layers: Optional[List[int]]
    aux_ctc_loss_scales: Optional[List[float]]

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        d["feature_extraction_config"] = LogMelFeatureExtractionV1Config(**d["feature_extraction_config"])
        d["frontend_config"] = VGG4LayerActFrontendV1Config_mod.from_dict(d["frontend_config"])
        d["specaug_config"] = SpecaugConfig.from_dict(d["specaug_config"])
        d["pos_emb_config"] = ConformerPosEmbConfig(**d["pos_emb_config"])
        d["predictor_config"] = PredictorConfig.from_dict(d["predictor_config"])
        return ModelConfig(**d)


