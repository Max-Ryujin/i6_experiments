import torch
from torch import nn
from typing import Callable, List, Optional
from enum import Enum
from dataclasses import dataclass

from i6_models.primitives.specaugment import specaugment_v1_by_length

from returnn.torch.context import get_run_ctx

from .model_lah_carryover_cfg import ModelConfig

from i6_models.assemblies.conformer.conformer_v1 import ConformerEncoderV1Config, ConformerBlockV1Config

# frontend imports
from i6_models.config import ModuleFactoryV1
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1
from ...rnnt.streaming_conformer.generic_frontend_v2 import GenericFrontendV2

# feature extract and conformer module imports
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1
from i6_models.parts.conformer.norm import LayerNormNC
from i6_models.parts.conformer.convolution import ConformerConvolutionV1Config
from i6_models.parts.conformer.feedforward import ConformerPositionwiseFeedForwardV1Config
from i6_models.parts.conformer.mhsa import ConformerMHSAV1Config

from ...rnnt.conformer_0924.conformer_uni_lah_carryover import ConformerEncoderCOV1
from ...rnnt.conformer_1124.conf_intra_batch_sampling import ConformerEncoderInBatchSamplingV1

from ..conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6 import (
    prior_step,
    prior_init_hook,
    prior_finish_hook,
)

from ...rnnt.auxil.functional import num_samples_to_frames, Mode, mask_tensor


class Model(torch.nn.Module):
    def __init__(self, model_config_dict, **kwargs):
        super().__init__()
        # net_args are passed as a dict to returnn and here the config is retransformed into its dataclass
        self.cfg = ModelConfig.from_dict(model_config_dict)

        frontend_config = self.cfg.frontend_config
        conformer_config = ConformerEncoderV1Config(
            num_layers=self.cfg.num_layers,
            frontend=ModuleFactoryV1(module_class=VGG4LayerActFrontendV1, cfg=frontend_config),
            block_cfg=ConformerBlockV1Config(
                ff_cfg=ConformerPositionwiseFeedForwardV1Config(
                    input_dim=self.cfg.conformer_size,
                    hidden_dim=self.cfg.ff_dim,
                    dropout=self.cfg.ff_dropout,
                    activation=nn.functional.silu,
                ),
                mhsa_cfg=ConformerMHSAV1Config(
                    input_dim=self.cfg.conformer_size,
                    num_att_heads=self.cfg.num_heads,
                    att_weights_dropout=self.cfg.att_weights_dropout,
                    dropout=self.cfg.mhsa_dropout,
                ),
                conv_cfg=ConformerConvolutionV1Config(
                    channels=self.cfg.conformer_size,
                    kernel_size=self.cfg.conv_kernel_size,
                    dropout=self.cfg.conv_dropout,
                    activation=nn.functional.silu,
                    norm=LayerNormNC(self.cfg.conformer_size),
                ),
            ),
        )

        self.feature_extraction = LogMelFeatureExtractionV1(cfg=self.cfg.feature_extraction_config)
        self.conformer = ConformerEncoderInBatchSamplingV1(cfg=conformer_config)
        self.final_linear = nn.Linear(self.cfg.conformer_size, self.cfg.label_target_size + 1)  # + CTC blank
        self.final_dropout = nn.Dropout(p=self.cfg.final_dropout)
        self.specaug_start_epoch = self.cfg.specauc_start_epoch

        self.lookahead_size = self.cfg.lookahead_size
        self.mode: Optional[Mode] = Mode.STREAMING
        self.carry_over_size = self.cfg.carry_over_size

    def extract_features(self, raw_audio: torch.Tensor, raw_audio_len: torch.Tensor):
        squeezed_features = torch.squeeze(raw_audio, dim=-1)
        with torch.no_grad():
            audio_features, audio_features_len = self.feature_extraction(squeezed_features, raw_audio_len)

            run_ctx = get_run_ctx()
            if self.training and run_ctx.epoch >= self.specaug_start_epoch:
                audio_features_masked_2 = specaugment_v1_by_length(
                    audio_features,
                    time_min_num_masks=2,
                    time_max_mask_per_n_frames=self.cfg.specaug_config.repeat_per_n_frames,
                    time_mask_max_size=self.cfg.specaug_config.max_dim_time,
                    freq_min_num_masks=2,
                    freq_mask_max_size=self.cfg.specaug_config.max_dim_feat,
                    freq_max_num_masks=self.cfg.specaug_config.num_repeat_feat,
                )
            else:
                audio_features_masked_2 = audio_features

        conformer_in = audio_features_masked_2
        mask = mask_tensor(conformer_in, audio_features_len)

        return conformer_in, mask

    def prep_streaming_input(self, conformer_in, mask):
        batch_size = conformer_in.size(0)

        chunk_size_frames = num_samples_to_frames(
            n_fft=self.feature_extraction.n_fft,
            hop_length=self.feature_extraction.hop_length,
            center=self.feature_extraction.center,
            num_samples=int(self.cfg.chunk_size)
        )

        # pad conformer time-dim to be able to chunk (by reshaping) below
        time_dim_pad = -conformer_in.size(1) % chunk_size_frames
        # (B, T, ...) -> (B, T+time_dim_pad, ...) = (B, T', ...)
        conformer_in = torch.nn.functional.pad(conformer_in, (0, 0, 0, time_dim_pad),
                                               "constant", 0)
        mask = torch.nn.functional.pad(mask, (0, time_dim_pad), "constant", False)

        # separate chunks to signal the conformer that we are chunking input
        conformer_in = conformer_in.view(batch_size, -1, chunk_size_frames,
                                         conformer_in.size(-1))  # (B, (T'/C), C, F) = (B, N, C, F)
        mask = mask.view(batch_size, -1, chunk_size_frames)  # (B, N, C)

        return conformer_in, mask

    def forward(
            self, raw_audio: torch.Tensor, raw_audio_len: torch.Tensor,
    ):
        """
        :param raw_audio: Audio samples as [B, T, 1]
        :param raw_audio_len: length of T as [B]
        :param labels: [B, N]
        :param labels_len: length of N as [B]
        :return: logprobs [B, T + N, #labels + blank]
        """
        conformer_in, mask = self.extract_features(raw_audio, raw_audio_len)
        conformer_in, mask = self.prep_streaming_input(conformer_in, mask)

        # assert conformer_in.size(2) == (self.cfg.chunk_size / 0.01)

        conformer_out, out_mask = self.conformer(conformer_in, mask,
                                                 lookahead_size=self.lookahead_size,
                                                 carry_over_size=self.carry_over_size,
                                                 online_scale=self.cfg.online_model_scale)

       # final linear layer
        conformer_out = self.final_dropout(conformer_out)
        logits = self.final_linear(conformer_out)
        log_probs = torch.log_softmax(logits, dim=2)

        return log_probs, torch.sum(out_mask, dim=1)


def train_step(*, model: Model, data, run_ctx, **kwargs):
    # import for training only, will fail on CPU servers
    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:size1"].to("cpu")  # [B]

    labels = data["labels"]  # [B, N] (sparse)
    labels_len = data["labels:size1"]  # [B, N]

    num_phonemes = torch.sum(labels_len)


    logprobs, audio_features_len = model(
        raw_audio=raw_audio, raw_audio_len=raw_audio_len,
    )

    transposed_logprobs = torch.permute(logprobs, (1, 0, 2))  # CTC needs [T, B, #vocab + 1]

    ctc_loss = nn.functional.ctc_loss(
        transposed_logprobs,
        labels,
        input_lengths=audio_features_len,
        target_lengths=labels_len,
        blank=model.cfg.label_target_size,
        reduction="sum",
        zero_infinity=True,
    )

    run_ctx.mark_as_loss(name="ctc.smpl", loss=ctc_loss, inv_norm_factor=num_phonemes,)
