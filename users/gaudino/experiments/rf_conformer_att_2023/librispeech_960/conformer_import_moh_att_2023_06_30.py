"""Param Import
"""

from __future__ import annotations

from typing import Optional, Any, Tuple, Dict, Sequence, List
from itertools import product

from i6_experiments.users.gaudino.models.asr.rf.nn_lm.lm_import_2023_11_09 import (
    Trafo_LM_Model,
)
from sisyphus import tk

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.encoder.conformer import ConformerEncoder, ConformerConvSubsample

from i6_experiments.users.gaudino.models.asr.rf.nn_lm.lm_import_2023_09_03 import (
    LSTM_LM_Model,
    # MakeModel,
)
from i6_experiments.users.gaudino.models.asr.rf.ilm_import_2024_04_17 import (
    MiniAtt_ILM_Model,
)
from i6_experiments.users.gaudino.model_interfaces.model_interfaces import ModelDef, TrainDef

from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.model_recogs.model_recog import (
    model_recog,
)
from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.model_recogs.model_recog_time_sync import (
    model_recog_time_sync,
)
from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.model_recogs.model_recog_ts_espnet import (
    model_recog_ts_espnet,
)

from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.model_recogs.model_recog_ts_robin import (
    model_recog as model_recog_ts_robin,
)
from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.model_recogs.model_recog_dump import (
    model_recog_dump,
)
from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.model_recogs.model_recog_ctc_greedy import (
    model_recog_ctc,
)
from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.model_recogs.model_forward_ctc_max import (
    model_forward_ctc_max,
)



import torch
import numpy

# from functools import partial


# From Mohammad, 2023-06-29
# dev-clean  2.27
# dev-other  5.39
# test-clean  2.41
# test-other  5.51
# _returnn_tf_config_filename = "/work/asr4/zeineldeen/setups-data/librispeech/2022-11-28--conformer-att/work/i6_core/returnn/search/ReturnnSearchJobV2.1oORPHJTAcW0/output/returnn.config"
# E.g. via /u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work
_returnn_tf_ckpt_filename = "i6_core/returnn/training/AverageTFCheckpointsJob.BxqgICRSGkgb/output/model/average.index"
_torch_ckpt_filename_w_lstm_lm = "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/full_w_lm_import_2023_10_18/average.pt"
_torch_ckpt_filename_w_trafo_lm = "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/librispeech/full_w_trafo_lm_import_2024_02_05/average.pt"
_torch_ckpt_filename_base_model = "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/librispeech/base_model/average.pt"
# The model gets raw features (16khz) and does feature extraction internally.
_log_mel_feature_dim = 80


def sis_run_with_prefix(prefix_name: str = None):
    """run the exp"""
    from .sis_setup import get_prefix_for_config
    from i6_core.returnn.training import PtCheckpoint
    from i6_experiments.users.zeyer.model_interfaces import ModelWithCheckpoint
    from i6_experiments.users.gaudino.recog import recog_model as recog_model_old
    from i6_experiments.users.gaudino.recog_2 import recog_model
    from i6_experiments.users.gaudino.forward import forward_model

    from i6_experiments.users.gaudino.datasets.librispeech import (
        get_librispeech_task_bpe10k_raw,
    )
    from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.support.search_errors import (
        ComputeSearchErrorsJob,
    )

    if not prefix_name:
        prefix_name = get_prefix_for_config(__file__)

    task = get_librispeech_task_bpe10k_raw(with_eos_postfix=True)

    bsf = 10
    single_seq_prefix_name = prefix_name + "/single_seq"
    prefix_name_60 = prefix_name + f"/bsf60"
    prefix_name_40 = prefix_name + f"/bsf40"
    prefix_name = prefix_name + f"/bsf{bsf}"

    ### Experiments without LM and with LSTM LM

    new_chkpt_path = tk.Path(
        _torch_ckpt_filename_w_lstm_lm, hash_overwrite="torch_ckpt_w_lstm_lm"
    )
    new_chkpt = PtCheckpoint(new_chkpt_path)
    model_with_checkpoint = ModelWithCheckpoint(
        definition=from_scratch_model_def, checkpoint=new_chkpt
    )

    recog_config = {
        "model_args": {
            "add_lstm_lm": True,
        },
    }

    # att only
    for beam_size in []:
        recog_name = f"/att_beam{beam_size}"
        name = prefix_name_40 + recog_name
        search_args = {
            "beam_size": beam_size,
            "bsf": 40,
        }
        recog_config["search_args"] = search_args

        res = recog_model(
            task,
            model_with_checkpoint,
            model_recog,
            dev_sets=None,  # None for all
            config=recog_config,
            name=name,
            compute_search_errors=True,
        )
        tk.register_output(
            name + f"/recog_results",
            res.output,
        )

    # espnet ctc prefix decoder
    # beam 12/32: {"dev-clean": 2.83, "dev-other": 6.69, "test-clean": 3.07, "test-other": 7.02}
    for prior_scale, beam_size in product([0.0], []):
        name = (
            prefix_name
            + f"/ctc_prefix_fix"
            + (f"_prior{prior_scale}" if prior_scale != 0.0 else "")
            # + "_noLenNorm"
            + f"_beam{beam_size}"
        )
        search_args = {
            "beam_size": beam_size,
            "att_scale": 0.0,
            "use_ctc": True,
            "ctc_scale": 1.0,
            "ctc_state_fix": True,
            "bsf": bsf,
            "prior_corr": prior_scale != 0.0,
            "ctc_prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-02-22--conformer-swb/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.ZeflcEHlQTjn/output/prior.txt",
            "prior_scale": prior_scale,
            # "length_normalization_exponent": 0.0,
        }
        res, _ = recog_model(
            task,
            model_with_checkpoint,
            model_recog,
            dev_sets=None,
            model_args=model_args,
            search_args=search_args,
            prefix_name=name,
        )
        tk.register_output(
            name + f"/recog_results",
            res.output,
        )

    # att + espnet ctc prefix
    # beam 32: {"dev-clean": 2.14, "dev-other": 5.21, "test-clean": 2.43, "test-other": 5.57}
    for scales, prior_scale, beam_size in product([(0.7, 0.3)], [0.1], []):
        att_scale, ctc_scale = scales

        name = (
            prefix_name
            + f"/opls_att{att_scale}_ctc{ctc_scale}_fix"
            + (f"_prior{prior_scale}" if prior_scale != 0.0 else "")
            + "_noLenNorm"
            + f"_beam{beam_size}"
        )
        search_args = {
            "beam_size": beam_size,
            "att_scale": att_scale,
            "use_ctc": True,
            "ctc_scale": ctc_scale,
            "ctc_state_fix": True,
            "bsf": bsf,
            "prior_corr": prior_scale != 0.0,
            "ctc_prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-02-22--conformer-swb/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.ZeflcEHlQTjn/output/prior.txt",
            "prior_scale": prior_scale,
            "length_normalization_exponent": 0.0,
        }
        res, _ = recog_model(
            task,
            model_with_checkpoint,
            model_recog,
            dev_sets=None,
            model_args=model_args,
            search_args=search_args,
            prefix_name=name,
        )
        tk.register_output(
            name + f"/recog_results",
            res.output,
        )

    # ctc only decoding
    # prior 0.0: {"dev-clean": 2.85, "dev-other": 6.68, "test-clean": 3.09, "test-other": 7.0}
    for prior_scale in []:
        search_args = {
            "bsf": bsf,
            "prior_corr": prior_scale > 0.0,
            "ctc_prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-02-22--conformer-swb/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.ZeflcEHlQTjn/output/prior.txt",
            "prior_scale": prior_scale,
        }
        name = (
            prefix_name
            + f"/ctc_greedy"
            + (f"_prior{prior_scale}" if prior_scale > 0.0 else "")
        )
        res, _ = recog_model(
            task,
            model_with_checkpoint,
            model_recog_ctc,
            dev_sets=None,
            model_args=model_args,
            search_args=search_args,
            prefix_name=name,
        )
        tk.register_output(
            name + f"/recog_results",
            res.output,
        )

    # optsr att + ctc
    # beam 32: {"dev-clean": 2.2, "dev-other": 5.33, "test-clean": 2.44, "test-other": 5.61}
    for scales, prior_scale, beam_size in product([(0.8, 0.2)], [0.1], []):
        att_scale, ctc_scale = scales
        name = (
            prefix_name_40
            + f"/optsr_att{att_scale}_ctc{ctc_scale}"
            + (f"_prior{prior_scale}" if prior_scale != 0.0 else "")
            + f"_beam{beam_size}_eos_end"
        )
        search_args = {
            "beam_size": beam_size,
            "att_scale": att_scale,
            "ctc_scale": ctc_scale,
            "bsf": 40,
            "mask_eos": True,
            "ctc_prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-02-22--conformer-swb/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.ZeflcEHlQTjn/output/prior.txt",
            "prior_scale": prior_scale,
            "hash_overwrite": "problem ctc log",
            "add_eos_to_end": True,
        }

        recog_res, recog_out = recog_model(
            task,
            model_with_checkpoint,
            model_recog_time_sync,
            # dev_sets=["dev-other"],
            dev_sets=None,
            model_args=model_args,
            search_args=search_args,
            prefix_name=name,
        )
        tk.register_output(
            name + f"/recog_results",
            recog_res.output,
        )

    # ------------------ with LSTM LM ------------------------

    # att + lstm lm TODO: debug difference
    for scales, beam_size in product([(1.0, 0.3), (1.0, 0.33), (1.0, 0.27)], []):
        att_scale, lm_scale = scales
        recog_name = f"/opls_att{att_scale}_lstm_lm{lm_scale}_beam{beam_size}"
        name = prefix_name + recog_name
        search_args = {
            "beam_size": beam_size,
            "add_lstm_lm": True,
            "lm_scale": lm_scale,
            "bsf": bsf,
        }
        res, _ = recog_model(
            task,
            model_with_checkpoint,
            model_recog,
            dev_sets=["dev-other"],
            model_args=model_args,
            search_args=search_args,
            prefix_name=name,
        )
        tk.register_output(
            name + f"/recog_results",
            res.output,
        )

    # att + espnet ctc prefix scorer + lstm lm
    for scales, prior_scale, lm_scale, beam_size in product(
        [(0.8, 0.2), (0.85, 0.15)],
        [0.0],
        [0.4, 0.45, 0.5, 0.55, 0.6, 0.65],
        [],
    ):
        att_scale, ctc_scale = scales
        recog_name = (
            f"/opls_att{att_scale}_ctc{ctc_scale}_fix"
            + (f"_prior{prior_scale}" if prior_scale > 0.0 else "")
            + f"_lstm_lm{lm_scale}_beam{beam_size}"
        )
        name = prefix_name + recog_name
        search_args = {
            "beam_size": beam_size,
            "add_lstm_lm": True,
            "lm_scale": lm_scale,
            "att_scale": att_scale,
            "ctc_scale": ctc_scale,
            "use_ctc": True,
            "bsf": bsf,
            "prior_corr": prior_scale > 0.0,
            "ctc_prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-02-22--conformer-swb/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.ZeflcEHlQTjn/output/prior.txt",
            "prior_scale": prior_scale,
            "ctc_state_fix": True,
        }
        res, _ = recog_model(
            task,
            model_with_checkpoint,
            model_recog,
            dev_sets=["dev-other"],
            model_args=model_args,
            search_args=search_args,
            prefix_name=name,
        )
        tk.register_output(
            name + f"/recog_results",
            res.output,
        )

    # ------------ Search Errors ------------

    # check for search errors
    for scales in [(0.7, 0.3)]:
        for beam_size in []:
            att_scale, ctc_scale = scales
            name = (
                prefix_name
                + f"/bsf40_espnet_att{att_scale}_ctc{ctc_scale}_beam{beam_size}"
            )
            search_args = {
                "beam_size": beam_size,
                # att decoder args
                "att_scale": att_scale,
                "ctc_scale": ctc_scale,
                "use_ctc": True,
                "mask_eos": True,
            }
            dev_sets = ["dev-other"]  # only dev-other for testing
            # dev_sets = None  # all

            # first recog
            recog_res, recog_out = recog_model(
                task,
                model_with_checkpoint,
                model_recog,
                dev_sets=dev_sets,
                model_args=model_args,
                search_args=search_args,
                prefix_name=name,
            )
            tk.register_output(
                name + f"/recog_results",
                recog_res.output,
            )

            # then forward
            forward_out = forward_model(
                task,
                model_with_checkpoint,
                model_forward,
                dev_sets=dev_sets,
                model_args=model_args,
                search_args=search_args,
                prefix_name=prefix_name,
            )

            res = ComputeSearchErrorsJob(
                forward_out.output, recog_out.output
            ).out_search_errors
            tk.register_output(
                name + f"/search_errors",
                res,
            )

    # opts att + ctc TODO: fix bugs
    for scales, blank_scale, beam_size in product([(0.65, 0.35)], [2.0], []):
        att_scale, ctc_scale = scales
        name = (
            prefix_name
            + f"/opts_att{att_scale}_ctc{ctc_scale}"
            + (f"_blank{blank_scale}" if blank_scale != 0.0 else "")
            + f"_beam{beam_size}"
        )
        search_args = {
            "beam_size": beam_size,
            "att_scale": att_scale,
            "ctc_scale": ctc_scale,
            "blank_scale": blank_scale,
            "bsf": bsf,
        }

        #
        # if prior_scale != 0.0:
        #     search_args.update(
        #         {
        #             "prior_corr": True,
        #             "ctc_prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-02-22--conformer-swb/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.ZeflcEHlQTjn/output/prior.txt",
        #             "prior_scale": prior_scale,
        #         }
        #     )

        # first recog
        recog_res, recog_out = recog_model(
            task,
            model_with_checkpoint,
            model_recog_time_sync,
            dev_sets=["dev-other"],
            model_args=model_args,
            search_args=search_args,
            prefix_name=name,
        )
        tk.register_output(
            name + f"/recog_results",
            recog_res.output,
        )

        compute_search_errors = False
        if compute_search_errors:
            # then forward
            forward_out = forward_model(
                task,
                model_with_checkpoint,
                model_forward_time_sync,
                dev_sets=dev_sets,
                model_args=model_args,
                search_args=search_args,
                prefix_name=name,
            )

            res = ComputeSearchErrorsJob(
                forward_out.output, recog_out.output
            ).out_search_errors
            tk.register_output(
                name + f"/search_errors",
                res,
            )

    #  ------------------ with Trafo LM ------------------------

    model_ckpt_path = tk.Path(
        _torch_ckpt_filename_base_model, hash_overwrite="torch_ckpt_base_model"
    )
    model_ckpt = PtCheckpoint(model_ckpt_path)
    model_with_checkpoint = ModelWithCheckpoint(
        definition=from_scratch_model_def, checkpoint=model_ckpt
    )

    recog_config = {
        "model_args": {
            "external_language_model": {
                "class": "Trafo_LM_Model",
                "num_layers": 24,
                "layer_out_dim": 1024,
                "att_num_heads": 8,
                "use_pos_enc": True,
                "ff_activation": "relu",
                "pos_enc_diff_pos": True,
            },
        },
        "preload_from_files": {
            "01_trafo_lm": {
                "prefix": "language_model.",
                "filename": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/librispeech/trafo_lm_only_24_02_06/network.023.pt",
            }
        }
    }

    with_lm_name = "/with_lm"

    # ilm ckpt torch: /work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/librispeech/mini_att_ilm_24_05_28/average.pt

    # optsr ctc + trafo lm
    for lm_scale, prior_scale,  beam_size in product([0.75], [0.4], []):
        name = (
            prefix_name_40
            + with_lm_name
            + f"/optsr_ctc_trafo_lm{lm_scale}"
            + (f"_prior{prior_scale}" if prior_scale > 0.0 else "")
            + f"_beam{beam_size}_w_trafo_eos"
        )
        search_args = {
            "beam_size": beam_size,
            "att_scale": 0.0,
            "ctc_scale": 1.0,
            "add_trafo_lm": True,
            "add_eos_to_end": True,
            "lm_scale": lm_scale,
            "lm_skip": True,
            "bsf": 40,
            "ctc_prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-02-22--conformer-swb/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.ZeflcEHlQTjn/output/prior.txt",
            "prior_scale": prior_scale,
            "hash_overwrite": "problem_ctc_log",
        }
        recog_config["search_args"] = search_args

        recog_res = recog_model(
            task,
            model_with_checkpoint,
            model_recog_time_sync,
            dev_sets=None,
            # dev_sets=["dev-other"],
            config=recog_config,
            prefix_name=name,
        )
        tk.register_output(
            name + f"/recog_results",
            recog_res.output,
        )

    # att + trafo lm
    # beam 32: {"dev-clean": 1.91, "dev-other": 4.14, "test-clean": 2.2, "test-other": 4.6}
    for lm_scale, beam_size in product([0.42], [32]): # 12, 18
        recog_name = f"/att_trafo_lm{lm_scale}_beam{beam_size}"
        name = prefix_name_40 + with_lm_name + recog_name
        search_args = {
            "beam_size": beam_size,
            "add_trafo_lm": True,
            "lm_scale": lm_scale,
            "bsf": 40,
        }
        recog_config["search_args"] = search_args
        res = recog_model(
            task,
            model_with_checkpoint,
            model_recog,
            dev_sets=None,
            config=recog_config,
            name=name,
            compute_search_errors=True,
        )
        tk.register_output(
            name + f"/recog_results",
            res.output,
        )

    # opls ctc + trafo lm
    # beam 32: {"dev-clean": 1.95, "dev-other": 4.39, "test-clean": 2.21, "test-other": 4.78}
    for ctc_scale, prior_scale, lm_scale, beam_size in product(
        [1.0], [0.0], [0.65], []
    ):
        recog_name = (
            f"/opls_ctc{ctc_scale}_fix"
            + (f"_prior{prior_scale}" if prior_scale > 0.0 else "")
            + f"_trafo_lm{lm_scale}_beam{beam_size}"
        )
        name = prefix_name + with_lm_name + recog_name
        search_args = {
            "beam_size": beam_size,
            "add_trafo_lm": True,
            "lm_scale": lm_scale,
            "att_scale": 0.0,
            "ctc_scale": ctc_scale,
            "use_ctc": True,
            "bsf": bsf,
            "prior_corr": prior_scale > 0.0,
            "ctc_prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-02-22--conformer-swb/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.ZeflcEHlQTjn/output/prior.txt",
            "prior_scale": prior_scale,
        }
        res, _ = recog_model(
            task,
            model_with_checkpoint,
            model_recog,
            dev_sets=None,  # None for all
            model_args=model_args,
            search_args=search_args,
            prefix_name=name,
        )
        tk.register_output(
            name + f"/recog_results",
            res.output,
        )

    # opls att + ctc + trafo lm
    # beam 32: {"dev-clean": 1.79, "dev-other": 3.94, "test-clean": 2.03, "test-other": 4.36}
    for scales, prior_scale, lm_scale, beam_size in product(
        [(0.85, 0.15)], [0.0], [0.5], []
    ):
        att_scale, ctc_scale = scales
        recog_name = (
            f"/opls_att{att_scale}_ctc{ctc_scale}_fix"
            + (f"_prior{prior_scale}" if prior_scale > 0.0 else "")
            + f"_trafo_lm{lm_scale}_beam{beam_size}_cpu"
        )
        name = prefix_name + with_lm_name + recog_name
        search_args = {
            "beam_size": beam_size,
            "add_trafo_lm": True,
            "lm_scale": lm_scale,
            "att_scale": att_scale,
            "ctc_scale": ctc_scale,
            "use_ctc": True,
            "bsf": bsf,
            "prior_corr": prior_scale > 0.0,
            "ctc_prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-02-22--conformer-swb/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.ZeflcEHlQTjn/output/prior.txt",
            "prior_scale": prior_scale,
        }
        res, _ = recog_model(
            task,
            model_with_checkpoint,
            model_recog,
            dev_sets=["dev-other"],
            model_args=model_args,
            search_args=search_args,
            prefix_name=name,
            device="cpu",
            search_mem_rqmt=15,
        )
        tk.register_output(
            name + f"/recog_results",
            res.output,
        )

    # optsr att + ctc + trafo lm
    # beam 32: {"dev-clean": 1.81, "dev-other": 4.03, "test-clean": 2.02, "test-other": 4.53}
    for scales, beam_size in product([(0.8, 0.2, 0.2, 0.5)], []): # did not try larger beam
        att_scale, ctc_scale, prior_scale, lm_scale = scales
        name = (
            prefix_name_40
            + with_lm_name
            + f"/optsr_att{att_scale}_ctc{ctc_scale}"
            + (f"_trafo_lm{lm_scale}" if lm_scale > 0.0 else "")
            + (f"_prior{prior_scale}" if prior_scale != 0.0 else "")
            + f"_beam{beam_size}_fix2"
        )
        search_args = {
            "beam_size": beam_size,
            "att_scale": att_scale,
            "ctc_scale": ctc_scale,
            "bsf": 40,
            "mask_eos": True,
            "ctc_prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-02-22--conformer-swb/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.ZeflcEHlQTjn/output/prior.txt",
            "prior_scale": prior_scale,
            "add_trafo_lm": lm_scale > 0.0,
            "lm_scale": lm_scale,
            "lm_skip": True,
            "hash_overwrite": "problem_ctc_log",
            "length_normalization_exponent": 0.0,
            "add_eos_to_end": True,
        }

        recog_res, recog_out = recog_model(
            task,
            model_with_checkpoint,
            model_recog_time_sync,
            # dev_sets=["dev-other"],
            dev_sets=None,
            model_args=model_args,
            search_args=search_args,
            prefix_name=name,
            search_rqmt={"time": 8}
        )
        tk.register_output(
            name + f"/recog_results",
            recog_res.output,
        )


    # ------------------ with Trafo LM + MiniAtt ILM ------------------------

    recog_config = {
        "model_args": {
            "external_language_model": {
                "class": "Trafo_LM_Model",
                "num_layers": 24,
                "layer_out_dim": 1024,
                "att_num_heads": 8,
                "use_pos_enc": True,
                "ff_activation": "relu",
                "pos_enc_diff_pos": True,
            },
            "internal_language_model": {
                "class": "MiniAtt_ILM_Model",
                "s_use_zoneout_output": False,
            },
        },
        "preload_from_files": {
            "01_trafo_lm": {
                "prefix": "language_model.",
                "filename": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/librispeech/trafo_lm_only_24_02_06/network.023.pt",
            },
            "01_mini_att_ilm": {
                "prefix": "ilm.",
                "filename": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/librispeech/mini_att_ilm_24_05_28/average.pt",
            },
        }
    }


    with_lm_ilm_name = "/with_lm_ilm"

    # ilm ckpt torch: /work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/librispeech/mini_att_ilm_24_05_28/average.pt

    # att + trafo lm + ilm old
    # beam 32: {"dev-clean": 1.78, "dev-other": 3.66, "test-clean": 1.99, "test-other": 4.19}
    for lm_scale, ilm_scale, beam_size in product([0.54], [0.4], [32]):
        recog_name = f"/att_trafolm{lm_scale}_ilm{ilm_scale}_beam{beam_size}_ffix_old"
        name = prefix_name + with_lm_ilm_name + recog_name
        search_args = {
            "beam_size": beam_size,
            "add_trafo_lm": True,
            "lm_scale": lm_scale,
            "ilm_scale": ilm_scale,
            "bsf": bsf,
            "use_lm_first_label": True,
            "hash_overwrite": 1,
        }
        model_args = {
            "external_language_model": {
                "class": "Trafo_LM_Model",
                "num_layers": 24,
                "layer_out_dim": 1024,
                "att_num_heads": 8,
                "use_pos_enc": True,
                "ff_activation": "relu",
                "pos_enc_diff_pos": True,
            },
            "internal_language_model": {
                "class": "MiniAtt_ILM_Model",
                "s_use_zoneout_output": False,
            },
            "preload_from_files": {
                "01_trafo_lm": {
                    "prefix": "language_model.",
                    "filename": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/librispeech/trafo_lm_only_24_02_06/network.023.pt",
                },
                "01_mini_att_ilm": {
                    "prefix": "ilm.",
                    "filename": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/librispeech/mini_att_ilm_24_05_28/average.pt",
                },
            },
        }
        res, _ = recog_model_old(
            task,
            model_with_checkpoint,
            model_recog,
            dev_sets=["dev-other"],
            model_args=model_args,
            search_args=search_args,
            prefix_name=name,
        )
        tk.register_output(
            name + f"/recog_results",
            res.output,
        )

    # att + trafo lm + ilm
    # beam 32: {"dev-clean": 1.77, "dev-other": 3.74, "test-clean": 1.99, "test-other": 4.22}
    # Slightly different results as with old rf setup.
    # Most probably due to the use of different data file (see vimdiff of configs)
    for lm_scale, ilm_scale, beam_size in product([0.54], [0.4], [1,2,4,8,16,32]):
        recog_name = f"/att_trafolm{lm_scale}_ilm{ilm_scale}_beam{beam_size}"
        name = prefix_name + with_lm_ilm_name + recog_name
        search_args = {
            "beam_size": beam_size,
            "add_trafo_lm": True,
            "lm_scale": lm_scale,
            "ilm_scale": ilm_scale,
            "bsf": bsf,
            "use_lm_first_label": True,
        }
        recog_config["search_args"] = search_args

        res = recog_model(
            task,
            model_with_checkpoint,
            model_recog,
            dev_sets=["dev-clean", "dev-other", "test-clean", "test-other"],
            config=recog_config,
            name=name,
            search_mem_rqmt=6 if beam_size <= 16 else 24,
            compute_search_errors=True,
        )
        tk.register_output(
            name + f"/recog_results",
            res.output,
        )

    # opls att + ctc + trafo lm + ilm
    # beam 32 {"dev-clean": 1.71, "dev-other": 3.58, "test-clean": 1.94, "test-other": 4.11}
    for scales, prior_scale, lm_scale, ilm_scale, beam_size in product(
        [(0.8, 0.2)], [0.05, 0.07], [0.65], [0.4], [32]
    ):
        att_scale, ctc_scale = scales
        recog_name = (
            f"/opls_att{att_scale}_ctc{ctc_scale}"
            + (f"_prior{prior_scale}" if prior_scale > 0.0 else "")
            + f"_trafo_lm{lm_scale}"
            + f"_ilm{ilm_scale}"
            + f"_beam{beam_size}"
        )
        name = prefix_name + with_lm_ilm_name + recog_name
        search_args = {
            "beam_size": beam_size,
            "add_trafo_lm": True,
            "lm_scale": lm_scale,
            "att_scale": att_scale,
            "ctc_scale": ctc_scale,
            "ilm_scale": ilm_scale,
            "use_ctc": True,
            "bsf": bsf,
            "prior_corr": prior_scale > 0.0,
            "ctc_prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-02-22--conformer-swb/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.ZeflcEHlQTjn/output/prior.txt",
            "prior_scale": prior_scale,
            "use_lm_first_label": True,
        }
        recog_config["search_args"] = search_args
        res = recog_model(
            task,
            model_with_checkpoint,
            model_recog,
            dev_sets=["dev-clean", "dev-other", "test-clean", "test-other"],
            config=recog_config,
            name=name,
            compute_search_errors=True,
            # device="cpu",
            # search_mem_rqmt=15,
        )
        tk.register_output(
            name + f"/recog_results",
            res.output,
        )

    # two pass rescoring att + ctc + trafo lm + ilm
    for scales, prior_scale, lm_scale, ilm_scale, beam_size in product(
        [(1.0, 0.0009), (1.0 ,0.001)],
        [0.0], [0.54], [0.4], [] # 12, 32, 40, 64, 80
    ):
        att_scale, ctc_scale = scales
        recog_name = (
            f"/two_pass_att{att_scale}_ctc{ctc_scale}"
            + (f"_prior{prior_scale}" if prior_scale > 0.0 else "")
            + f"_trafo_lm{lm_scale}"
            + (f"_ilm{ilm_scale}" if ilm_scale > 0.0 else "")
            + f"_beam{beam_size}"
        )
        name = prefix_name + with_lm_name + recog_name
        search_args = {
            "beam_size": beam_size,
            "add_trafo_lm": True,
            "lm_scale": lm_scale,
            "att_scale": 1.0,
            # "ctc_scale": ctc_scale,
            "ilm_scale": ilm_scale,
            # "use_ctc": True,
            "bsf": bsf,
            "prior_corr": prior_scale > 0.0,
            "ctc_prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-02-22--conformer-swb/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.ZeflcEHlQTjn/output/prior.txt",
            "prior_scale": prior_scale,
            "rescore_w_ctc": True,
            "rescore_att_scale": att_scale,
            "rescore_ctc_scale": ctc_scale,
            "hash_overwrite": "fix",
        }
        res, _ = recog_model(
            task,
            model_with_checkpoint,
            model_recog,
            # dev_sets=["dev-other"],
            dev_sets=["dev-clean", "dev-other", "test-clean", "test-other"],
            model_args=model_args,
            search_args=search_args,
            prefix_name=name,
            # device="cpu",
            # search_mem_rqmt=15,
        )
        tk.register_output(
            name + f"/recog_results",
            res.output,
        )

    # optsr att + ctc + trafo  + ilm
    for scales, prior_scale, lm_scale, ilm_scale, beam_size in product([(0.8, 0.2)], [0.2], [0.8], [0.42], [32]): #lm 0.7, 0.8 # 32
        att_scale, ctc_scale = scales
        name = (
            prefix_name_40
            + with_lm_ilm_name
            + f"/optsr_att{att_scale}_ctc{ctc_scale}"
            + (f"_trafo_lm{lm_scale}" if lm_scale > 0.0 else "")
            + (f"_prior{prior_scale}" if prior_scale != 0.0 else "")
            + (f"_ilm{ilm_scale}" if ilm_scale > 0.0 else "")
            + f"_beam{beam_size}"
        )
        search_args = {
            "beam_size": beam_size,
            "att_scale": att_scale,
            "ctc_scale": ctc_scale,
            "bsf": 40,
            "mask_eos": True,
            "ctc_prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-02-22--conformer-swb/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.ZeflcEHlQTjn/output/prior.txt",
            "prior_scale": prior_scale,
            "add_trafo_lm": lm_scale > 0.0,
            "lm_scale": lm_scale,
            "lm_skip": True,
            "hash_overwrite": "fix forward 2",
            "length_normalization_exponent": 0.0,
            "add_eos_to_end": True,
            "ilm_scale": ilm_scale,
        }
        recog_config["search_args"] = search_args

        recog_res = recog_model(
            task,
            model_with_checkpoint,
            model_recog_time_sync,
            dev_sets=None,
            # dev_sets=["dev-other"],
            config=recog_config,
            name=name,
            search_rqmt={"time": 8},
            compute_search_errors=True,
            forward_def=model_forward_ctc_max,
        )
        tk.register_output(
            name + f"/recog_results",
            recog_res.output,
        )

    # -------------- LSTM LM + Mini ILM ------------------------------

    model_args = {
        "external_language_model": {
            "class": "LSTM_LM_Model",
        },
        "internal_language_model": {
            "class": "MiniAtt_ILM_Model",
            "s_use_zoneout_output": False,
        },
        "preload_from_files": {
            "01_lstm_lm": {
                "prefix": "language_model.",
                "filename": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/librispeech/lstm_lm_only_24_05_31/network.035.pt",
            },
            "01_mini_att_ilm": {
                "prefix": "ilm.",
                "filename": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/librispeech/mini_att_ilm_24_05_28/average.pt",
            },
        },
    }

    # ilm ckpt torch: /work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/librispeech/mini_att_ilm_24_05_28/average.pt

    # att + lstm lm + ilm
    #
    for lm_scale, ilm_scale, beam_size in product([0.33], [0.4], []): # 24
        recog_name = f"/att_lstmlm{lm_scale}_ilm{ilm_scale}_beam{beam_size}"
        name = prefix_name + with_lm_ilm_name + recog_name
        search_args = {
            "beam_size": beam_size,
            "lm_scale": lm_scale,
            "ilm_scale": ilm_scale,
            "bsf": bsf,
            "use_lm_first_label": True,
        }
        res, _ = recog_model(
            task,
            model_with_checkpoint,
            model_recog,
            dev_sets=["dev-other"],
            model_args=model_args,
            search_args=search_args,
            prefix_name=name,
        )
        tk.register_output(
            name + f"/recog_results",
            res.output,
        )

    # opls att + ctc + lstm lm + ilm
    for scales, prior_scale, lm_scale, ilm_scale, beam_size in product(
        [(0.9, 0.1), (0.8, 0.2), (0.7, 0.3)], [0.05], [0.4, 0.5, 0.65], [0.4], [] # 32
    ):
        att_scale, ctc_scale = scales
        recog_name = (
            f"/opls_att{att_scale}_ctc{ctc_scale}"
            + (f"_prior{prior_scale}" if prior_scale > 0.0 else "")
            + f"_lstmlm{lm_scale}"
            + f"_ilm{ilm_scale}"
            + f"_beam{beam_size}"
        )
        name = prefix_name + with_lm_ilm_name + recog_name
        search_args = {
            "beam_size": beam_size,
            "lm_scale": lm_scale,
            "att_scale": att_scale,
            "ctc_scale": ctc_scale,
            "ilm_scale": ilm_scale,
            "use_ctc": True,
            "bsf": bsf,
            "prior_corr": prior_scale > 0.0,
            "ctc_prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-02-22--conformer-swb/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.ZeflcEHlQTjn/output/prior.txt",
            "prior_scale": prior_scale,
            "use_lm_first_label": True,
        }
        res, _ = recog_model(
            task,
            model_with_checkpoint,
            model_recog,
            dev_sets=["dev-other"],
            model_args=model_args,
            search_args=search_args,
            prefix_name=name,
            # device="cpu",
            # search_mem_rqmt=15,
        )
        tk.register_output(
            name + f"/recog_results",
            res.output,
        )

    # ctc bs att + ctc + lstm lm + ilm
    for scales, prior_scale, lm_scale, ilm_scale, beam_size in product(
        [(0.6, 0.4)], [0.3], [0.35, 0.38, 0.4, 0.42, 0.45, 0.5, 0.6, 0.65], [0.0], []
    ):
        att_scale, ctc_scale = scales
        recog_name = (
            f"/optsbs_att{att_scale}_ctc{ctc_scale}"
            + (f"_prior{prior_scale}" if prior_scale > 0.0 else "")
            + (f"_lstmlm{lm_scale}" if lm_scale > 0.0 else "")
            + (f"_ilm{ilm_scale}" if ilm_scale > 0.0 else "")
            + f"_beam{beam_size}"
        )
        name = single_seq_prefix_name + (with_lm_ilm_name if ilm_scale > 0.0 else with_lm_name) + recog_name
        search_args = {
            "beam_size": beam_size,
            "lm_scale": lm_scale,
            "att_scale": att_scale,
            "ctc_scale": ctc_scale,
            "ilm_scale": ilm_scale,
            "use_ctc": True,
            "max_seq": 1,
            "prior_corr": prior_scale > 0.0,
            "ctc_prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-02-22--conformer-swb/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.ZeflcEHlQTjn/output/prior.txt",
            "prior_scale": prior_scale,
            "use_lm_first_label": True,
            "hash_overwrite": 2,
        }
        res, _ = recog_model(
            task,
            model_with_checkpoint,
            model_recog_ts_espnet,
            dev_sets=["dev-other"],
            # dev_sets=None,
            model_args=model_args,
            search_args=search_args,
            prefix_name=name,
            # device="cpu",
            # search_rqmt={"time": 24}
            # search_mem_rqmt=15,
        )
        tk.register_output(
            name + f"/recog_results",
            res.output,
        )




py = sis_run_with_prefix  # if run directly via `sis m ...`


def sis_run_dump_scores(prefix_name: str = None):
    """run the exp"""
    from i6_experiments.users.zeyer.utils.generic_job_output import generic_job_output
    from ._moh_att_2023_06_30_import import map_param_func_v3
    from .sis_setup import get_prefix_for_config
    from i6_core.returnn.training import Checkpoint as TfCheckpoint, PtCheckpoint
    from i6_experiments.users.zeyer.model_interfaces import ModelWithCheckpoint
    from i6_experiments.users.gaudino.dump import recog_model_dump
    from i6_experiments.users.zeyer.returnn.convert_ckpt_rf import (
        ConvertTfCheckpointToRfPtJob,
    )
    from i6_experiments.users.zeyer.datasets.librispeech import (
        get_librispeech_task_bpe10k_raw,
    )

    if not prefix_name:
        prefix_name = get_prefix_for_config(__file__)

    task = get_librispeech_task_bpe10k_raw(with_eos_postfix=True)

    extern_data_dict = task.train_dataset.get_extern_data()
    default_target_key = task.train_dataset.get_default_target()
    targets = Tensor(name=default_target_key, **extern_data_dict[default_target_key])
    target_dim = targets.feature_dim_or_sparse_dim

    new_chkpt_path = ConvertTfCheckpointToRfPtJob(
        checkpoint=TfCheckpoint(
            index_path=generic_job_output(_returnn_tf_ckpt_filename)
        ),
        make_model_func=MakeModel(
            in_dim=_log_mel_feature_dim,
            target_dim=target_dim.dimension,
            eos_label=_get_eos_idx(target_dim),
        ),
        map_func=map_param_func_v3,
    ).out_checkpoint

    # att + ctc decoding
    search_args = {
        "beam_size": 12,
        # att decoder args
        "att_scale": 1.0,
        "ctc_scale": 1.0,
        "use_ctc": False,
        "mask_eos": True,
        "add_lstm_lm": False,
        "prior_corr": False,
        "prior_scale": 0.2,
        "length_normalization_exponent": 1.0,  # 0.0 for disabled
        # "window_margin": 10,
        "rescore_w_ctc": False,
        "dump_ctc": True,
    }

    # new_chkpt_path = tk.Path(_torch_ckpt_filename_w_ctc, hash_overwrite="torch_ckpt_w_ctc")
    new_chkpt = PtCheckpoint(new_chkpt_path)
    model_with_checkpoint = ModelWithCheckpoint(
        definition=from_scratch_model_def, checkpoint=new_chkpt
    )

    dev_sets = ["dev-other"]  # only dev-other for testing
    # dev_sets = None  # all
    res = recog_model_dump(
        task,
        model_with_checkpoint,
        model_recog_dump,
        dev_sets=dev_sets,
        search_args=search_args,
    )
    tk.register_output(
        prefix_name
        # + f"/espnet_att{search_args['att_scale']}_ctc{search_args['ctc_scale']}_beam{search_args['beam_size']}_maskEos"
        + f"/dump_ctc_scores" + f"/scores",
        res.output,
    )


class MakeModel:
    """for import"""

    def __init__(
        self,
        in_dim: int,
        target_dim: int,
        *,
        eos_label: int = 0,
        num_enc_layers: int = 12,
        model_args: Optional[Dict[str, Any]] = {},
        search_args: Optional[Dict[str, Any]] = {},
    ):
        self.in_dim = in_dim
        self.target_dim = target_dim
        self.eos_label = eos_label
        self.num_enc_layers = num_enc_layers
        self.search_args = search_args
        self.model_args = model_args

    def __call__(self) -> Model:
        from returnn.datasets.util.vocabulary import Vocabulary

        in_dim = Dim(name="in", dimension=self.in_dim, kind=Dim.Types.Feature)
        target_dim = Dim(
            name="target", dimension=self.target_dim, kind=Dim.Types.Feature
        )
        target_dim.vocab = Vocabulary.create_vocab_from_labels(
            [str(i) for i in range(target_dim.dimension)], eos_label=self.eos_label
        )

        return self.make_model(
            in_dim,
            target_dim,
            num_enc_layers=self.num_enc_layers,
            model_args=self.model_args,
            search_args=self.search_args,
        )

    @classmethod
    def make_model(
        cls,
        in_dim: Dim,
        target_dim: Dim,
        *,
        model_args: Optional[Dict[str, Any]] = {},
        search_args: Optional[Dict[str, Any]] = {},
        num_enc_layers: int = 12,
        lm_opts: Optional[Dict[str, Any]] = None,
        ilm_opts: Optional[Dict[str, Any]] = None,
        **extra,
    ) -> Model:
        """make"""

        target_embed_dim = Dim(name="target_embed", dimension=model_args.get("target_embed_dim", 640))
        lm = None
        ilm = None
        if lm_opts:
            assert isinstance(lm_opts, dict)
            lm_opts = lm_opts.copy()
            cls_name = lm_opts.pop("class")
            assert cls_name == "Trafo_LM_Model" or cls_name == "LSTM_LM_Model"
            lm_opts.pop("vocab_dim", None)  # will just overwrite

            if cls_name == "Trafo_LM_Model":
                lm = Trafo_LM_Model(target_dim, target_dim, **lm_opts)

            elif cls_name == "LSTM_LM_Model":
                lm = LSTM_LM_Model(target_dim, target_dim, **lm_opts)

        if ilm_opts:
            assert isinstance(ilm_opts, dict)
            ilm_opts = ilm_opts.copy()
            cls_name = ilm_opts.pop("class")

            ilm = MiniAtt_ILM_Model(target_embed_dim, target_dim, **ilm_opts)

        # lm = (lm, functools.partial(trafo_lm.make_label_scorer_torch, model=lm))

        return Model(
            in_dim,
            num_enc_layers=num_enc_layers,
            enc_model_dim=Dim(name="enc", dimension=512, kind=Dim.Types.Feature),
            enc_ff_dim=Dim(name="enc-ff", dimension=2048, kind=Dim.Types.Feature),
            enc_att_num_heads=8,
            enc_conformer_layer_opts=dict(
                conv_norm_opts=dict(use_mask=True),
                self_att_opts=dict(
                    # Shawn et al 2018 style, old RETURNN way.
                    with_bias=False,
                    with_linear_pos=False,
                    with_pos_bias=False,
                    learnable_pos_emb=True,
                    separate_pos_emb_per_head=False,
                ),
                ff_activation=lambda x: rf.relu(x) ** 2.0,
            ),
            target_embed_dim=target_embed_dim,
            target_dim=target_dim,
            blank_idx=target_dim.dimension,
            bos_idx=_get_bos_idx(target_dim),
            eos_idx=_get_eos_idx(target_dim),
            model_args=model_args,
            search_args=search_args,
            language_model=lm,
            ilm=ilm,
            **extra,
        )

class SepCTCEncoder(rf.Module):

    def __init__(
        self,
        in_dim: Dim,
        *,
        num_enc_layers: int = 12,
        enc_model_dim: Dim = Dim(name="enc", dimension=512),
        enc_ff_dim: Dim = Dim(name="enc-ff", dimension=2048),
        enc_att_num_heads: int = 8,
        enc_conformer_layer_opts: Optional[Dict[str, Any]] = None,
        enc_dropout: float = 0.1,
        enc_att_dropout: float = 0.1,
        target_dim_w_blank: Dim,

    ):
        super(SepCTCEncoder, self).__init__()

        self.encoder = ConformerEncoder(
            in_dim,
            enc_model_dim,
            ff_dim=enc_ff_dim,
            input_layer=ConformerConvSubsample(
                in_dim,
                out_dims=[
                    Dim(32, name="conv1"),
                    Dim(64, name="conv2"),
                    Dim(64, name="conv3"),
                ],
                filter_sizes=[(3, 3), (3, 3), (3, 3)],
                pool_sizes=[(1, 2)],
                strides=[(1, 1), (3, 1), (2, 1)],
            ),
            encoder_layer_opts=enc_conformer_layer_opts,
            num_layers=num_enc_layers,
            num_heads=enc_att_num_heads,
            dropout=enc_dropout,
            att_dropout=enc_att_dropout,
        )
        self.ctc = rf.Linear(
            self.encoder.out_dim, target_dim_w_blank
        )


class Model(rf.Module):
    """Model definition"""

    def __init__(
        self,
        in_dim: Dim,
        *,
        num_enc_layers: int = 12,
        target_embed_dim: Dim,
        target_dim: Dim,
        blank_idx: int,
        eos_idx: int,
        bos_idx: int,
        enc_model_dim: Dim = Dim(name="enc", dimension=512),
        enc_ff_dim: Dim = Dim(name="enc-ff", dimension=2048),
        enc_att_num_heads: int = 4,
        enc_conformer_layer_opts: Optional[Dict[str, Any]] = None,
        enc_key_total_dim: Dim = Dim(name="enc_key_total_dim", dimension=1024),
        att_num_heads: Dim = Dim(name="att_num_heads", dimension=1),
        att_dropout: float = 0.1,
        enc_dropout: float = 0.1,
        enc_att_dropout: float = 0.1,
        l2: float = 0.0001,
        model_args: Optional[Dict[str, Any]] = None,
        search_args: Optional[Dict[str, Any]] = None,
        language_model: Optional[rf.Module] = None,
        ilm: Optional[rf.Module] = None,
    ):
        super(Model, self).__init__()
        self.in_dim = in_dim
        self.target_dim = target_dim
        self.target_embed_dim = target_embed_dim
        self.blank_idx = blank_idx
        self.eos_idx = eos_idx
        self.bos_idx = bos_idx  # for non-blank labels; for with-blank labels, we use bos_idx=blank_idx

        self.enc_key_total_dim = enc_key_total_dim
        self.enc_key_per_head_dim = enc_key_total_dim.div_left(att_num_heads)
        self.att_num_heads = att_num_heads
        self.att_dropout = att_dropout

        self.target_dim_w_blank = Dim(
            name="target_w_b",
            dimension=self.target_dim.dimension + 1,
            kind=Dim.Types.Feature,
        )

        self.mel_normalization = model_args.get("mel_normalization", False)
        self.no_ctc = model_args.get("no_ctc", False)
        self.enc_layer_w_ctc = model_args.get("enc_layer_w_ctc", None)
        self.s_use_zoneout_output = model_args.get("s_use_zoneout_output", False)

        self.encoder = ConformerEncoder(
            in_dim,
            enc_model_dim,
            ff_dim=enc_ff_dim,
            input_layer=ConformerConvSubsample(
                in_dim,
                out_dims=[
                    Dim(32, name="conv1"),
                    Dim(64, name="conv2"),
                    Dim(64, name="conv3"),
                ],
                filter_sizes=[(3, 3), (3, 3), (3, 3)],
                pool_sizes=[(1, 2)],
                strides=[(1, 1), (3, 1), (2, 1)],
            ),
            encoder_layer_opts=enc_conformer_layer_opts,
            num_layers=num_enc_layers,
            num_heads=enc_att_num_heads,
            dropout=enc_dropout,
            att_dropout=enc_att_dropout,
        )

        if model_args.get("encoder_ctc", False):
            self.sep_enc_ctc = SepCTCEncoder(
                in_dim,
                enc_model_dim=enc_model_dim,
                enc_ff_dim=enc_ff_dim,
                enc_att_num_heads=enc_att_num_heads,
                enc_conformer_layer_opts=enc_conformer_layer_opts,
                enc_dropout=enc_dropout,
                enc_att_dropout=enc_att_dropout,
                num_enc_layers=num_enc_layers,
                target_dim_w_blank=self.target_dim_w_blank,
            )

        # https://github.com/rwth-i6/returnn-experiments/blob/master/2020-rnn-transducer/configs/base2.conv2l.specaug4a.ctc.devtrain.config

        self.enc_ctx = rf.Linear(self.encoder.out_dim, enc_key_total_dim)
        self.enc_ctx_dropout = 0.2
        self.enc_win_dim = Dim(name="enc_win_dim", dimension=5)

        self.search_args = search_args
        if not self.no_ctc:
            self.ctc = rf.Linear(self.encoder.out_dim, self.target_dim_w_blank)

        self.language_model = None
        if language_model:
            self.language_model = language_model

        self.ilm = None
        if ilm:
            self.ilm = ilm

        # if model_args.get("add_lstm_lm", False):
        #     self.lstm_lm = LSTM_LM_Model(target_dim, target_dim)
        # if model_args.get("add_trafo_lm", False):
        #     self.trafo_lm = Trafo_LM_Model(
        #         target_dim, target_dim, **model_args.get("trafo_lm_args", {})
        #     )

        self.inv_fertility = rf.Linear(
            self.encoder.out_dim, att_num_heads, with_bias=False
        )

        self.target_embed = rf.Embedding(
            target_dim,
            target_embed_dim,
        )

        self.s = rf.ZoneoutLSTM(
            self.target_embed.out_dim + att_num_heads * self.encoder.out_dim,
            Dim(name="lstm", dimension=1024),
            zoneout_factor_cell=0.15,
            zoneout_factor_output=0.05,
            use_zoneout_output=self.s_use_zoneout_output,  # like RETURNN/TF ZoneoutLSTM old default
            # use_zoneout_output=False,  # like RETURNN/TF ZoneoutLSTM old default # this was a bug
            # parts_order="icfo",  # like RETURNN/TF ZoneoutLSTM
            # parts_order="ifco",
            parts_order="jifo",  # NativeLSTM (the code above converts it...)
            forget_bias=0.0,  # the code above already adds it during conversion
        )

        self.weight_feedback = rf.Linear(
            att_num_heads, enc_key_total_dim, with_bias=False
        )
        self.s_transformed = rf.Linear(
            self.s.out_dim, enc_key_total_dim, with_bias=False
        )
        self.energy = rf.Linear(enc_key_total_dim, att_num_heads, with_bias=False)
        self.readout_in = rf.Linear(
            self.s.out_dim
            + self.target_embed.out_dim
            + att_num_heads * self.encoder.out_dim,
            Dim(name="readout", dimension=1024),
        )
        self.output_prob = rf.Linear(self.readout_in.out_dim // 2, target_dim)

        for p in self.parameters():
            p.weight_decay = l2



    def encode(
        self,
        source: Tensor,
        *,
        in_spatial_dim: Dim,
        collected_outputs: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Dict[str, Tensor], Dim]:
        """encode, and extend the encoder output for things we need in the decoder"""
        if source.feature_dim:
            assert source.feature_dim.dimension == 1
            source = rf.squeeze(source, source.feature_dim)

        # log mel filterbank features
        source, in_spatial_dim, in_dim_ = rf.stft(
            source,
            in_spatial_dim=in_spatial_dim,
            frame_step=160,
            frame_length=400,
            fft_length=512,
        )
        source = rf.abs(source) ** 2.0
        source = rf.audio.mel_filterbank(
            source, in_dim=in_dim_, out_dim=self.in_dim, sampling_rate=16000
        )
        source = rf.safe_log(source, eps=1e-10) / 2.3026

        if self.mel_normalization:
            ted2_global_mean = rf.Tensor(name="ted2_global_mean",
                                         dims=[source.feature_dim],
                                         dtype=source.dtype,
                                         raw_tensor=torch.tensor(numpy.loadtxt('/u/zeineldeen/setups/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/returnn/dataset/ExtractDatasetMeanStddevJob.UHCZghp269OR/output/mean', dtype='float32')),
            )
            ted2_global_stddev = rf.Tensor(name="ted2_global_stddev",
                                         dims=[source.feature_dim],
                                         dtype=source.dtype,
                                         raw_tensor=torch.tensor(numpy.loadtxt('/u/zeineldeen/setups/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/returnn/dataset/ExtractDatasetMeanStddevJob.UHCZghp269OR/output/std_dev', dtype='float32')),
            )

            source = (source - rf.copy_to_device(ted2_global_mean)) / rf.copy_to_device(ted2_global_stddev)


        # TODO specaug
        # source = specaugment_wei(source, spatial_dim=in_spatial_dim, feature_dim=self.in_dim)  # TODO
        if self.enc_layer_w_ctc and not collected_outputs:
            collected_outputs = {}
        enc, enc_spatial_dim = self.encoder(
            source, in_spatial_dim=in_spatial_dim, collected_outputs=collected_outputs
        )
        enc_ctx = self.enc_ctx(enc)
        inv_fertility = rf.sigmoid(self.inv_fertility(enc))
        ctc = None
        if not self.no_ctc:
            if self.enc_layer_w_ctc:
                ctc = rf.softmax(self.ctc(collected_outputs[str(self.enc_layer_w_ctc - 1)]), axis=self.target_dim_w_blank)
            else:
                ctc = rf.softmax(self.ctc(enc), axis=self.target_dim_w_blank)
        return (
            dict(enc=enc, enc_ctx=enc_ctx, inv_fertility=inv_fertility, ctc=ctc),
            enc_spatial_dim,
        )

    def encode_ctc(
        self,
        source: Tensor,
        *,
        in_spatial_dim: Dim,
        collected_outputs: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Dict[str, Tensor], Dim]:
        """encode ctc"""
        assert self.sep_enc_ctc is not None, "sep_enc_ctc_encoder is None"
        if source.feature_dim:
            assert source.feature_dim.dimension == 1
            source = rf.squeeze(source, source.feature_dim)
        # log mel filterbank features
        source, in_spatial_dim, in_dim_ = rf.stft(
            source,
            in_spatial_dim=in_spatial_dim,
            frame_step=160,
            frame_length=400,
            fft_length=512,
        )
        source = rf.abs(source) ** 2.0
        source = rf.audio.mel_filterbank(
            source, in_dim=in_dim_, out_dim=self.in_dim, sampling_rate=16000
        )
        source = rf.safe_log(source, eps=1e-10) / 2.3026

        if self.mel_normalization:
            ted2_global_mean = rf.Tensor(name="ted2_global_mean",
                                         dims=[source.feature_dim],
                                         dtype=source.dtype,
                                         raw_tensor=torch.tensor(numpy.loadtxt('/u/zeineldeen/setups/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/returnn/dataset/ExtractDatasetMeanStddevJob.UHCZghp269OR/output/mean', dtype='float32')),
            )
            ted2_global_stddev = rf.Tensor(name="ted2_global_stddev",
                                         dims=[source.feature_dim],
                                         dtype=source.dtype,
                                         raw_tensor=torch.tensor(numpy.loadtxt('/u/zeineldeen/setups/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/returnn/dataset/ExtractDatasetMeanStddevJob.UHCZghp269OR/output/std_dev', dtype='float32')),
            )

            source = (source - rf.copy_to_device(ted2_global_mean)) / rf.copy_to_device(ted2_global_stddev)

        enc, enc_spatial_dim = self.sep_enc_ctc.encoder(
            source, in_spatial_dim=in_spatial_dim, collected_outputs=collected_outputs
        )

        ctc = rf.softmax(self.sep_enc_ctc.ctc(enc), axis=self.target_dim_w_blank)
        return (
            dict(enc=enc, ctc=ctc),
            enc_spatial_dim,
        )

    @staticmethod
    def encoder_unstack(ext: Dict[str, rf.Tensor]) -> Dict[str, rf.Tensor]:
        """
        prepare the encoder output for the loop (full-sum or time-sync)
        """
        # We might improve or generalize the interface later...
        # https://github.com/rwth-i6/returnn_common/issues/202
        loop = rf.inner_loop()
        return {k: loop.unstack(v) for k, v in ext.items()}

    def decoder_default_initial_state(
        self, *, batch_dims: Sequence[Dim], enc_spatial_dim: Dim
    ) -> rf.State:
        """Default initial state"""
        state = rf.State(
            s=self.s.default_initial_state(batch_dims=batch_dims),
            att=rf.zeros(
                list(batch_dims) + [self.att_num_heads * self.encoder.out_dim]
            ),
            accum_att_weights=rf.zeros(
                list(batch_dims) + [enc_spatial_dim, self.att_num_heads],
                feature_dim=self.att_num_heads,
            ),
        )
        state.att.feature_dim_axis = len(state.att.dims) - 1
        return state

    def loop_step_output_templates(self, batch_dims: List[Dim], enc_spatial_dim:Dim ) -> Dict[str, Tensor]:
        """loop step out"""
        return {
            "s": Tensor(
                "s",
                dims=batch_dims + [self.s.out_dim],
                dtype=rf.get_default_float_dtype(),
                feature_dim_axis=-1,
            ),
            "att": Tensor(
                "att",
                dims=batch_dims + [self.att_num_heads * self.encoder.out_dim],
                dtype=rf.get_default_float_dtype(),
                feature_dim_axis=-1,
            ),
            "att_weights": Tensor(
                "att_weights",
                dims=batch_dims + [enc_spatial_dim, self.att_num_heads],
                dtype=rf.get_default_float_dtype(),
                feature_dim_axis=-1,
            ),
        }

    def loop_step(
        self,
        *,
        enc: rf.Tensor,
        enc_ctx: rf.Tensor,
        inv_fertility: rf.Tensor,
        enc_spatial_dim: Dim,
        input_embed: rf.Tensor,
        state: Optional[rf.State] = None,
    ) -> Tuple[Dict[str, rf.Tensor], rf.State]:
        """step of the inner loop"""
        if state is None:
            batch_dims = enc.remaining_dims(
                remove=(enc.feature_dim, enc_spatial_dim)
                if enc_spatial_dim != single_step_dim
                else (enc.feature_dim,)
            )
            state = self.decoder_default_initial_state(
                batch_dims=batch_dims, enc_spatial_dim=enc_spatial_dim
            )
        state_ = rf.State()

        prev_att = state.att

        s, state_.s = self.s(
            rf.concat_features(input_embed, prev_att),
            state=state.s,
            spatial_dim=single_step_dim,
        )

        weight_feedback = self.weight_feedback(state.accum_att_weights)
        s_transformed = self.s_transformed(s)
        energy_in = enc_ctx + weight_feedback + s_transformed
        energy = self.energy(rf.tanh(energy_in))
        att_weights = rf.softmax(energy, axis=enc_spatial_dim)
        state_.accum_att_weights = (
            state.accum_att_weights + att_weights * inv_fertility * 0.5
        )
        att0 = rf.dot(att_weights, enc, reduce=enc_spatial_dim, use_mask=False)
        att0.feature_dim = self.encoder.out_dim
        att, _ = rf.merge_dims(att0, dims=(self.att_num_heads, self.encoder.out_dim))
        state_.att = att

        return {"s": s, "att": att, "att_weights": att_weights}, state_

    def decode_logits(self, *, s: Tensor, input_embed: Tensor, att: Tensor) -> Tensor:
        """logits for the decoder"""
        readout_in = self.readout_in(rf.concat_features(s, input_embed, att))
        readout = rf.reduce_out(
            readout_in, mode="max", num_pieces=2, out_dim=self.output_prob.in_dim
        )
        readout = rf.dropout(
            readout, drop_prob=0.3, axis=readout.feature_dim
        )  # why is this here?
        logits = self.output_prob(readout)
        return logits


def _get_bos_idx(target_dim: Dim) -> int:
    """for non-blank labels"""
    assert target_dim.vocab
    if target_dim.vocab.bos_label_id is not None:
        bos_idx = target_dim.vocab.bos_label_id
    elif target_dim.vocab.eos_label_id is not None:
        bos_idx = target_dim.vocab.eos_label_id
    elif "<sil>" in target_dim.vocab.user_defined_symbol_ids:
        bos_idx = target_dim.vocab.user_defined_symbol_ids["<sil>"]
    else:
        raise Exception(f"cannot determine bos_idx from vocab {target_dim.vocab}")
    return bos_idx


def _get_eos_idx(target_dim: Dim) -> int:
    """for non-blank labels"""
    assert target_dim.vocab
    if target_dim.vocab.eos_label_id is not None:
        eos_idx = target_dim.vocab.eos_label_id
    else:
        raise Exception(f"cannot determine eos_idx from vocab {target_dim.vocab}")
    return eos_idx


def from_scratch_model_def(
    *,
    epoch: int,
    in_dim: Dim,
    target_dim: Dim,
    model_args: Optional[Dict[str, Any]],
    search_args: Optional[Dict[str, Any]],
) -> Model:
    """Function is run within RETURNN."""
    in_dim, epoch  # noqa
    # real input is raw audio, internally it does logmel
    in_dim = Dim(name="logmel", dimension=_log_mel_feature_dim, kind=Dim.Types.Feature)
    lm_opts = model_args.get("external_language_model")
    ilm_opts = model_args.get("internal_language_model")
    return MakeModel.make_model(
        in_dim, target_dim, model_args=model_args, search_args=search_args, lm_opts=lm_opts, ilm_opts=ilm_opts
    )


from_scratch_model_def: ModelDef[Model]
from_scratch_model_def.behavior_version = 16
from_scratch_model_def.backend = "torch"
from_scratch_model_def.batch_size_factor = (  # bsf * 20000
    20  # change batch size here - 20 for att_window - 40 for ctc_prefix
)
from_scratch_model_def.max_seqs = 200  # 1


def from_scratch_training(
    *,
    model: Model,
    data: rf.Tensor,
    data_spatial_dim: Dim,
    targets: rf.Tensor,
    targets_spatial_dim: Dim,
):
    """Function is run within RETURNN."""
    assert not data.feature_dim  # raw samples
    enc_args, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim)

    enc_args.pop("ctc")

    batch_dims = data.remaining_dims(data_spatial_dim)
    input_embeddings = model.target_embed(targets)
    input_embeddings = rf.shift_right(
        input_embeddings, axis=targets_spatial_dim, pad_value=0.0
    )

    def _body(input_embed: Tensor, state: rf.State):
        new_state = rf.State()
        loop_out_, new_state.decoder = model.loop_step(
            **enc_args,
            enc_spatial_dim=enc_spatial_dim,
            input_embed=input_embed,
            state=state.decoder,
        )
        return loop_out_, new_state

    loop_out, _, _ = rf.scan(
        spatial_dim=targets_spatial_dim,
        xs=input_embeddings,
        ys=model.loop_step_output_templates(batch_dims=batch_dims, enc_spatial_dim=enc_spatial_dim),
        initial=rf.State(
            decoder=model.decoder_default_initial_state(
                batch_dims=batch_dims, enc_spatial_dim=enc_spatial_dim
            ),
        ),
        body=_body,
    )

    loop_out.pop("att_weights")

    logits = model.decode_logits(input_embed=input_embeddings, **loop_out)

    log_prob = rf.log_softmax(logits, axis=model.target_dim)
    # log_prob = rf.label_smoothed_log_prob_gradient(log_prob, 0.1)
    loss = rf.cross_entropy(
        target=targets,
        estimated=log_prob,
        estimated_type="log-probs",
        axis=model.target_dim,
    )
    loss.mark_as_loss("ce")


from_scratch_training: TrainDef[Model]
from_scratch_training.learning_rate_control_error_measure = "dev_score_full_sum"
