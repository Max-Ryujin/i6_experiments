import copy
import os
from typing import Dict

import i6_core.rasr as rasr
from i6_core.returnn import Checkpoint
from i6_core.returnn.config import ReturnnConfig
from i6_experiments.users.berger.args.experiments import transducer as exp_args
from i6_experiments.users.berger.args.returnn.config import get_returnn_config
from i6_experiments.users.berger.args.returnn.learning_rates import (
    LearningRateSchedules,
)
from i6_experiments.users.berger.corpus.librispeech.viterbi_transducer_data import (
    get_librispeech_data,
)
import i6_experiments.users.berger.network.models.context_1_transducer as transducer_model
from i6_experiments.users.berger.recipe.summary.report import SummaryReport
from i6_experiments.users.berger.systems.returnn_seq2seq_system import (
    ReturnnSeq2SeqSystem,
)
from i6_experiments.users.berger.systems.dataclasses import ReturnnConfigs, FeatureType, SummaryKey
from i6_experiments.users.berger.util import default_tools
from i6_private.users.vieting.helpers.returnn import serialize_dim_tags
from i6_experiments.users.berger.systems.dataclasses import AlignmentData
from i6_experiments.users.berger.network.helpers.label_context import ILMMode
from .config_01b_ctc_blstm_rasr_features import py as py_ctc
from .config_02b_transducer_rasr_features import py as py_transducer
from sisyphus import gs, tk

# ********** Settings **********

tools = copy.deepcopy(default_tools)

rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}


num_classes = 79


# ********** Return Config **********


def generate_returnn_config(
    train: bool,
    *,
    train_data_config: dict,
    dev_data_config: dict,
    model_preload: tk.Path,
    **kwargs,
) -> ReturnnConfig:
    if train:
        (
            network_dict,
            extra_python,
        ) = transducer_model.make_context_1_conformer_transducer_fullsum(
            num_outputs=num_classes,
            specaug_args={
                "max_time_num": 1,
                "max_time": 15,
                "max_feature_num": 5,
                "max_feature": 5,
            },
            conformer_args={
                "num_blocks": 12,
                "size": 512,
                "dropout": 0.1,
                "l2": 5e-06,
            },
            compress_joint_input=True,
            decoder_args={
                "dec_mlp_args": {
                    "num_layers": 2,
                    "size": 640,
                    "activation": "tanh",
                    "dropout": 0.1,
                    "l2": 5e-06,
                },
                "combination_mode": "concat",
                "joint_mlp_args": {
                    "num_layers": 1,
                    "size": 1024,
                    "dropout": 0.1,
                    "l2": 5e-06,
                    "activation": "tanh",
                },
            },
            fullsum_v2=True,
        )
    else:
        (
            network_dict,
            extra_python,
        ) = transducer_model.make_context_1_conformer_transducer_recog(
            num_outputs=num_classes,
            conformer_args={
                "num_blocks": 12,
                "size": 512,
            },
            decoder_args={
                "dec_mlp_args": {
                    "num_layers": 2,
                    "size": 640,
                    "activation": "tanh",
                },
                "combination_mode": "concat",
                "joint_mlp_args": {
                    "num_layers": 1,
                    "size": 1024,
                    "activation": "tanh",
                },
                "ilm_mode": ILMMode.ZeroEnc,
                "ilm_scale": kwargs.get("ilm_scale", 0.0),
            },
        )

    returnn_config = get_returnn_config(
        network=network_dict,
        target="classes",
        num_epochs=300,
        extra_python=extra_python,
        python_prolog=[
            "import sys",
            "sys.setrecursionlimit(10 ** 6)",
        ],
        num_inputs=50,
        num_outputs=num_classes,
        extern_data_config=True,
        extern_target_kwargs={"dtype": "int8" if train else "int32"},
        grad_noise=0.0,
        grad_clip=0.0,
        schedule=LearningRateSchedules.CONST_DECAY,
        const_lr=kwargs.get("lr", 8e-05),
        decay_lr=1e-05,
        final_lr=1e-06,
        batch_size=kwargs.get("batch_size", 3000),
        accum_grad=kwargs.get("accum_grad", 3),
        use_chunking=False,
        extra_config={
            "max_seq_length": {"classes": 600},
            "train": train_data_config,
            "dev": dev_data_config,
            "preload_from_files": {
                "base": {
                    "init_for_train": True,
                    "ignore_missing": True,
                    "filename": model_preload,
                }
            },
        },
    )
    returnn_config = serialize_dim_tags(returnn_config)

    return returnn_config


def run_exp(alignments: Dict[str, AlignmentData], viterbi_model_checkpoint: Checkpoint) -> SummaryReport:
    assert tools.returnn_root is not None
    assert tools.returnn_python_exe is not None
    assert tools.rasr_binary_path is not None

    data = get_librispeech_data(
        tools.returnn_root,
        tools.returnn_python_exe,
        lm_names=["4gram", "kazuki_transformer"],
        alignments=alignments,
        rasr_binary_path=tools.rasr_binary_path,
        add_unknown_phoneme_and_mapping=False,
        use_augmented_lexicon=True,
        use_wei_lexicon=False,
        feature_type=FeatureType.GAMMATONE_16K,
    )

    for data_input in data.data_inputs.values():
        data_input.create_lm_images(tools.rasr_binary_path)

    # ********** Step args **********

    train_args = exp_args.get_transducer_train_step_args(
        num_epochs=300,
    )

    recog_args = exp_args.get_transducer_recog_step_args(
        num_classes,
        lm_scales=[0.7, 0.8, 0.9, 1.0, 1.1],
        epochs=[300],
        search_parameters={
            "label-pruning": 19.8,
            "label-pruning-limit": 20000,
            "word-end-pruning": 0.8,
            "word-end-pruning-limit": 2000,
        },
        feature_type=FeatureType.GAMMATONE_16K,
        reduction_factor=4,
        reduction_subtrahend=0,
    )

    # ********** System **********

    system = ReturnnSeq2SeqSystem(
        tools,
        summary_keys=[
            SummaryKey.TRAIN_NAME,
            SummaryKey.CORPUS,
            SummaryKey.RECOG_NAME,
            SummaryKey.EPOCH,
            SummaryKey.PRIOR,
            SummaryKey.LM,
            SummaryKey.WER,
            SummaryKey.SUB,
            SummaryKey.INS,
            SummaryKey.DEL,
            SummaryKey.ERR,
        ],
    )

    # ********** Returnn Configs **********

    config_generator_kwargs = {
        "train_data_config": data.train_data_config,
        "dev_data_config": data.cv_data_config,
        "model_preload": viterbi_model_checkpoint,
    }

    for lr, batch_size, accum_grad in [
        # (8e-05, 3000, 3),
        # (4e-05, 3000, 3),
        (1e-04, 3000, 3),
        # (8e-05, 3000, 10),
    ]:
        train_config = generate_returnn_config(
            train=True, lr=lr, batch_size=batch_size, accum_grad=accum_grad, **config_generator_kwargs
        )
        recog_configs = {
            f"recog_ilm-{ilm_scale}": generate_returnn_config(
                train=False, ilm_scale=ilm_scale, **config_generator_kwargs
            )
            for ilm_scale in [0.0, 0.1, 0.2, 0.25, 0.3, 0.4]
        }

        returnn_configs = ReturnnConfigs(
            train_config=train_config,
            recog_configs=recog_configs,
        )
        system.add_experiment_configs(
            f"Conformer_Transducer_Fullsum_lr-{lr}_bs-{batch_size*accum_grad}", returnn_configs
        )

    system.init_corpora(
        dev_keys=data.dev_keys,
        test_keys=data.test_keys,
        align_keys=data.align_keys,
        corpus_data=data.data_inputs,
    )
    system.setup_scoring()

    system.run_train_step(**train_args)

    system.run_recog_step_for_corpora(corpora=["dev-clean_4gram", "dev-other_4gram"], **recog_args)
    system.run_recog_step_for_corpora(corpora=["test-clean_4gram", "test-other_4gram"], **recog_args)

    recog_args["lm_scales"] = [0.8]
    recog_args["seq2seq_v2"] = True
    recog_args["search_parameters"].update(
        {
            "full-sum-decoding": True,
            "label-full-sum": True,
            "label-recombination-limit": 1,
            "separate-recombination-lm": True,
            "recombination-lm.type": "simple-history",
        }
    )

    system.run_recog_step_for_corpora(
        recog_descriptor="fs",
        recog_exp_names={"Conformer_Transducer_Fullsum_lr-0.0001_bs-9000": ["recog_ilm-0.2"]},
        corpora=["dev-clean_4gram", "dev-other_4gram", "test-clean_4gram", "test-other_4gram"],
        **recog_args,
    )

    recog_args["search_parameters"].update(
        {
            "separate-lookahead-lm": True,
            "label-full-sum": False,
            "label-pruning": 16.2,
        }
    )
    recog_args["use_gpu"] = True
    recog_args["rtf"] = 100
    recog_args["mem"] = 24

    # recog_args["lm_scales"] = [0.8, 0.9]
    # for lm_lookahead_scale in [0.3, 0.4, 0.45, 0.5, 0.6]:
    recog_args["lm_scales"] = [0.9]
    # for lm_lookahead_scale in [0.3, 0.4, 0.45, 0.5, 0.6]:
    for lm_lookahead_scale in [0.45]:
        recog_args["lookahead_options"].update({"lm_lookahead_scale": lm_lookahead_scale})

        system.run_recog_step_for_corpora(
            recog_descriptor=f"fs_lookahead-{lm_lookahead_scale}",
            recog_exp_names={"Conformer_Transducer_Fullsum_lr-0.0001_bs-9000": ["recog_ilm-0.2", "recog_ilm-0.3"]},
            corpora=[
                "dev-clean_kazuki_transformer",
                "dev-other_kazuki_transformer",
                "test-clean_kazuki_transformer",
                "test-other_kazuki_transformer",
            ],
            **recog_args,
        )

    assert system.summary_report
    return system.summary_report


def py() -> SummaryReport:
    _, _, alignments = py_ctc()
    _, model = py_transducer()

    filename_handle = os.path.splitext(os.path.basename(__file__))[0][len("config_") :]
    gs.ALIAS_AND_OUTPUT_SUBDIR = f"{filename_handle}/"

    summary_report = run_exp(alignments, model)

    tk.register_report(f"{gs.ALIAS_AND_OUTPUT_SUBDIR}/summary.report", summary_report)

    return summary_report
