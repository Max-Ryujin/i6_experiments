import copy
import os
from typing import Dict, Tuple

import i6_core.rasr as rasr
from i6_core.returnn.config import ReturnnConfig
from i6_experiments.users.berger.args.experiments import ctc as exp_args
from i6_experiments.users.berger.args.returnn.config import get_returnn_config
from i6_experiments.users.berger.args.returnn.learning_rates import (
    LearningRateSchedules,
)
import i6_experiments.users.berger.network.models.fullsum_ctc_raw_samples as ctc_model
from i6_experiments.users.berger.recipe.summary.report import SummaryReport
from i6_experiments.users.berger.systems.returnn_seq2seq_system import (
    ReturnnSeq2SeqSystem,
)
from i6_experiments.users.berger.systems.dataclasses import ReturnnConfigs
from i6_experiments.users.berger.util import default_tools
from i6_private.users.vieting.helpers.returnn import serialize_dim_tags
from recipe.i6_experiments.users.berger.corpus.librispeech.ctc_data import (
    get_librispeech_data_hdf,
)
from sisyphus import gs, tk

tools = copy.deepcopy(default_tools)

# ********** Settings **********

rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}


num_classes = 79


# ********** Return Config generators **********


def generate_returnn_config(
    train: bool,
    *,
    loss_corpus: tk.Path,
    loss_lexicon: tk.Path,
    am_args: dict,
    train_data_config: dict,
    dev_data_config: dict,
) -> ReturnnConfig:
    if train:
        network_dict, extra_python = ctc_model.make_conformer_fullsum_ctc_model(
            num_outputs=num_classes,
            gt_args={
                "sample_rate": 16000,
                "specaug_v2": False,
                "specaug_args": {
                    "max_time_num": 1,
                    "max_time": 15,
                    "max_feature_num": 5,
                    "max_feature": 5,
                },
            },
            conformer_args={
                "num_blocks": 12,
                "size": 512,
                "dropout": 0.1,
                "l2": 1e-04,
            },
            output_args={
                "rasr_binary_path": tools.rasr_binary_path,
                "loss_corpus_path": loss_corpus,
                "loss_lexicon_path": loss_lexicon,
                "am_args": am_args,
            },
        )
    else:
        network_dict, extra_python = ctc_model.make_conformer_ctc_recog_model(
            num_outputs=79,
            gt_args={
                "sample_rate": 16000,
                "specaug_after_dct": False,
            },
            conformer_args={
                "num_blocks": 12,
                "size": 512,
            },
        )

    returnn_config = get_returnn_config(
        network=network_dict,
        target=None,
        num_epochs=500,
        num_inputs=1,
        python_prolog=[
            "import sys",
            "sys.setrecursionlimit(10 ** 6)",
        ],
        extra_python=extra_python,
        extern_data_config=True,
        extern_data_kwargs={"dtype": "int16" if train else "float32"},
        grad_noise=0.0,
        grad_clip=100.0,
        schedule=LearningRateSchedules.OCLR,
        initial_lr=1e-05,
        peak_lr=3e-04,
        final_lr=1e-05,
        batch_size=1_600_000,
        use_chunking=False,
        extra_config={
            "train": train_data_config,
            "dev": dev_data_config,
        },
    )
    returnn_config = serialize_dim_tags(returnn_config)

    return returnn_config


def run_exp() -> Tuple[SummaryReport, Dict]:
    assert tools.returnn_root is not None
    assert tools.returnn_python_exe is not None

    data = get_librispeech_data_hdf(
        tools.returnn_root,
        add_unknown=False,
        augmented_lexicon=True,
        test_keys=[
            "test-clean",
            "test-other",
        ],
    )

    # ********** Step args **********

    train_args = exp_args.get_ctc_train_step_args(
        num_epochs=500,
    )

    recog_args = exp_args.get_ctc_recog_step_args(num_classes)
    align_args = exp_args.get_ctc_align_step_args(num_classes)
    recog_args["epochs"] = [400, 480, 500, "best"]
    recog_args["prior_scales"] = [0.3]
    recog_args["lm_scales"] = [0.9]
    align_args["epochs"] = ["best"]

    # ********** System **********

    system = ReturnnSeq2SeqSystem(tools)

    system.init_corpora(
        dev_keys=data.dev_keys,
        test_keys=data.test_keys,
        align_keys=data.align_keys,
        corpus_data=data.data_inputs,
        am_args=exp_args.ctc_recog_am_args,
    )
    system.setup_scoring()

    # ********** Returnn Configs **********

    config_generator_kwargs = {
        "loss_corpus": data.loss_corpus,
        "loss_lexicon": data.loss_lexicon,
        "am_args": exp_args.ctc_loss_am_args,
        "train_data_config": data.train_data_config,
        "dev_data_config": data.cv_data_config,
    }

    train_config = generate_returnn_config(train=True, **config_generator_kwargs)
    recog_config = generate_returnn_config(train=False, **config_generator_kwargs)

    returnn_configs = ReturnnConfigs(
        train_config=train_config,
        recog_configs={"recog": recog_config},
    )

    system.add_experiment_configs("Conformer_CTC", returnn_configs)

    system.run_train_step(**train_args)
    system.run_dev_recog_step(**recog_args)
    system.run_test_recog_step(**recog_args)
    alignments = system.run_align_step(**align_args)

    assert system.summary_report
    return system.summary_report, alignments


def py() -> Tuple[SummaryReport, Dict]:
    filename_handle = os.path.splitext(os.path.basename(__file__))[0][len("config_") :]
    gs.ALIAS_AND_OUTPUT_SUBDIR = f"{filename_handle}/"

    summary_report, alignments = run_exp()

    tk.register_report(f"{gs.ALIAS_AND_OUTPUT_SUBDIR}/summary.report", summary_report)

    return summary_report, alignments