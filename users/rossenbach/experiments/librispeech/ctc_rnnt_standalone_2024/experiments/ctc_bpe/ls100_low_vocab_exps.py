from sisyphus import tk

import copy
from dataclasses import asdict
import numpy as np
from typing import cast

from i6_core.tools.parameter_tuning import GetOptimalParametersAsVariableJob

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from ...data.common import DatasetSettings, build_test_dataset
from ...data.bpe import build_bpe_training_datasets, get_text_lexicon
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...lm import get_4gram_binary_lm
from ...pipeline import training, prepare_asr_model, search, ASRModel
from ...storage import add_ctc_model

from ...report import tune_and_evalue_report

def bpe_ls100_0824_low_vocab_test():
    prefix_name = "experiments/librispeech/ctc_rnnt_standalone_2024/ls100_ctc_bpe_low_vocab"

    train_settings = DatasetSettings(
        preemphasis=0.97,  # TODO: Check if this is really useful
        peak_normalization=True,  # TODO: Also check if really useful, older Attention setups did not have that
        # training
        train_partition_epoch=3,
        train_seq_ordering="laplace:.1000",
    )
    
    arpa_4gram_lm = get_4gram_binary_lm(prefix_name=prefix_name)

    default_returnn = {
        "returnn_exe": RETURNN_EXE,
        "returnn_root": MINI_RETURNN_ROOT,
    }

    from ...pytorch_networks.ctc.decoder.flashlight_ctc_v1 import DecoderConfig
    from ...pytorch_networks.ctc.decoder.greedy_bpe_ctc_v3 import DecoderConfig as GreedyDecoderConfig
    
    from ...pytorch_networks.ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6_cfg import \
        SpecaugConfig, VGG4LayerActFrontendV1Config_mod, ModelConfig, LogMelFeatureExtractionV1Config

    fe_config = LogMelFeatureExtractionV1Config(
        sample_rate=16000,
        win_size=0.025,
        hop_size=0.01,
        f_min=60,
        f_max=7600,
        min_amp=1e-10,
        num_filters=80,
        center=False,
    )
    specaug_config_full = SpecaugConfig(
        repeat_per_n_frames=25,
        max_dim_time=20,
        max_dim_feat=16,  # Normal Style
        num_repeat_feat=5,
    )
    frontend_config = VGG4LayerActFrontendV1Config_mod(
        in_features=80,
        conv1_channels=32,
        conv2_channels=64,
        conv3_channels=64,
        conv4_channels=32,
        conv_kernel_size=(3, 3),
        conv_padding=None,
        pool1_kernel_size=(2, 1),
        pool1_stride=(2, 1),
        pool1_padding=None,
        pool2_kernel_size=(2, 1),
        pool2_stride=(2, 1),
        pool2_padding=None,
        activation_str="ReLU",
        out_features=384,
        activation=None,
    )
    
    def tune_and_evaluate_helper(training_name, dev_dataset_tuples, test_dataset_tuples, asr_model, base_decoder_config, lm_scales, prior_scales):
        tune_parameters = []
        tune_values_clean = []
        tune_values_other = []
        report_values = {}
        for lm_weight in lm_scales:
            for prior_scale in prior_scales:
                decoder_config = copy.deepcopy(base_decoder_config)
                decoder_config.lm_weight = lm_weight
                decoder_config.prior_scale = prior_scale
                search_name = training_name + "/search_lm%.1f_prior%.1f" % (lm_weight, prior_scale)
                search_jobs, wers = search(
                    search_name,
                    forward_config={},
                    asr_model=asr_model,
                    decoder_module="ctc.decoder.flashlight_ctc_v1",
                    decoder_args={"config": asdict(decoder_config)},
                    test_dataset_tuples=dev_dataset_tuples,
                    **default_returnn
                )
                tune_parameters.append((lm_weight, prior_scale))
                tune_values_clean.append((wers[search_name + "/dev-clean"]))
                tune_values_other.append((wers[search_name + "/dev-other"]))

        for key, tune_values in [("test-clean", tune_values_clean), ("test-other", tune_values_other)]:
            pick_optimal_params_job = GetOptimalParametersAsVariableJob(parameters=tune_parameters, values=tune_values, mode="minimize")
            pick_optimal_params_job.add_alias(training_name + f"/pick_best_{key}")
            decoder_config = copy.deepcopy(base_decoder_config)
            decoder_config.lm_weight = pick_optimal_params_job.out_optimal_parameters[0]
            decoder_config.prior_scale = pick_optimal_params_job.out_optimal_parameters[1]
            search_jobs, wers = search(
                training_name, forward_config={}, asr_model=asr_model, decoder_module="ctc.decoder.flashlight_ctc_v1",
                decoder_args={"config": asdict(decoder_config)}, test_dataset_tuples={key: test_dataset_tuples[key]},
                **default_returnn
            )
            report_values[key] = wers[training_name + "/" + key]

        tune_and_evalue_report(
            training_name=training_name,
            tune_parameters=tune_parameters,
            tuning_names=["LM", "Prior"],
            tune_values_clean=tune_values_clean,
            tune_values_other=tune_values_other,
            report_values=report_values
        )

    def greedy_search_helper(
            training_name: str,
            asr_model: ASRModel,
            decoder_config: GreedyDecoderConfig,
            dev_dataset_tuples,
            test_dataset_tuples,
        ):
        # remove prior if exists
        asr_model = copy.deepcopy(asr_model)
        asr_model.prior_file = None

        search_name = training_name + "/search_greedy"
        search_jobs, wers = search(
            search_name,
            forward_config={},
            asr_model=asr_model,
            decoder_module="ctc.decoder.greedy_bpe_ctc_v3",
            decoder_args={"config": asdict(decoder_config)},
            test_dataset_tuples={**dev_dataset_tuples, **test_dataset_tuples},
            **default_returnn,
        )

    for BPE_SIZE in [0, 128, 512]:

        # build the training datasets object containing train, cv, dev-train and the extern_data dict
        train_data_bpe = build_bpe_training_datasets(
            prefix=prefix_name,
            librispeech_key="train-clean-100",
            bpe_size=BPE_SIZE,
            settings=train_settings,
            use_postfix=False,
        )

        label_datastream_bpe = cast(LabelDatastream, train_data_bpe.datastreams["labels"])
        vocab_size_without_blank = label_datastream_bpe.vocab_size

        dev_dataset_tuples = {}
        for testset in ["dev-clean", "dev-other"]:
            dev_dataset_tuples[testset] = build_test_dataset(
                dataset_key=testset,
                settings=train_settings,
            )

        test_dataset_tuples = {}
        for testset in ["test-clean", "test-other"]:
            test_dataset_tuples[testset] = build_test_dataset(
                dataset_key=testset,
                settings=train_settings,
            )

        model_config = ModelConfig(
            feature_extraction_config=fe_config,
            frontend_config=frontend_config,
            specaug_config=specaug_config_full,
            label_target_size=vocab_size_without_blank,
            conformer_size=384,
            num_layers=12,
            num_heads=8,
            ff_dim=1536,
            att_weights_dropout=0.2,
            conv_dropout=0.2,
            ff_dropout=0.2,
            mhsa_dropout=0.2,
            conv_kernel_size=31,
            final_dropout=0.2,
            specauc_start_epoch=11,  # BPE does not converge otherwise
        )

        default_decoder_config_bpe = DecoderConfig(
            lexicon=get_text_lexicon(prefix=prefix_name, librispeech_key="train-clean-100", bpe_size=BPE_SIZE),
            returnn_vocab=label_datastream_bpe.vocab,
            beam_size=1024,
            beam_size_token=16,  # makes it much faster
            arpa_lm=arpa_4gram_lm,
            beam_threshold=14,
        )

        train_config_11gbgpu = {
            "optimizer": {"class": "radam", "epsilon": 1e-16, "weight_decay": 1e-2, "decoupled_weight_decay": True},
            # There is a higher LR, because this model is only 384 in dimension
            "learning_rates": list(np.linspace(7e-5, 7e-4, 140)) + list(
                np.linspace(7e-4, 7e-5, 140)) + list(np.linspace(7e-5, 1e-7, 20)),
            #############
            "batch_size": 240 * 16000,
            "max_seq_length": {"audio_features": 35 * 16000},
            "accum_grad_multiple_step": 1,
            "gradient_clip_norm": 1.0,
        }
        
        # for a first try still use old model with conv first
        network_module_conv_first = "ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6_conv_first"
        train_args_conv_first = {
            "config": train_config_11gbgpu,
            "network_module": network_module_conv_first,
            "net_args": {"model_config_dict": asdict(model_config)},
            "use_speed_perturbation": True,
            "debug": False,
        }

        training_name = prefix_name + "/" + str(BPE_SIZE) + "/" + network_module_conv_first + ".384dim_sub4_11gbgpu_100eps_radam"
        train_job = training(training_name, train_data_bpe, train_args_conv_first, num_epochs=300,
                             **default_returnn)
        train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        asr_model = prepare_asr_model(
            training_name, train_job, train_args_conv_first, with_prior=True, datasets=train_data_bpe,
            get_specific_checkpoint=300
        )
        tune_and_evaluate_helper(training_name, dev_dataset_tuples, test_dataset_tuples, asr_model, default_decoder_config_bpe, lm_scales=[1.6, 1.8, 2.0, 2.2, 2.4, 2.6],
                                 prior_scales=[0.1, 0.2, 0.3, 0.4])
        greedy_decoder_config = GreedyDecoderConfig(
            returnn_vocab=label_datastream_bpe.vocab,
        )
        greedy_search_helper(training_name, asr_model=asr_model, decoder_config=greedy_decoder_config, dev_dataset_tuples=dev_dataset_tuples, test_dataset_tuples=test_dataset_tuples)