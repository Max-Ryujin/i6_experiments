from sisyphus import tk

import copy
from dataclasses import asdict
import numpy as np
from typing import cast, List

from i6_core.tools.parameter_tuning import GetOptimalParametersAsVariableJob

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from ...data.common import DatasetSettings, build_test_dataset
from ...data.bpe import build_bpe_training_datasets, get_text_lexicon
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...lm import get_4gram_binary_lm
from ...pipeline import training, prepare_asr_model, search, ASRModel
from ...storage import get_ctc_model


def rnnt_bpe_ls960_1023_low_bpe_streaming_finetune():
    prefix_name = "experiments/librispeech/ctc_rnnt_standalone_2024/ls960_rnnt_bpe_low_bpe_streaming_finetune"

    train_settings = DatasetSettings(
        preemphasis=0.97,  # TODO: Check if this is really useful
        peak_normalization=True,  # TODO: Also check if really useful, older Attention setups did not have that
        # training
        train_partition_epoch=10,
        train_seq_ordering="laplace:.1000",
    )

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

    default_returnn = {
        "returnn_exe": RETURNN_EXE,
        "returnn_root": MINI_RETURNN_ROOT,
    }

    from ...pytorch_networks.rnnt.decoder.carryover_prelah_decoder import DecoderConfig

    def evaluate_helper(
            training_name: str,
            asr_model: ASRModel,
            base_decoder_config: DecoderConfig,
            beam_size: int = 1,
            use_gpu=False,
            decoder_module: str = "rnnt.decoder.experimental_rnnt_decoder"
    ):
        """
        Example helper to execute tuning over lm_scales and prior scales.
        With the best values runs test-clean and test-other.

        This is just a reference helper and can (should) be freely changed, copied, modified etc...

        :param training_name: for alias and output names
        :param asr_model: ASR model to use
        :param base_decoder_config: any decoder config dataclass

        """
        decoder_config = copy.deepcopy(base_decoder_config)
        decoder_config.beam_size = beam_size
        search_name = training_name + "/search_bs%i" % beam_size
        search_jobs, wers = search(
            search_name,
            forward_config={"seed": 2} if use_gpu else {},
            asr_model=asr_model,
            decoder_module=decoder_module,
            decoder_args={"config": asdict(decoder_config)},
            test_dataset_tuples={**dev_dataset_tuples}, #**test_dataset_tuples},
            use_gpu=use_gpu,
            **default_returnn,
        )

    from ...pytorch_networks.rnnt.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v9_cfg import (
        SpecaugConfig,
        VGG4LayerActFrontendV1Config_mod,
        ModelConfig,
        LogMelFeatureExtractionV1Config,
        PredictorConfig
    )

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
    specaug_config = SpecaugConfig(
        repeat_per_n_frames=25,
        max_dim_time=20,
        max_dim_feat=8,  # Jingjing style
        num_repeat_feat=5,
    )
    specaug_config_full = SpecaugConfig(
        repeat_per_n_frames=25,
        max_dim_time=20,
        max_dim_feat=16,  # Old style
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
        pool1_stride=(3, 1),
        pool1_padding=None,
        pool2_kernel_size=(2, 1),
        pool2_stride=(2, 1),
        pool2_padding=None,
        activation_str="ReLU",
        out_features=512,
        activation=None,
    )
    predictor_config = PredictorConfig(
        symbol_embedding_dim=256,
        emebdding_dropout=0.2,
        num_lstm_layers=1,
        lstm_hidden_dim=512,
        lstm_dropout=0.1,
    )
    for BPE_SIZE in [128, 256, 512, 1024]:
        # build the training datasets object containing train, cv, dev-train and the extern_data dict
        train_data_bpe = build_bpe_training_datasets(
            prefix=prefix_name,
            librispeech_key="train-other-960",
            bpe_size=BPE_SIZE,
            settings=train_settings,
            use_postfix=True,  # RNN-T now, use postfix
        )
        label_datastream_bpe = cast(LabelDatastream, train_data_bpe.datastreams["labels"])
        vocab_size_without_blank = label_datastream_bpe.vocab_size

        # decoder_config_bpeany_greedy = DecoderConfig(
        #     beam_size=1,  # greedy as default
        #     returnn_vocab=label_datastream_bpe.vocab
        # )

        model_config_v5_sub6_512lstm = ModelConfig(
            feature_extraction_config=fe_config,
            frontend_config=frontend_config,
            specaug_config=specaug_config,
            predictor_config=predictor_config,
            label_target_size=vocab_size_without_blank,
            conformer_size=512,
            num_layers=12,
            num_heads=8,
            ff_dim=2048,
            att_weights_dropout=0.1,
            conv_dropout=0.1,
            ff_dropout=0.1,
            mhsa_dropout=0.1,
            conv_kernel_size=31,
            final_dropout=0.1,
            specauc_start_epoch=21,
            joiner_dim=640,
            joiner_activation="relu",
            joiner_dropout=0.1,
            ctc_output_loss=0.0
        )
        model_config_v5_sub6_512lstm_start1 = copy.deepcopy(model_config_v5_sub6_512lstm)
        model_config_v5_sub6_512lstm_start1.specauc_start_epoch = 1

        model_config_v5_sub6_512lstm_start1_full_spec = copy.deepcopy(model_config_v5_sub6_512lstm_start1)
        model_config_v5_sub6_512lstm_start1_full_spec.specaug_config = specaug_config_full

        # Default configs for continued training
        train_config_24gbgpu_amp = {
            "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-3},
            "learning_rates": list(np.linspace(5e-5, 5e-4, 120)) + list(
                np.linspace(5e-4, 5e-5, 120)) + list(np.linspace(5e-5, 1e-7, 10)),
            #############
            "batch_size": 120 * 16000,  # RNN-T has very high memory consumption
            "max_seq_length": {"audio_features": 35 * 16000},
            "accum_grad_multiple_step": 2,
            "torch_amp_options": {"dtype": "bfloat16"},
        }

        network_module = "rnnt.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v9_i6_native"
        train_args_fullspec = {
            "config": train_config_24gbgpu_amp,
            "network_module": network_module,
            "net_args": {"model_config_dict": asdict(model_config_v5_sub6_512lstm_start1_full_spec)},
            "include_native_ops": True,
            "debug": False,
        }

        network_module = "rnnt.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v9_i6_native"
        train_args_warprnnt_fullspec_from_ctc = copy.deepcopy(train_args_fullspec)
        train_args_warprnnt_fullspec_from_ctc["config"]["preload_from_files"] = {
            "encoder": {
                # "filename": get_ctc_model(
                #     f"ls960_ctc_bpe_{BPE_SIZE}.ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6.512dim_sub4_24gbgpu_50eps_ckpt500"
                # ).checkpoint,
                "init_for_train": True,
                "ignore_missing": True,
            }
        }

        if BPE_SIZE == 128:
            # DO HERE AGAIN IN CORRECT
            KEEP = [300, 400, 500, 600, 700, 800, 900, 950, 980]
            network_module = "rnnt.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v9_i6_native_conv_first"
            train_args_warprnnt_fullspec_from_ctc100 = copy.deepcopy(train_args_fullspec)
            train_args_warprnnt_fullspec_from_ctc100["network_module"] = network_module
            train_args_warprnnt_fullspec_from_ctc100["config"]["preload_from_files"] = {
                "encoder": {
                    # "filename": get_ctc_model(
                    #     f"ls960_ctc_bpe_{BPE_SIZE}.ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6_conv_first.512dim_sub4_24gbgpu_100eps_ckpt1000"
                    # ).checkpoint,
                    "init_for_train": True,
                    "ignore_missing": True,
                }
            }
            train_args_warprnnt_fullspec_from_ctc100["config"]["learning_rates"] = list(
                np.linspace(5e-5, 5e-4, 240)) + list(
                np.linspace(5e-4, 5e-5, 720)) + list(np.linspace(5e-5, 1e-7, 40))
            train_args_warprnnt_fullspec_from_ctc100["config"]["cleanup_old_models"] = {
                "keep_last_n": 4,
                "keep_best_n": 4,
                "keep": KEEP
            }
            train_args_warprnnt_fullspec_from_ctc100["config"]["gradient_clip"] = 1.0

            # small BPE saves a lot of memory, train without grad accum
            train_args_warprnnt_fullspec_from_ctc100_noacumm = copy.deepcopy(train_args_warprnnt_fullspec_from_ctc100)
            train_args_warprnnt_fullspec_from_ctc100_noacumm["config"]["accum_grad_multiple_step"] = 1
            train_args_warprnnt_fullspec_from_ctc100_noacumm["config"]["batch_size"] = 240 * 16000

            # With speed perturbation
            train_args_warprnnt_fullspec_from_ctc100_noacumm_sp = copy.deepcopy(
                train_args_warprnnt_fullspec_from_ctc100_noacumm)
            train_args_warprnnt_fullspec_from_ctc100_noacumm_sp["use_speed_perturbation"] = True

            # With more L2
            train_args_warprnnt_fullspec_from_ctc100_noacumm_sp_morel2 = copy.deepcopy(
                train_args_warprnnt_fullspec_from_ctc100_noacumm_sp)
            train_args_warprnnt_fullspec_from_ctc100_noacumm_sp_morel2["config"]["optimizer"]["weight_decay"] = 1e-2

            # From scratch
            train_args_warprnnt_fullspec_noacumm_morel2_radam = copy.deepcopy(
                train_args_warprnnt_fullspec_from_ctc100_noacumm_sp_morel2)
            train_args_warprnnt_fullspec_noacumm_morel2_radam["config"].pop("preload_from_files")
            train_args_warprnnt_fullspec_noacumm_morel2_radam["config"].pop("gradient_clip")
            train_args_warprnnt_fullspec_noacumm_morel2_radam["config"]["gradient_clip_norm"] = 1.0
            train_args_warprnnt_fullspec_noacumm_morel2_radam["config"]["optimizer"] = {"class": "radam",
                                                                                        "epsilon": 1e-12,
                                                                                        "weight_decay": 1e-2,
                                                                                        "decoupled_weight_decay": True}
            train_args_warprnnt_fullspec_noacumm_morel2_radam["use_speed_perturbation"] = False

            model_config_v5_sub6_512lstm_start11_full_spec = copy.deepcopy(
                model_config_v5_sub6_512lstm_start1_full_spec)
            model_config_v5_sub6_512lstm_start11_full_spec.specauc_start_epoch = 11
            model_config_v5_sub6_512lstm_start11_full_spec.ctc_output_loss = 0.3

            train_args_warprnnt_fullspec_noacumm_morel2_radam["net_args"] = {
                "model_config_dict": asdict(model_config_v5_sub6_512lstm_start11_full_spec)}

            network_module = "rnnt.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v9_i6_native"

            training_name = prefix_name + "/" + str(
                BPE_SIZE) + "/" + network_module + ".512dim_sub6_24gbgpu_100eps_accum1_gradclip_fullspec11_morel2_from_scratch"
            train_job = training(training_name, train_data_bpe, train_args_warprnnt_fullspec_noacumm_morel2_radam,
                                 num_epochs=1000, **default_returnn)
            train_job.rqmt["gpu_mem"] = 24
            train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

            # finetune from this checkpoint
            asr_model = prepare_asr_model(
                training_name, train_job, train_args_warprnnt_fullspec_noacumm_morel2_radam, with_prior=False,
                datasets=train_data_bpe, get_specific_checkpoint=1000
            )

            from ...pytorch_networks.rnnt.conformer_0924.uni_lah_carryover_cfg import ModelConfigPrefrontendLAH
            model_config_finetuned = ModelConfigPrefrontendLAH(
                feature_extraction_config=fe_config,
                frontend_config=frontend_config,
                specaug_config=specaug_config_full,
                predictor_config=predictor_config,
                label_target_size=vocab_size_without_blank,
                conformer_size=512,
                num_layers=12,
                num_heads=8,
                ff_dim=2048,
                att_weights_dropout=0.1,
                conv_dropout=0.1,
                ff_dropout=0.1,
                mhsa_dropout=0.1,
                conv_kernel_size=31,
                final_dropout=0.1,
                specauc_start_epoch=11,
                joiner_dim=640,
                joiner_activation="relu",
                joiner_dropout=0.1,
                ctc_output_loss=0.3,
                use_vgg=None,
                fastemit_lambda=None,

                chunk_size=2.4 * 16e3,
                carry_over_size=2,
                lookahead_size=0.48 * 16e3,
                online_model_scale=0.5,
            )

            decoder_config = DecoderConfig(
                beam_size=12,
                returnn_vocab=label_datastream_bpe.vocab,
                left_size=int(model_config_finetuned.chunk_size),
                right_size=int(model_config_finetuned.lookahead_size),
                test_version=0.0,
            )

            train_args_streaming_finetune = copy.deepcopy(train_args_warprnnt_fullspec_noacumm_morel2_radam)

            train_args_streaming_finetune["net_args"] = {
                "model_config_dict": asdict(model_config_finetuned)}

            # NOTE: preloading baseline model
            train_args_streaming_finetune["config"]["preload_from_files"] = {
                "existing model": {
                    "filename": asr_model.checkpoint,
                    "init_for_train": True,
                    "ignore_missing": False,
                }
            }

            train_args_streaming_finetune["config"]["learning_rates"] = list(
                np.linspace(5e-5, 5e-4, 24)) + list(
                np.linspace(5e-4, 5e-5, 72)) + list(np.linspace(5e-5, 1e-7, 4))

            KEEP = [10, 25, 40, 60, 80, 90, 95, 98]
            train_args_streaming_finetune["config"]["cleanup_old_models"] = {
                "keep_last_n": 4,
                "keep_best_n": 4,
                "keep": KEEP
            }

            network_module = "rnnt.conformer_0924.model_streaming_lah_co_prefrontendlah_v1"
            train_args_streaming_finetune["network_module"] = network_module

            training_name = prefix_name + "/" + str(
                BPE_SIZE) + "/" + network_module + ".512dim_sub6_24gbgpu_10eps_streaming_finetune"
            train_job = training(training_name, train_data_bpe, train_args_streaming_finetune,
                                 num_epochs=100, **default_returnn)
            train_job.rqmt["gpu_mem"] = 24
            train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

            for keep in KEEP:
                asr_model = prepare_asr_model(
                    training_name, train_job, train_args_streaming_finetune, with_prior=False,
                    datasets=train_data_bpe, get_specific_checkpoint=keep
                )
                evaluate_helper(
                    training_name + "/keep_%i" % keep,
                    asr_model,
                    decoder_config,
                    use_gpu=True,
                    beam_size=12,
                    decoder_module="rnnt.decoder.carryover_prelah_decoder"
                )
            asr_model = prepare_asr_model(
                training_name, train_job, train_args_streaming_finetune, with_prior=False,
                datasets=train_data_bpe, get_specific_checkpoint=100
            )
            evaluate_helper(
                training_name + "/keep_%i" % 100,
                asr_model,
                decoder_config,
                use_gpu=True,
                beam_size=12,
                decoder_module="rnnt.decoder.carryover_prelah_decoder"
            )
