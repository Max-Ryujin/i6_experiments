from typing import Tuple, Optional, List, Union, Dict

from i6_core.returnn import PtCheckpoint
from i6_core.returnn.training import Checkpoint
from sisyphus import Path

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder.segmental import LibrispeechConformerSegmentalAttentionConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.recog_new import ReturnnSegmentalAttDecodingPipeline, RasrSegmentalAttDecodingExperiment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.realignment_new import RasrRealignmentExperiment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.recog import _returnn_v2_forward_step, _returnn_v2_get_forward_callback
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.recog import model_recog
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.recog_chunked import model_recog as model_recog_chunked
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.label_singletons import LibrispeechBPE10025_CTC_ALIGNMENT
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.bpe.bpe import LibrispeechBPE10025
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.global_att.config_builder import get_global_att_config_builder_rf
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.phonemes.gmm_alignments import LIBRISPEECH_GMM_WORD_ALIGNMENT
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.checkpoints import lm_checkpoints


def center_window_returnn_frame_wise_beam_search(
        alias: str,
        config_builder: LibrispeechConformerSegmentalAttentionConfigBuilder,
        checkpoint: Union[Checkpoint, Dict],
        base_model_scale: float = 1.0,
        lm_scale_list: Tuple[float, ...] = (0.0,),
        lm_type: Optional[str] = None,
        ilm_scale_list: Tuple[float, ...] = (0.0,),
        ilm_type: Optional[str] = None,
        lm_alias: Optional[str] = "kazuki-10k",
        lm_checkpoint: Optional[Checkpoint] = lm_checkpoints["kazuki-10k"],
        subtract_ilm_eos_score: bool = True,
        beam_size_list: Tuple[int, ...] = (12,),
        checkpoint_aliases: Tuple[str, ...] = ("last", "best", "best-4-avg"),
        run_analysis: bool = False,
        att_weight_seq_tags: Optional[List] = (
          "dev-other/3660-6517-0005/3660-6517-0005",
          "dev-other/6467-62797-0001/6467-62797-0001",
          "dev-other/6467-62797-0002/6467-62797-0002",
          "dev-other/7697-105815-0015/7697-105815-0015",
          "dev-other/7697-105815-0051/7697-105815-0051",
        ),
        pure_torch: bool = False,
        use_recombination: Optional[str] = "sum",
        batch_size: Optional[int] = None,
        corpus_keys: Tuple[str, ...] = ("dev-other",),
        reset_eos_params: bool = False,
        analyze_gradients: bool = False,
        analyze_gradients_search: bool = False,
        concat_num: Optional[int] = None,
        only_do_analysis: bool = False,
        analysis_dump_gradients: bool = False,
        analysis_ground_truth_hdf: Optional[Path] = None,
        analysis_analyze_gradients_plot_encoder_layers: bool = False,
        analsis_analyze_gradients_plot_log_gradients: bool = False,
        att_readout_scale: Optional[float] = None,
        h_t_readout_scale: Optional[float] = None,
        external_aed_opts: Optional[Dict] = None,
        external_transducer_opts: Optional[Dict] = None,
        calc_search_errors: bool = False,
        add_lm_eos_to_non_blank_end_hyps: bool = False,
        lm_eos_scale: float = 1.0,
        time_rqmt: Optional[int] = None,
        blank_scale: Optional[float] = 1.0,
        emit_scale: Optional[float] = 1.0,
        length_normalization_exponent: Optional[float] = None,
        sbatch_args: Optional[List[str]] = None,
):
  if lm_type is not None:
    assert len(checkpoint_aliases) == 1, "Do LM recog only for the best checkpoint"

  ilm_opts = {"type": ilm_type, "correct_eos": subtract_ilm_eos_score}
  if ilm_type == "mini_att":
    ilm_opts.update({
      "use_se_loss": False,
      "get_global_att_config_builder_rf_func": get_global_att_config_builder_rf,
    })

  recog_opts = {
    "recog_def": model_recog_chunked if config_builder.use_vertical_transitions else model_recog,
    "forward_step_func": _returnn_v2_forward_step,
    "forward_callback": _returnn_v2_get_forward_callback,
    "use_recombination": use_recombination,
    "reset_eos_params": reset_eos_params,
    "att_readout_scale": att_readout_scale,
    "h_t_readout_scale": h_t_readout_scale,
    "dataset_opts": {"target_is_alignment": True},
    "external_aed_opts": external_aed_opts,
    "base_model_scale": base_model_scale,
    "blank_scale": blank_scale,
    "emit_scale": emit_scale,
    "length_normalization_exponent": length_normalization_exponent,
    "external_transducer_opts": external_transducer_opts,
  }
  if concat_num is not None:
    recog_opts["dataset_opts"]["concat_num"] = concat_num

  if batch_size is not None:
    recog_opts["batch_size"] = batch_size

  if time_rqmt is not None:
    recog_rqmt = {"time": time_rqmt}
  else:
    recog_rqmt = {}

  if sbatch_args is not None:
    recog_rqmt["sbatch_args"] = sbatch_args

  if run_analysis:
    assert len(corpus_keys) == 1, "Only one corpus key is supported for analysis"
    assert corpus_keys[0] in ("train", "dev-other")

    if corpus_keys[0] == "train":
      ref_alignment_hdf = LIBRISPEECH_GMM_WORD_ALIGNMENT.alignment_paths["train"]
      ref_alignment_blank_idx = LIBRISPEECH_GMM_WORD_ALIGNMENT.model_hyperparameters.blank_idx
      ref_alignment_vocab_path = LIBRISPEECH_GMM_WORD_ALIGNMENT.vocab_path
    else:
      ref_alignment_hdf = LibrispeechBPE10025_CTC_ALIGNMENT.alignment_paths["dev-other"]
      ref_alignment_blank_idx = LibrispeechBPE10025_CTC_ALIGNMENT.model_hyperparameters.blank_idx
      ref_alignment_vocab_path = LibrispeechBPE10025_CTC_ALIGNMENT.vocab_path

    analysis_opts = {
      "att_weight_seq_tags": list(att_weight_seq_tags) if att_weight_seq_tags is not None else None,
      "analyze_gradients": analyze_gradients,
      "analyze_gradients_search": analyze_gradients_search,
      "ref_alignment_hdf": ref_alignment_hdf,
      "ref_alignment_blank_idx": ref_alignment_blank_idx,
      "ref_alignment_vocab_path": ref_alignment_vocab_path,
      "dump_gradients": analysis_dump_gradients,
      "analyze_gradients_plot_encoder_layers": analysis_analyze_gradients_plot_encoder_layers,
      "analyze_gradients_plot_log_gradients": analsis_analyze_gradients_plot_log_gradients,
      "calc_search_errors": calc_search_errors,
    }
    if analysis_ground_truth_hdf is None and isinstance(config_builder.variant_params["dependencies"], LibrispeechBPE10025):
      analysis_ground_truth_hdf = LibrispeechBPE10025_CTC_ALIGNMENT.alignment_paths[corpus_keys[0]]

    analysis_opts.update({
      "ground_truth_hdf": analysis_ground_truth_hdf,
    })
  else:
    analysis_opts = None

  pipeline = ReturnnSegmentalAttDecodingPipeline(
    alias=alias,
    config_builder=config_builder,
    checkpoint=checkpoint,
    checkpoint_aliases=checkpoint_aliases,
    beam_sizes=beam_size_list,
    lm_scales=lm_scale_list,
    lm_opts={
      "type": lm_type,
      "add_lm_eos_last_frame": True,
      "add_lm_eos_to_non_blank_end_hyps": add_lm_eos_to_non_blank_end_hyps,
      "eos_scale": lm_eos_scale,
      "alias": lm_alias,
      "checkpoint": lm_checkpoint
    },
    ilm_scales=ilm_scale_list,
    ilm_opts=ilm_opts,
    run_analysis=run_analysis,
    analysis_opts=analysis_opts,
    recog_opts=recog_opts,
    search_alias=f'returnn_decoding{"_pure_torch" if pure_torch else ""}',
    corpus_keys=corpus_keys,
    only_do_analysis=only_do_analysis,
    search_rqmt=recog_rqmt
  )
  pipeline.run()

  return pipeline
