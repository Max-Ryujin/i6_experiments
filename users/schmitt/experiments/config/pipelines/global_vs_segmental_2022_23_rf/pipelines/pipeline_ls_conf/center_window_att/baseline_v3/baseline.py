from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att.baseline_v3 import (
  get_config_builder
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import (
  train, recog, realign
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.label_singletons import LibrispeechBPE10025_CTC_ALIGNMENT

from i6_core.returnn.training import PtCheckpoint
from sisyphus import Path


def run_exps():
  # ------------------------------------- from-scratch Viterbi training -------------------------------------

  # for model_alias, config_builder in baseline.center_window_att_baseline_rf(
  #   win_size_list=(5,),
  # ):
  #   for train_alias, checkpoint in train.train_center_window_att_viterbi_from_scratch(
  #     alias=model_alias,
  #     config_builder=config_builder,
  #     n_epochs_list=(500,),
  #   ):
  #     recog.center_window_returnn_frame_wise_beam_search(
  #       alias=train_alias,
  #       config_builder=config_builder,
  #       checkpoint=checkpoint,
  #     )

  # ------------------------------------- best models: KEEP! -------------------------------------

  for win_size in (1, 5, 11, 25, 129, 499):
    for model_alias, config_builder in get_config_builder.center_window_att_baseline_rf(
            win_size_list=(win_size,), use_weight_feedback=not win_size == 1,
    ):
      for train_alias, checkpoint in train.train_center_window_att_viterbi_import_global_tf(
              alias=model_alias,
              config_builder=config_builder,
              n_epochs_list=(300,),
              const_lr_list=(1e-4,),
      ):
        recog.center_window_returnn_frame_wise_beam_search(
          alias=train_alias,
          config_builder=config_builder,
          checkpoint=checkpoint,
        )
        recog.center_window_returnn_frame_wise_beam_search(
          alias=train_alias,
          config_builder=config_builder,
          checkpoint=checkpoint,
          checkpoint_aliases=("last",),
          run_analysis=True,
          analyze_gradients=True,
        )
        recog.center_window_returnn_frame_wise_beam_search(
          alias=train_alias,
          config_builder=config_builder,
          checkpoint=checkpoint,
          checkpoint_aliases=("last",),
          lm_type="trafo",
          lm_scale_list=(0.54,),
          ilm_type="mini_att",
          ilm_scale_list=(0.4,),
          subtract_ilm_eos_score=True,
          use_recombination="sum",
          corpus_keys=("dev-other", "test-other"),
          beam_size_list=(84,),
        )
        if win_size in (1, 5):
          recog.center_window_returnn_frame_wise_beam_search(
            alias=train_alias,
            config_builder=config_builder,
            checkpoint=checkpoint,
            checkpoint_aliases=("last",),
            use_recombination="sum",
            corpus_keys=("test-other",),
            beam_size_list=(12,),
          )
        if win_size == 1:
          for corpus_key in [
            "dev-other_0.1-5.1",
            "dev-other_5.1-10.1",
            "dev-other_10.1-15.1",
            "dev-other_15.1-20.1",
            "dev-other_20.1-25.1",
            "dev-other_25.1-30.1",
            "dev-other_30.1-35.1",
          ]:
            recog.center_window_returnn_frame_wise_beam_search(
              alias=train_alias,
              config_builder=config_builder,
              checkpoint=checkpoint,
              checkpoint_aliases=("last",),
              corpus_keys=(corpus_key,),
            )
        if win_size in (1, 5, 25, 129):
          for concat_num in (2, 4, 8, 10, 20, 30):
            if concat_num in (20, 30):
              batch_size = 40_000
              time_rqmt = 6
            else:
              batch_size = 15_000
              time_rqmt = 4
            recog.center_window_returnn_frame_wise_beam_search(
              alias=train_alias,
              config_builder=config_builder,
              checkpoint=checkpoint,
              checkpoint_aliases=("last",),
              batch_size=batch_size,
              concat_num=concat_num,
              time_rqmt=time_rqmt
            )
        recog.center_window_returnn_frame_wise_beam_search(
          alias=train_alias,
          config_builder=config_builder,
          checkpoint=checkpoint,
          checkpoint_aliases=(f"last",),
          run_analysis=True,
          only_do_analysis=True,
          analyze_gradients=True,
          analsis_analyze_gradients_plot_log_gradients=False,
          analysis_analyze_gradients_plot_encoder_layers=False,
          analysis_ground_truth_hdf=LibrispeechBPE10025_CTC_ALIGNMENT.alignment_paths["train"],
          att_weight_seq_tags=[
            "train-other-960/1246-124548-0042/1246-124548-0042",
            "train-other-960/40-222-0033/40-222-0033",
            "train-other-960/103-1240-0038/103-1240-0038",
          ],
          corpus_keys=("train",),
        )

        if win_size == 129:
          recog.center_window_returnn_frame_wise_beam_search(
            alias=train_alias,
            config_builder=config_builder,
            checkpoint=checkpoint,
            checkpoint_aliases=(f"last",),
            run_analysis=True,
            analyze_gradients_search=True,
            analsis_analyze_gradients_plot_log_gradients=False,
            analysis_analyze_gradients_plot_encoder_layers=False,
            analysis_ground_truth_hdf=LibrispeechBPE10025_CTC_ALIGNMENT.alignment_paths["dev-other"],
            att_weight_seq_tags=[
              "dev-other/7697-105815-0015/7697-105815-0015",
            ],
            corpus_keys=("dev-other",),
          )

  for model_alias, config_builder in get_config_builder.center_window_att_baseline_rf(
          win_size_list=(5,),
  ):
    for train_alias, checkpoint in train.train_center_window_att_viterbi_import_global_tf(
            alias=model_alias,
            config_builder=config_builder,
            n_epochs_list=(300,),
            const_lr_list=(1e-4,),
    ):
      recog.center_window_returnn_frame_wise_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        checkpoint_aliases=("last",),
        run_analysis=True,
      )
      recog.center_window_returnn_frame_wise_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        checkpoint_aliases=("last",),
        length_normalization_exponent=1.0,
      )
      for concat_num in (8,):
        recog.center_window_returnn_frame_wise_beam_search(
          alias=train_alias,
          config_builder=config_builder,
          checkpoint=checkpoint,
          checkpoint_aliases=("last",),
          run_analysis=True,
          analyze_gradients=True,
          concat_num=concat_num,
          batch_size=30_000,
          att_weight_seq_tags=[
            "dev-other/116-288045-0008/116-288045-0008;dev-other/116-288045-0009/116-288045-0009;dev-other/116-288045-0010/116-288045-0010;dev-other/116-288045-0011/116-288045-0011;dev-other/116-288045-0012/116-288045-0012;dev-other/116-288045-0013/116-288045-0013;dev-other/116-288045-0014/116-288045-0014;dev-other/116-288045-0015/116-288045-0015",
          ]
        )
