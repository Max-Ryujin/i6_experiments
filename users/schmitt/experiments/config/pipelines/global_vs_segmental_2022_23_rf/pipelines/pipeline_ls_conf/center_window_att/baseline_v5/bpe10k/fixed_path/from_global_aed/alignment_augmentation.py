from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att.baseline_v5 import (
  get_config_builder,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import (
  train, recog, realign
)

from i6_core.returnn.training import PtCheckpoint
from sisyphus import Path


def run_exps():
  for model_alias, config_builder in get_config_builder.center_window_att_baseline_rf(
          win_size_list=(1,),
          blank_decoder_version=8,
          use_att_ctx_in_state=False,
          use_weight_feedback=False,
  ):
    for num_iterations in (2, 10,):
      for train_alias, checkpoint in train.train_center_window_att_viterbi_import_global_tf(
              alias=model_alias,
              config_builder=config_builder,
              n_epochs_list=(300,),
              const_lr_list=(1e-4,),
              alignment_augmentation_opts={"max_shift": 1, "num_iterations": num_iterations},
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
          checkpoint_aliases=("best-4-avg",),
          run_analysis=True,
          analyze_gradients=True,
        )
