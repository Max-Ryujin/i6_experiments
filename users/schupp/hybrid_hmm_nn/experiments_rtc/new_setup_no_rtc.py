# TODO: package, make imports smaller
from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.pipeline import librispeech_hybrid_tim_refactor as system
from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args import conformer_config_returnn_baseargs as experiment_config_args
from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args import conformer_rasr_config_maker as rasr_config_args_maker
from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args import conformer_returnn_dict_network_generator
from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.pipeline import hybrid_job_dispatcher as job_dispatcher

from recipe.i6_core.returnn import ReturnnConfig, ReturnnRasrTrainingJob

from sisyphus import gs

import inspect

OUTPUT_PATH = "conformer/test_new_nortc/"
gs.ALIAS_AND_OUTPUT_SUBDIR = OUTPUT_PATH
NAME = "testrun"

# All possible inputs that 'diversify' a config
# - NAME
# - OUTPUT_PATH 
# - config_base_args
# - network_args:
#   - sampling_func_args
#   - ff1_func_args
#   - ff2_func_args
#   - sa_func_args
#   - conv_func_args
#   - conformer_default_args_00
# - train:
#   - returnn_train_post_config_00
#   - returnn_rasr_args_defaults_00

# Start system:
# - register alignments and features ...
system = system.LibrispeechHybridSystemTim()

# Make a returnn config
config_base_args = experiment_config_args.config_baseline_00

train_corpus_key = 'train-other-960'

system.create_rasr_am_config(train_corpus_key=train_corpus_key)

# Conformer generation code ( should be moved somewhere else )

network = conformer_returnn_dict_network_generator.make_conformer_00(

  # sampling args
  sampling_func_args = experiment_config_args.sampling_default_args_00,

  # Feed forward args, both the same by default
  ff1_func_args = experiment_config_args.ff_default_args_00,
  ff2_func_args = experiment_config_args.ff_default_args_00,

  # Self attention args
  sa_func_args = experiment_config_args.sa_default_args_00,

  # Conv mod args
  conv_func_args = experiment_config_args.conv_default_args_00,

  # Shared model args
  shared_model_args = experiment_config_args.shared_network_args_00,

  # Conformer args
  **experiment_config_args.conformer_default_args_00,

  print_net = True
)

returnn_train_config : ReturnnConfig = job_dispatcher.make_returnn_train_config_old(
  network = network,
  config_base_args=config_base_args,
  post_config_args=experiment_config_args.returnn_train_post_config_00
)

# We test the construction now to avaoid error when running on cluster
#job_dispatcher.test_net_contruction(returnn_train_config)

returnn_rasr_config_args : dict = rasr_config_args_maker.get_returnn_rasr_args(
  system, 
  train_corpus_key=train_corpus_key,
  **experiment_config_args.returnn_rasr_args_defaults_00
)

train_job : ReturnnRasrTrainingJob = job_dispatcher.make_and_register_returnn_rasr_train(
    returnn_train_config,
    returnn_rasr_config_args,
    output_path=OUTPUT_PATH
)

rec_corpus = "dev-other"
# Prepare args for rasr recog
system.init_rasr_am_lm_config_recog(
  recog_corpus_key=rec_corpus
)

job_dispatcher.make_and_register_returnn_rasr_search(
  system=system,
  returnn_train_config=returnn_train_config,
  train_job=train_job,
  recog_corpus_key=rec_corpus,
  feature_name="gammatone",
  limit_eps=experiment_config_args.returnn_train_post_config_00["cleanup_old_models"]["keep"],
  exp_name=NAME
)
