import copy
from sisyphus import gs, tk

from i6_core.tools.git import CloneGitRepositoryJob
from i6_core.features import FilterbankJob, samples_flow

from i6_experiments.users.hilmes.common.setups.rasr.util import RasrSteps

from .corpus_data import get_corpus_data_inputs
from i6_experiments.users.hilmes.common.tedlium2.hybrid.baseline_args import get_log_mel_feature_extraction_args
from .parakeet_args import get_nn_args
from i6_experiments.users.hilmes.modules.onnx_precomputed_hybrid_system import OnnxPrecomputedHybridSystem


def run_gmm_system():
    from i6_experiments.users.hilmes.common.tedlium2.gmm.baseline_config import (
        run_tedlium2_common_baseline,
    )

    system = run_tedlium2_common_baseline()
    flow = samples_flow(dc_detection=False, input_options={"block-size": "1"}, scale_input=2 ** -15)
    system.extract_features(
        feat_args={"samples": {"feature_flow": flow, "port_name": "samples"}},
        corpus_list=system.dev_corpora + system.test_corpora,
    )
    return system


def run_tedlium2_torch_parakeet():
    prefix = "experiments/tedlium2/hybrid/parakeet"
    gs.ALIAS_AND_OUTPUT_SUBDIR = prefix

    gmm_system = run_gmm_system()

    rasr_init_args = copy.deepcopy(gmm_system.rasr_init_args)
    rasr_init_args.feature_extraction_args = get_log_mel_feature_extraction_args()

    feature_extraction_args = rasr_init_args.feature_extraction_args["fb"]
    feature_extraction_class = FilterbankJob

    (
        nn_train_data_inputs,
        nn_cv_data_inputs,
        nn_devtrain_data_inputs,
        nn_dev_data_inputs,
        nn_test_data_inputs,
    ) = get_corpus_data_inputs(
        gmm_system, feature_extraction_args, feature_extraction_class, alias_prefix=prefix
    )
    steps = RasrSteps()
    steps.add_step("extract", rasr_init_args.feature_extraction_args)
    gmm_system.run(steps)
    nn_args = get_nn_args(num_epochs=250)
    nn_dev_data_inputs["dev"].feature_flow = gmm_system.feature_flows["dev"]

    nn_steps = RasrSteps()
    nn_steps.add_step("nn", nn_args)

    # image only, so just python3
    returnn_exe = tk.Path("/usr/bin/python3", hash_overwrite="GENERIC_RETURNN_LAUNCHER")
    rasr_binary = tk.Path(
        "/work/asr4/hilmes/dev/rasr_onnx_115_24_01_24/arch/linux-x86_64-standard")
    rasr_binary.hash_overwrite = "TEDLIUM2_DEFAULT_RASR_BINARY_PATH_HUBERT"

    returnn_root = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/returnn",
        commit="b69cbd6f06379af830e92fda91b8c8ccf3000147",
    ).out_repository.copy()
    returnn_root.hash_overwrite = "TEDLIUM_RETURNN_PARAKEET"

    tedlium_nn_system = OnnxPrecomputedHybridSystem(
        returnn_root=returnn_root,
        returnn_python_exe=returnn_exe,
        blas_lib=None,
        rasr_arch="linux-x86_64-standard",
        rasr_binary_path=rasr_binary,
    )
    tedlium_nn_system.init_system(
        rasr_init_args=rasr_init_args,
        train_data=nn_train_data_inputs,
        cv_data=nn_cv_data_inputs,
        devtrain_data=nn_devtrain_data_inputs,
        dev_data=nn_dev_data_inputs,
        test_data=nn_test_data_inputs,
        train_cv_pairing=[tuple(["train.train", "dev.cv"])],
    )
    tedlium_nn_system.run(nn_steps)

    gs.ALIAS_AND_OUTPUT_SUBDIR = ""

