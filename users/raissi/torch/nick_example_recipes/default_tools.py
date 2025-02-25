"""
Defines the external software to be used for the Experiments
"""
from sisyphus import tk

from i6_core.tools.git import CloneGitRepositoryJob
from i6_core.lm.kenlm import CompileKenLMJob

from i6_experiments.common.helpers.text_labels.subword_nmt_bpe import get_returnn_subword_nmt
from i6_experiments.common.tools.sctk import compile_sctk

# python from apptainer/singularity/docker
RETURNN_EXE = tk.Path("/usr/bin/python3", hash_overwrite="GENERIC_RETURNN_LAUNCHER")

#latest on Nov. 8th 2024
MINI_RETURNN_ROOT = CloneGitRepositoryJob(
    "https://github.com/JackTemaki/MiniReturnn", commit="9c26ea0b6c95e598e1031fb572cd405d51ab2e6c"
).out_repository.copy()
MINI_RETURNN_ROOT.hash_overwrite = "LIBRISPEECH_DEFAULT_MINIRETURNN"


I6_MODELS_REPO_PATH = CloneGitRepositoryJob(
    url="https://github.com/rwth-i6/i6_models",
    commit="e22c46a6bdc220a1c7474e7ba916cb9ceb46bf89",
    checkout_folder_name="i6_models",
).out_repository.copy()
I6_MODELS_REPO_PATH.hash_overwrite = "LIBRISPEECH_I6_MODELS_NOVEMBER2024"

#latest on Nov. 8th 2024
I6_NATIVE_OPS_REPO_PATH = CloneGitRepositoryJob(
    url="https://github.com/rwth-i6/i6_native_ops",
    commit="bd105ebb49e2a219f02e2bdc1954b1867e6d4f92",
    checkout_folder_name="i6_native_ops",
).out_repository.copy()
I6_NATIVE_OPS_REPO_PATH.hash_overwrite = "LIBRISPEECH_I6_NATIVE_OPS_NOVEMBER2024"

SCTK_BINARY_PATH = compile_sctk(branch="v2.4.12").copy()  # use last published version
SCTK_BINARY_PATH.hash_overwrite = "LIBRISPEECH_DEFAULT_SCTK_BINARY_PATH"

kenlm_repo = CloneGitRepositoryJob("https://github.com/kpu/kenlm").out_repository.copy()
KENLM_BINARY_PATH = CompileKenLMJob(repository=kenlm_repo).out_binaries.copy()
KENLM_BINARY_PATH.hash_overwrite = "LIBRISPEECH_DEFAULT_KENLM_BINARY_PATH"

SUBWORD_NMT_REPO = get_returnn_subword_nmt(
    commit_hash="5015a45e28a958f800ef1c50e7880c0c9ef414cf",
).copy()
SUBWORD_NMT_REPO.hash_overwrite = "I6_SUBWORD_NMT_V2"
