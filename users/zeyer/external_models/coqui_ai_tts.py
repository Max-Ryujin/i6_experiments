"""
Coqui-ai TTS: https://github.com/coqui-ai/TTS
own fork: https://github.com/albertz/coqui-ai-tts
"""

from __future__ import annotations
import os
import sys
import functools
from typing import Optional, TypeVar


_my_dir = os.path.dirname(__file__)
_base_dir = functools.reduce(lambda p, _: os.path.dirname(p), range(4), _my_dir)
_sis_dir = os.path.dirname(_base_dir) + "/tools/sisyphus"

T = TypeVar("T")


def _setup():
    # In case the user started this script directly.
    if not globals().get("__package__"):
        globals()["__package__"] = "i6_experiments.users.zeyer.external_models"
        if _base_dir not in sys.path:
            sys.path.append(_base_dir)
        if _sis_dir not in sys.path:
            sys.path.append(_sis_dir)


_setup()


from sisyphus import Job, Task, Path, tk


def py():
    """
    demo to run this directly with Sisyphus
    """
    path = download_model("tts_models/multilingual/multi-dataset/your_tts")
    tk.register_output("external_models/coqui_ai_tts/your_tts", path)


def download_model(model_name: str, *, tts_repo_dir: Optional[Path] = None):
    """
    :param model_name: for example "tts_models/multilingual/multi-dataset/your_tts"
    :param tts_repo_dir: if not specified, uses :func:`get_default_tts_repo_dir`
    """
    if tts_repo_dir is None:
        tts_repo_dir = get_default_tts_repo_dir()
    download_job = DownloadModel(model_name=model_name, tts_repo_dir=tts_repo_dir)
    return download_job.out_tts_data_dir


class DownloadModel(Job):
    """
    Downloads a model via the Coqui-ai TTS api.
    See: https://github.com/coqui-ai/TTS?tab=readme-ov-file#-python-api

    To use it later, set the ``TTS_HOME`` env var to ``out_tts_data_dir.get_path()``.
    Also, set the env var ``COQUI_TOS_AGREED=1``.

    Thus, it requires the TTS repo (with TTS source code; e.g. via :func:`get_default_tts_repo_dir`),
    and then will use that Python API.
    Thus, it requires the TTS dependencies: https://github.com/coqui-ai/TTS/blob/dev/requirements.txt
    Specifically, that should be:
    - torch
    - coqpit
    - trainer (might need --ignore-requires-python)
    - tqdm
    - pysbd
    - mutagen
    - pandas
    - anyascii
    - inflect
    - bangla
    - bnnumerizer
    - bnunicodenormalizer
    - gruut
    - jamo
    - jieba
    - pypinyin
    """

    def __init__(self, *, model_name: str, tts_repo_dir: Path):
        """
        :param model_name: for example "tts_models/multilingual/multi-dataset/your_tts"
        :param tts_repo_dir:
        """
        super().__init__()
        self.model_name = model_name
        self.tts_repo_dir = tts_repo_dir
        self.out_tts_data_dir = self.output_path("tts_home", directory=True)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import sys
        import os
        import tempfile
        import shutil
        from sisyphus import gs

        sys.path.insert(0, self.tts_repo_dir.get_path())

        with tempfile.TemporaryDirectory(prefix=gs.TMP_PREFIX) as tmp_dir:
            print("using temp-dir:", tmp_dir)

            os.environ["TTS_HOME"] = tmp_dir
            os.environ["COQUI_TOS_AGREED"] = "1"

            from TTS.api import TTS

            tts = TTS(model_name=self.model_name, progress_bar=False)
            assert str(tts.manager.output_prefix) == tmp_dir + "/tts"
            assert os.path.isdir(tts.manager.output_prefix)
            dir_content = os.listdir(tts.manager.output_prefix)
            print(".../tts dir content:", dir_content)
            assert dir_content  # non-empty
            shutil.copytree(tts.manager.output_prefix, self.out_tts_data_dir.get_path() + "/tts")


def get_default_tts_repo_dir() -> Path:
    """
    :return: upstream, via :func:`get_upstream_tts_git_repo`
    """
    return get_upstream_tts_git_repo()


def get_upstream_tts_git_repo() -> Path:
    """
    :return: upstream, via :class:`CloneGitRepositoryJob`, from https://github.com/coqui-ai/TTS.git
    """
    from i6_core.tools.git import CloneGitRepositoryJob

    clone_job = CloneGitRepositoryJob(
        "https://github.com/coqui-ai/TTS.git", commit="dbf1a08a0d4e47fdad6172e433eeb34bc6b13b4e"
    )
    return clone_job.out_repository


def _demo():
    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model-name", default="tts_models/multilingual/multi-dataset/your_tts")
    arg_parser.add_argument("--model-dir", default="output/external_models/coqui_ai_tts/your_tts")
    arg_parser.add_argument("--tts-repo-dir")
    arg_parser.add_argument("--device", default="cuda")
    arg_parser.add_argument("--language")
    arg_parser.add_argument("--seed", type=int, default=42)
    args = arg_parser.parse_args()

    print(f"{os.path.basename(__file__)} demo, {args=}")

    import random
    import torch
    import numpy as np

    dev = torch.device(args.device)
    rnd = random.Random(args.seed)
    torch.random.manual_seed(rnd.randint(0, 2**32 - 1))

    try:
        import lovely_tensors

        lovely_tensors.monkey_patch()
    except ImportError:
        pass

    model_name = args.model_name
    model_dir = args.model_dir
    assert os.path.exists(model_dir), f"model dir does not exist: {model_dir}"
    assert os.path.exists(model_dir + "/tts"), f"model dir does not exist: {model_dir}/tts"

    tts_repo_dir = args.tts_repo_dir
    if not tts_repo_dir:
        tts_repo_dir = get_default_tts_repo_dir().get_path()
        print("Using default TTS repo dir:", tts_repo_dir)

    sys.path.insert(0, tts_repo_dir)
    os.environ["TTS_HOME"] = model_dir
    os.environ["COQUI_TOS_AGREED"] = "1"

    print("Importing TTS...")

    from TTS.api import TTS
    from TTS.utils.manage import ModelManager
    from TTS.utils.synthesizer import synthesis
    from TTS.utils.audio.numpy_transforms import save_wav
    from TTS.tts.models.vits import Vits

    def _disallowed_create_dir_and_download_model(*_args, **_kwargs):
        raise RuntimeError(
            f"Disallowed create_dir_and_download_model({_args}, {_kwargs}),"
            f" model {model_name} not found, model dir {model_dir} not valid? "
        )

    # patch to avoid any accidental download
    assert hasattr(ModelManager, "create_dir_and_download_model")
    ModelManager.create_dir_and_download_model = _disallowed_create_dir_and_download_model

    print("Loading TTS model...")
    tts = TTS(model_name=model_name, progress_bar=False)
    tts.to(dev)

    # See tts.tts() func to how it generates audio.
    # This is a high-level wrapper, but we want to call it more directly to be able to use it in batch mode.

    tts_model: Vits = tts.synthesizer.tts_model  # typing Vits is just an example
    tts_config = tts.synthesizer.tts_config
    sample_rate = tts.synthesizer.output_sample_rate

    print(f"{type(tts_model) = }")

    speakers = tts.speakers
    print(f"speakers: {speakers}")
    speaker = None
    speaker_id = None
    speaker_embedding = None
    if tts_model.speaker_manager:
        speaker_id = rnd.randint(0, tts_model.speaker_manager.num_embeddings - 1)
        print("random speaker id:", speaker_id)
        emb = list(tts_model.speaker_manager.embeddings.values())[speaker_id]
        speaker = emb["name"]
        speaker_embedding = np.array(emb["embedding"])  # [emb_dim]
    print(f"speaker: {speaker!r}")

    print("emb_g:", tts_model.emb_g if hasattr(tts_model, "emb_g") else None)  # speaker embedding
    print(f"{tts_config.use_d_vector_file = }")
    if tts_model.speaker_manager:
        print(f"{len(tts_model.speaker_manager.embeddings) = }")
        # print("embedding keys:", list(tts_model.speaker_manager.embeddings.keys()))
        # for emb in tts_model.speaker_manager.embeddings.values():
        #     print(f"  name={emb['name']!r}, keys:{list(emb.keys())}, dim:{len(emb['embedding'])}")

    language = args.language
    if tts_model.language_manager:
        languages = tts_model.language_manager.language_names
        print(f"{len(languages) = }, {languages = }")
        if language is None:
            language = rnd.choice(languages)
            print("Picked random language:", language)
        else:
            assert language in languages, f"language {language} not in {languages}"
    else:
        print("No language manager")

    print(f"{hasattr(tts_model, "synthesize") = }")  # if False, then it will call the synthesis func
    assert not hasattr(tts_model, "synthesize")  # we assume that we can call the synthesis func
    print(f"{sample_rate = }")

    # could also take this from config test_sentences
    text = "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent."

    # This is maybe Vits specific:
    print(f"{tts_model.length_scale = }")
    tts_model.length_scale = 1.0  # overwrite (testing...)
    print(f"{tts_model.inference_noise_scale = }")
    print(f"{tts_model.inference_noise_scale_dp = }")

    # Test high-level API.
    wav = tts.tts(text, speaker=speaker, language=language)
    print("tts() output:", type(wav))
    # if tensor convert to numpy
    if torch.is_tensor(wav):
        wav = wav.cpu().numpy()
    if isinstance(wav, list):
        wav = np.array(wav)
    assert isinstance(wav, np.ndarray)
    print("shape:", wav.shape)
    save_wav(wav=wav, path="demo.wav", sample_rate=sample_rate)

    # Generate another one. How much randomness?
    wav = tts.tts(text, speaker=speaker, language=language)
    print("tts() output:", type(wav))
    # if tensor convert to numpy
    if torch.is_tensor(wav):
        wav = wav.cpu().numpy()
    if isinstance(wav, list):
        wav = np.array(wav)
    assert isinstance(wav, np.ndarray)
    print("shape:", wav.shape)
    save_wav(wav=wav, path="demo2.wav", sample_rate=sample_rate)

    if tts_model.speaker_manager:
        if tts_config.use_d_vector_file:
            speaker_id = None  # the model expects speaker_embedding only, not speaker_id
        else:
            speaker_embedding = None  # speaker_id is used

    # outputs = synthesis(
    #     model=tts_model,
    #     text=txt,
    #     CONFIG=tts_config,
    #     use_cuda=use_cuda,
    #     speaker_id=speaker_id,
    #     style_wav=style_wav,
    #     style_text=style_text,
    #     use_griffin_lim=use_gl,
    #     d_vector=speaker_embedding,
    #     language_id=language_id,
    # )
    # waveform = outputs["wav"]

    # See also TTS.utils.synthesizer.synthesis().

    # convert text to sequence of token IDs
    text_inputs = np.asarray(
        tts_model.tokenizer.text_to_ids(text, language=language),
        dtype=np.int32,
    )  # [T_in]
    text_inputs = torch.tensor(text_inputs, dtype=torch.int32, device=dev)[None]  # [B,T_in]
    print(f"{text_inputs.shape = }")
    text_inputs_lens = torch.tensor([text_inputs.shape[1]], device=dev)  # [B]
    speaker_id = torch.tensor(speaker_id, device=dev)[None] if speaker_id is not None else None  # [B]
    speaker_embedding = torch.tensor(speaker_embedding, device=dev, dtype=torch.float32) if speaker_embedding is not None else None
    speaker_embedding = speaker_embedding[None] if speaker_embedding is not None else None  # [B, emb_dim]
    language_id = torch.tensor(tts_model.language_manager.name_to_id[language], device=dev)[None] if language else None  # [B]

    outputs = tts_model.inference(
        text_inputs,
        aux_input={
            "x_lengths": text_inputs_lens,
            "speaker_ids": speaker_id,
            "d_vectors": speaker_embedding,
            "language_ids": language_id,
        },
    )
    print(outputs)

    model_outputs = outputs["model_outputs"]  # for Vits: [B, 1, T_wav]
    y_mask = outputs["y_mask"]  # before the final waveform_decoder
    print(f"{model_outputs.shape = }")
    print(f"{y_mask.shape = }")

    # convert outputs to numpy. select first batch
    model_outputs = model_outputs[0].cpu().squeeze().numpy()  # [T_wav]
    assert model_outputs.ndim == 1
    wav = model_outputs
    save_wav(wav=wav, path="demo3.wav", sample_rate=sample_rate)

    # if hasattr(tts_model, "waveform_decoder"):
    #     print("waveform_decoder:", tts_model.waveform_decoder)


if __name__ == "__main__":
    _demo()
