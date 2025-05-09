import os

import numpy as np
import time
import torch
import torchaudio
import subprocess
import multiprocessing as mp
from ..vocoder.simple_gl.blstm_gl_predictor import Model
from ..tts_shared.corpus import Corpus, Recording, Segment

from returnn.datasets.util.hdf import SimpleHDFWriter

# global environment where thread count for ffmpeg is set to 2
ENVIRON = os.environ.copy()
ENVIRON["OMP_NUM_THREADS"] = "2"


def forward_init_hook(run_ctx, **kwargs):
    run_ctx.norm_mean = kwargs["norm_mean"]
    run_ctx.norm_std = kwargs["norm_std_dev"]
    simple_gl_checkpoint = kwargs["gl_net_checkpoint"]
    simple_gl_net_config = kwargs["gl_net_config"]
    n_iter = kwargs.get("gl_iter", 32)
    momentum = kwargs.get("gl_momentum", 0.99)
    run_ctx.noise_scale = kwargs.get("glowtts_noise_scale", 1.0)

    run_ctx.create_plots = kwargs.get("create_plots", True)
    run_ctx.store_log_mels = kwargs.get("store_log_mels", False)
    if run_ctx.store_log_mels is True:
        # hardcoded dim for now
        run_ctx.hdf_writer = SimpleHDFWriter("log_mels.hdf", dim=80, ndim=2)

    run_ctx.corpus = Corpus()
    run_ctx.corpus.name = None

    run_ctx.pool = mp.Pool(processes=kwargs.get("num_pool_processes", 4))

    assert isinstance(simple_gl_net_config, dict)

    run_ctx.gl_model = Model(config=simple_gl_net_config)
    checkpoint_state = torch.load(
        simple_gl_checkpoint,
        map_location=run_ctx.device,
    )
    run_ctx.gl_model.load_state_dict(checkpoint_state["model"])
    run_ctx.gl_model.to(device=run_ctx.device)

    num_freq = 800
    run_ctx.griffin_lim = torchaudio.transforms.GriffinLim(
        num_freq,
        n_iter=n_iter,
        win_length=int(0.05 * 16000),
        hop_length=int(0.0125 * 16000),
        power=1.0,
        momentum=momentum,
    )
    run_ctx.griffin_lim.to(device=run_ctx.device)

    import os
    if not os.path.exists("audio_files"):
        os.mkdir("audio_files")


def forward_finish_hook(run_ctx, **kwargs):
    run_ctx.corpus.dump("out_corpus.xml.gz")
    if run_ctx.store_log_mels is True:
        run_ctx.hdf_writer.close()


MAX_WAV_VALUE = 32768.0


def save_wav(wav, path, sr, peak_normalization=True):
    from scipy.io import wavfile
    if peak_normalization:
        wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    else:
        wav *= 32767
    wavfile.write(path, sr, wav.astype(np.int16))


def save_ogg(args):
    """
    :param args: wav, path and sr
    """
    wav, path, sr = args
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    p1 = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-threads", "1", "-f", "s16le", "-ar", "%i" % sr, "-i", "pipe:0", "-c:a", "libvorbis", "-q", "3.0", path],
                          stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE,
                          env=ENVIRON)
    p1.communicate(input=wav.astype(np.int16).tobytes())
    p1.terminate()


def save_plot(args):
    from matplotlib import pyplot as plt
    log_mel, tag = args
    plt.imshow(log_mel)
    plt.gca().invert_yaxis()
    plt.savefig(f"audio_files/{tag.replace('/', '_')}.png")



def forward_step(*, model: Model, data, run_ctx, **kwargs):
    phonemes = data["phonemes"]  # [B, N] (sparse)
    phonemes_len = data["phonemes:size1"]  # [B]
    speaker_labels = data["speaker_labels"]  # [B, 1] (sparse)

    tags = data["seq_tag"]

    forward_start = time.time()
    (log_mels, z_m, z_logs, logdet, z_mask, y_lengths), (x_m, x_logs, x_mask), (attn, logw, logw_, w_ceil) = model(
        phonemes,
        phonemes_len,
        g=speaker_labels,
        gen=True,
        noise_scale=run_ctx.noise_scale,
        length_scale=1.0,
    )
    log_mels.detach()
    forward_time = time.time() - forward_start

    vocoder_model_start = time.time()
    _, linears = run_ctx.gl_model(log_mels.transpose(1, 2), y_lengths)
    linears = linears.transpose(1, 2)
    linears.detach()
    vocoder_net_time = time.time() - vocoder_model_start

    from matplotlib import pyplot as plt

    pool_args = []
    plot_pool_args = []

    iterate_start = time.time()
    for linear, log_mel, length, tag in zip(linears, log_mels, y_lengths, tags):
        corpus_name, recording_name, segment_name = tag.split("/")
        if run_ctx.corpus.name is None:
            run_ctx.corpus.name = corpus_name
        wave = run_ctx.griffin_lim(linear[:, :length]).cpu().numpy()
        audio_path = f"audio_files/{tag.replace('/', '_')}.ogg"
        pool_args.append((wave, audio_path, 16000))

        segment = Segment()
        segment.name = segment_name
        segment.start = 0
        segment.end = len(wave) / 16000.0

        recording = Recording()
        recording.name = recording_name
        recording.audio = audio_path
        recording.add_segment(segment)

        run_ctx.corpus.add_recording(recording)

        if run_ctx.create_plots:
            plot_pool_args.append((log_mel[:, :length].cpu().numpy(), tag))
            #plt.imshow()
            #plt.gca().invert_yaxis()
            #plt.savefig(f"audio_files/{tag.replace('/', '_')}.png")

        if run_ctx.store_log_mels is True:
            run_ctx.hdf_writer.insert_batch(inputs=np.expand_dims(log_mel.transpose(0, 1).cpu().numpy()[:length], 0), seq_len=[length], seq_tag=[tag])
    iterate_time = time.time() - iterate_start

    save_ogg_start = time.time()
    run_ctx.pool.map(save_ogg, pool_args)
    save_ogg_time = time.time() - save_ogg_start

    save_plot_time = 0.0
    if run_ctx.create_plots:
        save_plot_start = time.time()
        run_ctx.pool.map(save_plot, plot_pool_args)
        save_plot_time = time.time() - save_plot_start

    print(f"forward_time: {forward_time:.3f}")
    print(f"vocoder_net_time: {vocoder_net_time:.3f}")
    print(f"iterate_time (+G&L): {iterate_time:.3f}")
    print(f"save_ogg_time: {save_ogg_time:.3f}")
    if run_ctx.create_plots:
        print(f"save_plot_time: {save_plot_time:.3f}")

