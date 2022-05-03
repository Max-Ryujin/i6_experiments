from functools import lru_cache
import numpy

from sisyphus import tk

from i6_core.tools.git import CloneGitRepositoryJob
from i6_core.returnn.config import ReturnnConfig

from i6_experiments.common.datasets.librispeech import get_ogg_zip_dict

from i6_experiments.users.rossenbach.datasets.librispeech import get_librispeech_bpe
from i6_experiments.users.rossenbach.setups import returnn_standalone



@lru_cache()
def get_audio_datastream(statistics_ogg_zip, returnn_python_exe, returnn_root):
    # default: mfcc-40-dim
    extract_audio_opts = returnn_standalone.data.audio.AudioFeatureDatastream(
        available_for_inference=True,
        window_len=0.025,
        step_len=0.010,
        num_feature_filters=40,
        features="mfcc")

    audio_datastream = returnn_standalone.data.audio.add_global_statistics_to_audio_features(extract_audio_opts, statistics_ogg_zip,
                                                                 returnn_python_exe=returnn_python_exe,
                                                                 returnn_root=returnn_root)
    return audio_datastream

@lru_cache()
def get_bpe_datastream(bpe_size, is_recog):
    """

    :param bpe_size:
    :param is_recog:
    :return:
    """
    # build dataset
    bpe_settings = get_librispeech_bpe(corpus_key="train-clean-100", bpe_size=bpe_size, unk_label='<unk>')
    bpe_targets = returnn_standalone.data.vocabulary.BpeDatastream(
        available_for_inference=False,
        bpe_settings=bpe_settings,
        use_unk_label=is_recog
    )
    return bpe_targets


def build_training_datasets(returnn_python_exe, returnn_root):
    bpe_size=2000

    ogg_zip_dict = get_ogg_zip_dict("corpora")
    train_clean_100_ogg = ogg_zip_dict['train-clean-100']
    dev_clean_ogg = ogg_zip_dict['dev-clean']
    dev_other_ogg = ogg_zip_dict['dev-other']

    train_bpe_datastream = get_bpe_datastream(bpe_size=bpe_size, is_recog=False)

    audio_datastream = get_audio_datastream(
        statistics_ogg_zip=train_clean_100_ogg,
        returnn_python_exe=returnn_python_exe,
        returnn_root=returnn_root)

    extern_data = {
        'audio_features': audio_datastream.as_returnn_data_opts(),
        'bpe_labels': train_bpe_datastream.as_returnn_data_opts()
    }

    data_map = {"audio_features": ("zip_dataset", "data"),
                "bpe_labels": ("zip_datasets", "classes")}

    train_zip_dataset = returnn_standalone.data.datasets.OggZipDataset(
        path=train_clean_100_ogg,
        audio_opts=audio_datastream.as_returnn_audio_opts(),
        target_opts=train_bpe_datastream.as_returnn_targets_opts(),
        partition_epoch=3,
        seq_ordering="laplace:.1000"
    )
    train_dataset = returnn_standalone.data.datasets.MetaDataset(
        data_map=data_map,
        datasets={"zip_dataset": train_zip_dataset},
        seq_order_control_dataset="zip_dataset"
    )

    from i6_experiments.users.rossenbach.datasets.librispeech import get_mixed_cv_segments
    cv_zip_dataset = returnn_standalone.data.datasets.OggZipDataset(
        path=[dev_clean_ogg, dev_other_ogg],
        audio_opts=audio_datastream.as_returnn_audio_opts(),
        target_opts=train_bpe_datastream.as_returnn_targets_opts(),
        segment_file=get_mixed_cv_segments(),
        seq_ordering="sorted_reverse"
    )
    cv_dataset = returnn_standalone.data.datasets.MetaDataset(
        data_map=data_map,
        datasets={"zip_dataset": cv_zip_dataset},
        seq_order_control_dataset="zip_dataset"
    )

    devtrain_zip_dataset = returnn_standalone.data.datasets.OggZipDataset(
        path=train_clean_100_ogg,
        audio_opts=audio_datastream.as_returnn_audio_opts(),
        target_opts=train_bpe_datastream.as_returnn_targets_opts(),
        segment_file=get_mixed_cv_segments(),
        seq_ordering="sorted_reverse",
        subset=3000,
    )
    devtrain_dataset = returnn_standalone.data.datasets.MetaDataset(
        data_map=data_map,
        datasets={"zip_dataset": devtrain_zip_dataset},
        seq_order_control_dataset="zip_dataset"
    )

    return train_dataset, cv_dataset, devtrain_dataset, extern_data



def get_config():

    # changing these does not change the hash
    post_config = {
        'use_tensorflow': True,
        'tf_log_memory_usage': True,
        'cleanup_old_models': True,
        'log_batch_size': True,
        'debug_print_layer_output_template': True,
    }

    wup_start_lr = 0.0003
    initial_lr = 0.0008

    learning_rates = list(numpy.linspace(wup_start_lr, initial_lr, num=10))

    config = {
        'gradient_clip': 0,
        'optimizer': {'class': 'Adam'},
        'optimizer_epsilon': 1e-8,
        'accum_grad_multiple_step': 2,
        'gradient_noise': 0.0,
        'learning_rates': learning_rates,
        'min_learning_rate': 0.00001,
        'learning_rate_control': "newbob_multi_epoch",
        'learning_rate_control_relative_error_relative_lr': True,
        'learning_rate_control_min_num_epochs_per_new_lr': 3,
        'use_learning_rate_control_always': True,
        'newbob_multi_num_epochs': 3,
        'newbob_multi_update_interval': 1,
        'newbob_learning_rate_decay': 0.9,
        'batch_size': 10000,
        'max_seq_len': {'bpe_labels': 75},  # just for security
        'max_seqs': 200,
        # 'truncation': -1
    }


    from .prototype_network import ConvBLSTMEncoder, static_decoder

    # stage 1
    import time
    start = time.time()
    stage_nets = []
    for i in range(5):
        encoder = ConvBLSTMEncoder(audio_feature_key="audio_features", target_label_key="bpe_labels", num_lstm_layers=2 + i)
        encoder_dict = {'encoder': {'class': 'subnetwork', 'from': [], 'subnetwork': encoder.make_root_net_dict()}}
        encoder_dict['encoder']['ConvBLSTMEncoder']['specaug_block']['eval_layer'].pop("kind")
        print("constructing took %f" % (time.time() - start))
        stage_net = {**encoder_dict, **static_decoder}
        stage_nets.append(stage_net)
    network_dict = {
        1: stage_nets[0],
        6: stage_nets[1],
        11: stage_nets[2],
        16: stage_nets[3],
        21: stage_nets[4],
    }

    from .specaugment_clean import get_funcs

    returnn_config = ReturnnConfig(
        config=config,
        post_config=post_config,
        staged_network_dict=network_dict,
        python_prolog=get_funcs(),
    )
    return returnn_config

def test():
    returnn_exe = tk.Path("/u/rossenbach/bin/returnn_tf2.3_launcher_custom.sh", hash_overwrite="GENERIC_RETURNN_LAUNCHER")
    returnn_root = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn",
                                         commit="82dee6281a92a637a57cd97e0d1655eeb10e9a2c").out_repository


    returnn_config = get_config()
    train_dataset, cv_dataset, devtrain_dataset, extern_data = build_training_datasets(
        returnn_python_exe=returnn_exe, returnn_root=returnn_root)

    returnn_config.config["extern_data"] = extern_data
    returnn_config.config["train"] = train_dataset.as_returnn_opts()
    returnn_config.config["dev"] = cv_dataset.as_returnn_opts()

    from i6_core.returnn.training import ReturnnTrainingJob

    default_rqmt = {
        'mem_rqmt': 15,
        'time_rqmt': 80,
        'log_verbosity': 5,
        'returnn_python_exe': returnn_exe,
        'returnn_root': returnn_root,
    }

    train_job = ReturnnTrainingJob(
        returnn_config=returnn_config,
        num_epochs=250,
        **default_rqmt
    )
    prefix_name = "new_training_test"
    train_job.add_alias(prefix_name + "/training")
    tk.register_output(prefix_name + "/learning_rates", train_job.out_learning_rates)


