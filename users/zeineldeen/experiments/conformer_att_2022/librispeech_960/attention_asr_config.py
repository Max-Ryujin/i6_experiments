import numpy
import copy
from dataclasses import asdict

from i6_experiments.users.zeineldeen.models.asr.encoder.args import (
    EncoderArgs,
    ConformerEncoderArgs,
    ConformerEncoderV2Args,
    EBranchformerEncoderArgs,
)
from i6_experiments.users.zeineldeen.models.asr.encoder.conformer_encoder import ConformerEncoder
from i6_experiments.users.zeineldeen.models.asr.encoder.conformer_encoder_v2 import ConformerEncoderV2
from i6_experiments.users.zeineldeen.models.asr.encoder.ebranchformer_encoder import EBranchformerEncoder
from i6_experiments.users.zeineldeen.models.asr.decoder.args import (
    DecoderArgs,
    TransformerDecoderArgs,
    RNNDecoderArgs,
    ConformerDecoderArgs,
)
from i6_experiments.users.zeineldeen.models.asr.decoder.transformer_decoder import TransformerDecoder
from i6_experiments.users.zeineldeen.models.asr.decoder.conformer_decoder import ConformerDecoder
from i6_experiments.users.zeineldeen.models.asr.decoder.rnn_decoder import RNNDecoder
from i6_experiments.users.zeineldeen.models.lm.external_lm_decoder import ExternalLMDecoder
from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.librispeech_960.search_helpers import (
    add_joint_ctc_att_subnet,
    add_filter_blank_and_merge_labels_layers,
    create_ctc_decoder,
    update_tensor_entry,
)

from i6_experiments.users.zeineldeen import data_aug
from i6_experiments.users.zeineldeen.data_aug import specaugment

from i6_core.returnn.config import ReturnnConfig, CodeWrapper

# -------------------------- Base Config -------------------------- #

config = {}

# changing these does not change the hash
post_config = {
    "use_tensorflow": True,
    "tf_log_memory_usage": True,
    "cleanup_old_models": True,
    "log_batch_size": True,
    "debug_print_layer_output_template": True,
    "debug_mode": False,
    "batching": "random",
    # "tf_session_opts": {"gpu_options": {"per_process_gpu_memory_fraction": 0.92}},
}

# -------------------------- LR Scheduling -------------------------- #

# Noam LR
noam_lr_str = """
def noam(n, model_d={model_d}, warmup_n={warmup_n}):
  from returnn.tf.compat import v1 as tf
  model_d = tf.cast(model_d, tf.float32)
  n = tf.cast(n, tf.float32)
  warmup_n = tf.cast(warmup_n, tf.float32)
  return tf.pow(model_d, -0.5) * tf.minimum(tf.pow(n, -0.5), n * tf.pow(warmup_n, -1.5))

def dynamic_learning_rate(*, network, global_train_step, learning_rate, **kwargs):
  return learning_rate * noam(n=global_train_step)
"""

warmup_lr_str = """
def warmup_lr(step, warmup_steps={warmup_steps}):
    from returnn.tf.compat import v1 as tf
    step = tf.cast(step, tf.float32)
    warmup_steps = tf.cast(warmup_steps, tf.float32)
    return tf.pow(warmup_steps, 0.5) * tf.minimum(tf.pow(step, -0.5), step * tf.pow(warmup_steps, -1.5))

def dynamic_learning_rate(*, network, global_train_step, learning_rate, **kwargs):
  return learning_rate * warmup_lr(step=global_train_step)
"""

cycle_lr_str = """
def cyclic_lr(step, decay={decay}, interval={interval}):
    from returnn.tf.compat import v1 as tf
    return tf.pow(decay, tf.cast(tf.math.floormod(step, interval), dtype=float))

def dynamic_learning_rate(*, network, global_train_step, learning_rate, **kwargs):
    from returnn.tf.compat import v1 as tf
    return learning_rate * cyclic_lr(step=global_train_step)
"""

oclr_str = """
def dynamic_learning_rate(*, network, global_train_step, learning_rate, **kwargs):
      initialLR  = {initial_lr}
      peakLR     = {peak_lr}
      finalLR    = {final_lr}
      cycleEpoch = {cycle_ep}
      totalEpoch = {total_ep}
      nStep      = {n_step}
    
      steps     = cycleEpoch * nStep
      stepSize  = (peakLR - initialLR) / steps
      steps2    = (totalEpoch - 2 * cycleEpoch) * nStep
      stepSize2 = (initialLR - finalLR) / steps2
    
      import tensorflow as tf
      n = tf.cast(global_train_step, tf.float32)
      return tf.where(global_train_step <= steps, initialLR + stepSize * n,
                 tf.where(global_train_step <= 2*steps, peakLR - stepSize * (n - steps), 
                     tf.maximum(initialLR - stepSize2 * (n - 2*steps), finalLR)))
"""

# -------------------------- SpecAugment -------------------------- #

specaug_transform_func = """
def transform(
    data,
    network,
    max_time_dim={max_time_dim},
    max_time_num={max_time_num},
    min_num_add_factor={min_num_add_factor},
    freq_dim_factor={freq_dim_factor},
):
  x = data.placeholder
  from returnn.tf.compat import v1 as tf
  step = network.global_train_step
  step1 = tf.where(tf.greater_equal(step, 1000), 1, 0)
  step2 = tf.where(tf.greater_equal(step, 2000), 1, 0)
  def get_masked():
      x_masked = x
      x_masked = random_mask(
        x_masked, batch_axis=data.batch_dim_axis, axis=data.time_dim_axis,
        min_num=step1 + step2 + min_num_add_factor,
        max_num=tf.maximum(tf.shape(x)[data.time_dim_axis] // max_time_num, 2) * (1 + step1 + step2 * 2),
        max_dims=max_time_dim
      )
      x_masked = random_mask(
        x_masked, batch_axis=data.batch_dim_axis, axis=data.feature_dim_axis,
        min_num=step1 + step2 + min_num_add_factor,
        max_num=2 + step1 + step2 * 2,
        max_dims=data.dim // freq_dim_factor
      )
      return x_masked
  x = network.cond_on_train(get_masked, lambda: x)
  return x
"""

# allow disabled specaug initially (step configurable)
specaug_transform_func_v2 = """
def transform(
    data,
    network,
    step0={step0},
    step1={step1},
    step2={step2},
    max_time_dim={max_time_dim},
    max_time_num={max_time_num},
    min_num_add_factor={min_num_add_factor},
    freq_dim_factor={freq_dim_factor},
):
  x = data.placeholder
  from returnn.tf.compat import v1 as tf
  step = network.global_train_step
  step0 = tf.where(tf.greater_equal(step, step0), 1, 0)
  step1 = tf.where(tf.greater_equal(step, step1), 1, 0)
  step2 = tf.where(tf.greater_equal(step, step2), 1, 0)
  def get_masked():
      x_masked = x
      x_masked = random_mask(
        x_masked, batch_axis=data.batch_dim_axis, axis=data.time_dim_axis,
        min_num=step1 + step2 + min_num_add_factor,
        max_num=tf.maximum(tf.shape(x)[data.time_dim_axis] // max_time_num, 2) * (step0 + step1 + step2 * 2),
        max_dims=max_time_dim
      )
      x_masked = random_mask(
        x_masked, batch_axis=data.batch_dim_axis, axis=data.feature_dim_axis,
        min_num=step1 + step2 + step0 * min_num_add_factor,
        max_num=step0 * 2 + step1 + step2 * 2,
        max_dims=data.dim // freq_dim_factor
      )
      return x_masked
  x = network.cond_on_train(get_masked, lambda: x)
  return x
"""

# -------------------------- Pretraining -------------------------- #


def pretrain_layers_and_dims(
    idx,
    net_dict: dict,
    encoder_type,
    decoder_type,
    encoder_args,
    decoder_args,
    variant,
    reduce_dims=True,
    initial_dim_factor=0.5,
    initial_batch_size=None,
    initial_batch_size_idx=3,
    second_bs=None,
    second_bs_idx=None,
    enc_dec_share_grow_frac=True,
    repeat_first=True,
    ignored_keys_for_reduce_dim=None,
    extra_net_dict_override=None,
    initial_disabled_regularization_patterns=None,
):
    """
    Pretraining implementation that works for multiple encoder/decoder combinations

    :param idx:
    :param net_dict:
    :param encoder_type:
    :param decoder_type:
    :param encoder_args:
    :param decoder_args:
    :param variant:
    :param reduce_dims:
    :param initial_dim_factor:
    :param initial_batch_size:
    :param initial_batch_size_idx:
    :param second_bs:
    :param second_bs_idx:
    :param enc_dec_share_grow_frac:
    :return:
    """

    InitialDimFactor = initial_dim_factor
    encoder_keys = ["ff_dim", "enc_key_dim", "conv_kernel_size"]  # TODO: effect of pretraining conv font-end?
    decoder_keys = ["ff_dim"]
    encoder_args_copy = copy.deepcopy(encoder_args)
    decoder_args_copy = copy.deepcopy(decoder_args)

    final_num_blocks = encoder_args["num_blocks"]

    assert final_num_blocks >= 2

    extra_net_dict = dict()
    extra_net_dict["#config"] = {}

    if initial_batch_size:
        if idx < initial_batch_size_idx:
            extra_net_dict["#config"]["batch_size"] = initial_batch_size
        elif second_bs:
            assert second_bs_idx is not None
            if idx < second_bs_idx:
                extra_net_dict["#config"]["batch_size"] = second_bs

    if extra_net_dict_override:
        for k, v in extra_net_dict_override:
            if k == idx:
                extra_net_dict["#config"].update(v)

    if repeat_first:
        idx = max(idx - 1, 0)  # repeat first 0, 0, 1, 2, ...

    if variant == 1:
        num_blocks = max(2 * idx, 1)  # 1/1/2/4/6/8/10/12 -> 8
        StartNumLayers = 1
    elif variant == 2:
        num_blocks = 2**idx  # 1/1/2/4/8/12 -> 6
        StartNumLayers = 1
    elif variant == 3:
        idx += 1
        num_blocks = 2 * idx  # 2/2/4/6/8/10/12 -> 7
        StartNumLayers = 2
    elif variant == 4:
        idx += 1
        num_blocks = 2**idx  # 2/2/4/8/12 -> 5
        StartNumLayers = 2
    elif variant == 5:
        idx += 2
        num_blocks = 2**idx  # 4/4/8/12 -> 4
        StartNumLayers = 4
    elif variant == 6:
        idx += 1  # 1 1 2 3
        num_blocks = 4 * idx  # 4 4 8 12 16
        StartNumLayers = 4
    else:
        raise ValueError("variant {} is not defined".format(variant))

    if num_blocks > final_num_blocks:
        return None

    encoder_args_copy["num_blocks"] = num_blocks
    decoder_args_copy["label_smoothing"] = 0
    EncoderAttNumHeads = encoder_args_copy["att_num_heads"]
    DecoderAttNumHeads = decoder_args_copy["att_num_heads"]

    if reduce_dims:
        grow_frac_enc = 1.0 - float(final_num_blocks - num_blocks) / (final_num_blocks - StartNumLayers)
        dim_frac_enc = InitialDimFactor + (1.0 - InitialDimFactor) * grow_frac_enc

        for key in encoder_keys:
            if ignored_keys_for_reduce_dim and key in ignored_keys_for_reduce_dim:
                continue
            encoder_args_copy[key] = (
                int(encoder_args[key] * dim_frac_enc / float(EncoderAttNumHeads)) * EncoderAttNumHeads
            )

        if decoder_type == TransformerDecoder or decoder_type == ConformerDecoder:
            transf_dec_layers = decoder_args_copy["num_layers"]
            num_transf_layers = min(num_blocks, transf_dec_layers)
            decoder_args_copy["num_layers"] = num_transf_layers

            if enc_dec_share_grow_frac:
                grow_frac_dec = grow_frac_enc
            else:
                grow_frac_dec = 1.0 - float(transf_dec_layers - num_transf_layers) / (
                    transf_dec_layers - StartNumLayers
                )

            dim_frac_dec = InitialDimFactor + (1.0 - InitialDimFactor) * grow_frac_dec

            if decoder_type == ConformerDecoder:
                decoder_keys += ["conv_kernel_size"]

            for key in decoder_keys:
                if ignored_keys_for_reduce_dim and key in ignored_keys_for_reduce_dim:
                    continue
                decoder_args_copy[key] = (
                    int(decoder_args[key] * dim_frac_dec / float(DecoderAttNumHeads)) * DecoderAttNumHeads
                )
        else:
            dim_frac_dec = 1
    else:
        dim_frac_enc = 1
        dim_frac_dec = 1

    # do not enable regulizations in the first pretraining step to make it more stable
    if initial_disabled_regularization_patterns is None:
        # dropout, weight dropout, l2, weight noise
        regs_words = ["dropout", "weight_noise", "l2"]
    else:
        regs_words = initial_disabled_regularization_patterns

    excluded_keys = ["param_dropout_min_ndim", "weight_noise_layers"]

    for k in encoder_args_copy.keys():
        if k in excluded_keys:
            continue
        for regs_word in regs_words:
            if regs_word in k and encoder_args_copy[k] is not None:
                if not isinstance(encoder_args_copy[k], float):
                    continue
                if idx <= 1:
                    encoder_args_copy[k] = 0.0
                else:
                    encoder_args_copy[k] *= dim_frac_enc

    for k in decoder_args_copy.keys():
        if k in excluded_keys:
            continue
        for regs_word in regs_words:
            if regs_word in k and decoder_args_copy[k] is not None:
                if not isinstance(decoder_args_copy[k], float):
                    continue
                if idx <= 1:
                    decoder_args_copy[k] = 0.0
                else:
                    decoder_args_copy[k] *= dim_frac_dec

    encoder_model = encoder_type(**encoder_args_copy)
    encoder_model.create_network()

    decoder_model = decoder_type(base_model=encoder_model, **decoder_args_copy)
    decoder_model.create_network()

    net_dict = encoder_model.network.get_net()

    net_dict.update(decoder_model.network.get_net())

    net_dict.update(extra_net_dict)

    return net_dict


# -------------------------------------------------------------------- #


def create_config(
    training_datasets,
    encoder_args: EncoderArgs,
    decoder_args: DecoderArgs,
    with_staged_network=False,
    is_recog=False,
    input_key="audio_features",
    lr=0.0008,
    wup_start_lr=0.0003,
    lr_decay=0.9,
    const_lr=0,
    wup=10,
    epoch_split=20,
    batch_size=10000,
    accum_grad=2,
    pretrain_reps=5,
    max_seq_length=75,
    min_seq_length=None,
    max_seqs=200,
    noam_opts=None,
    warmup_lr_opts=None,
    with_pretrain=True,
    pretrain_opts=None,
    speed_pert=True,
    oclr_opts=None,
    gradient_clip_global_norm=0.0,
    gradient_clip=0.0,
    ext_lm_opts=None,
    beam_size=12,
    recog_epochs=None,
    prior_lm_opts=None,
    gradient_noise=0.0,
    param_variational_noise=None,
    adamw=False,
    optimizer=None,
    retrain_checkpoint=None,
    decouple_constraints_factor=0.025,
    extra_str=None,
    extra_prolog=None,
    preload_from_files=None,
    min_lr_factor=50,
    specaug_str_func_opts=None,
    recursion_limit=3000,
    feature_extraction_net=None,
    config_override=None,
    freeze_bn=False,
    keep_all_epochs=False,
    allow_lr_scheduling=True,
    learning_rates_list=None,
    min_lr=None,
    global_stats=None,
    speed_pert_version=1,
    specaug_version=1,
    ctc_decode=False,
    ctc_blank_idx=None,
    ctc_log_prior_file=None,
    ctc_prior_scale=None,
    ctc_remove_eos=False,
    joint_ctc_att_decode_args=None,
    staged_hyperparams: dict = None,
    keep_best_n=None,
    param_dropout=0.0,
    mixup_aug_opts=None,
    enable_mixup_in_pretrain=True,
    seq_train_opts=None,
    horovod_params=None,
):
    exp_config = copy.deepcopy(config)  # type: dict
    exp_post_config = copy.deepcopy(post_config)

    exp_config["extern_data"] = training_datasets.extern_data.copy()

    if not is_recog:
        exp_config["train"] = training_datasets.train.as_returnn_opts()
        exp_config["dev"] = training_datasets.cv.as_returnn_opts()
        if training_datasets.devtrain:
            exp_config["eval_datasets"] = {"devtrain": training_datasets.devtrain.as_returnn_opts()}

    target = "bpe_labels"

    # add default hyperparameters
    # model and learning_rates paths are added in the CRNNTraining job
    hyperparams = {
        "gradient_clip": gradient_clip,
        "accum_grad_multiple_step": accum_grad,
        "gradient_noise": gradient_noise,
        "batch_size": batch_size,
        "max_seqs": max_seqs,
        "truncation": -1,
    }
    if param_dropout:
        hyperparams["param_dropout"] = param_dropout  # weight dropout applied to all params
    if param_variational_noise:
        assert isinstance(param_variational_noise, float)
        hyperparams["param_variational_noise"] = param_variational_noise  # applied to all params

    # default: Adam optimizer
    if optimizer:
        hyperparams["optimizer"] = optimizer
    else:
        hyperparams["adam"] = True
        hyperparams["optimizer_epsilon"] = 1e-8

    if adamw:
        hyperparams["decouple_constraints"] = True
        hyperparams["decouple_constraints_factor"] = decouple_constraints_factor

    if max_seq_length:
        if isinstance(max_seq_length, dict):
            hyperparams["max_seq_length"] = max_seq_length
        else:
            hyperparams["max_seq_length"] = {target: max_seq_length}  # char-bpe
    if min_seq_length:
        assert isinstance(min_seq_length, dict), "must be a dict to specify filtering per data key names"
        hyperparams["min_seq_length"] = min_seq_length
    if gradient_clip_global_norm:
        hyperparams["gradient_clip_global_norm"] = gradient_clip_global_norm

    extra_python_code = "\n".join(
        ["import sys", "sys.setrecursionlimit({})".format(recursion_limit)]
    )  # for network construction

    # LR scheduling
    if noam_opts and retrain_checkpoint is None and allow_lr_scheduling:
        noam_opts["model_d"] = encoder_args.enc_key_dim
        exp_config["learning_rate"] = noam_opts["lr"]
        exp_config["learning_rate_control"] = "constant"
        extra_python_code += "\n" + noam_lr_str.format(**noam_opts)
    elif warmup_lr_opts and retrain_checkpoint is None and allow_lr_scheduling:
        if warmup_lr_opts.get("learning_rates", None):
            exp_config["learning_rates"] = warmup_lr_opts["learning_rates"]
        exp_config["learning_rate"] = warmup_lr_opts["peak_lr"]
        exp_config["learning_rate_control"] = "constant"
        extra_python_code += "\n" + warmup_lr_str.format(**warmup_lr_opts)
    elif oclr_opts and retrain_checkpoint is None and allow_lr_scheduling:
        if oclr_opts.get("learning_rates", None):
            exp_config["learning_rates"] = oclr_opts["learning_rates"]
        exp_config["learning_rate"] = oclr_opts["peak_lr"]
        exp_config["learning_rate_control"] = "constant"
        oclr_peak_lr = oclr_opts["peak_lr"]
        oclr_initial_lr = oclr_peak_lr / 10
        extra_python_code += "\n" + oclr_str.format(**oclr_opts, initial_lr=oclr_initial_lr)
    else:  # newbob
        if learning_rates_list:
            learning_rates = learning_rates_list
        else:
            if const_lr is None:
                const_lr = 0
            if retrain_checkpoint is not None:
                learning_rates = None
            elif not allow_lr_scheduling:
                learning_rates = None
            elif isinstance(const_lr, int):
                learning_rates = [wup_start_lr] * const_lr + list(numpy.linspace(wup_start_lr, lr, num=wup))
            elif isinstance(const_lr, list):
                assert len(const_lr) == 2
                learning_rates = (
                    [wup_start_lr] * const_lr[0] + list(numpy.linspace(wup_start_lr, lr, num=wup)) + [lr] * const_lr[1]
                )
            else:
                raise ValueError("unknown const_lr format")

        exp_config["learning_rate"] = lr
        exp_config["learning_rates"] = learning_rates
        exp_config["min_learning_rate"] = lr / min_lr_factor if min_lr is None else min_lr
        exp_config["learning_rate_control"] = "newbob_multi_epoch"
        exp_config["learning_rate_control_relative_error_relative_lr"] = True
        exp_config["learning_rate_control_min_num_epochs_per_new_lr"] = 3
        exp_config["use_learning_rate_control_always"] = True
        exp_config["newbob_multi_num_epochs"] = epoch_split
        exp_config["newbob_multi_update_interval"] = 1
        exp_config["newbob_learning_rate_decay"] = lr_decay

    # -------------------------- network -------------------------- #

    if type(encoder_args) is ConformerEncoderArgs:
        encoder_type = ConformerEncoder
    elif type(encoder_args) is ConformerEncoderV2Args:
        encoder_type = ConformerEncoderV2
    elif type(encoder_args) is EBranchformerEncoderArgs:
        encoder_type = EBranchformerEncoder
    else:
        raise ValueError("invalid encoder_args type")

    if isinstance(decoder_args, TransformerDecoderArgs):
        decoder_type = TransformerDecoder
        dec_type = "transformer"
    elif isinstance(decoder_args, RNNDecoderArgs):
        decoder_type = RNNDecoder
        dec_type = "lstm"
    elif isinstance(decoder_args, ConformerDecoderArgs):
        decoder_type = ConformerDecoder
        dec_type = "conformer"  # TODO: check if same as transformer
    else:
        raise ValueError("invalid decoder_args type")

    encoder_args = asdict(encoder_args)
    if feature_extraction_net:
        encoder_args.update({"target": target, "input": "log_mel_features"})
    else:
        encoder_args.update({"target": target, "input": "data:" + input_key})

    if freeze_bn:
        # freeze BN during training (e.g when retraining.)
        encoder_args["batch_norm_opts"] = {"momentum": 0.0, "use_sample": 1.0}

    encoder_args_input_ = encoder_args["input"]
    if mixup_aug_opts:
        encoder_args.update({"input": "mixup_features"})  # name of mixup layer which will be input to specaug

    conformer_encoder = encoder_type(**encoder_args)
    conformer_encoder.create_network()

    decoder_args = asdict(decoder_args)
    decoder_args.update({"target": target, "beam_size": beam_size})

    transformer_decoder = decoder_type(base_model=conformer_encoder, **decoder_args)
    transformer_decoder.create_network()

    decision_layer_name = transformer_decoder.decision_layer_name
    exp_config["search_output_layer"] = decision_layer_name

    if ext_lm_opts:
        transformer_decoder = ExternalLMDecoder(
            transformer_decoder,
            ext_lm_opts,
            prior_lm_opts=prior_lm_opts,
            beam_size=beam_size,
            dec_type=dec_type,
            coverage_scale=decoder_args["coverage_scale"],
            use_monotonic_att_weights_loss_in_recog=decoder_args["use_monotonic_att_weights_loss_in_recog"],
            length_normalization=decoder_args["length_normalization"],
            length_normalization_exponent=decoder_args["length_normalization_exponent"],
        )
        transformer_decoder.create_network()

    # add full network
    exp_config["network"] = conformer_encoder.network.get_net()  # type: dict

    exp_config["network"].update(transformer_decoder.network.get_net())

    if feature_extraction_net:
        exp_config["network"].update(feature_extraction_net)

    if ctc_log_prior_file:
        add_ctc_log_prior(exp_config, ctc_log_prior_file)

    if ctc_decode:
        assert ctc_blank_idx is not None, "Set CTC blank index."
        add_ctc_decoding(exp_config, beam_size, ctc_prior_scale, ctc_remove_eos, ext_lm_opts, ctc_blank_idx)

    if joint_ctc_att_decode_args:
        add_att_ctc_joint_decoding(exp_config, joint_ctc_att_decode_args, ctc_blank_idx)

    # -------------------------- end network -------------------------- #

    # add hyperparmas
    exp_config.update(hyperparams)

    if retrain_checkpoint is not None:
        exp_config["import_model_train_epoch1"] = retrain_checkpoint

    if ext_lm_opts and ext_lm_opts.get("preload_from_files"):
        if "preload_from_files" not in exp_config:
            exp_config["preload_from_files"] = {}
        exp_config["preload_from_files"].update(copy.deepcopy(ext_lm_opts["preload_from_files"]))

    if preload_from_files:
        if "preload_from_files" not in exp_config:
            exp_config["preload_from_files"] = {}
        exp_config["preload_from_files"].update(preload_from_files)

    if specaug_str_func_opts:
        python_prolog = specaugment.specaug_helpers.get_funcs()
        specaug_str_func_opts_ = copy.deepcopy(specaug_str_func_opts)
        version = specaug_str_func_opts_["version"]
        specaug_str_func_opts_.pop("version")
        if version == 1:
            extra_python_code += "\n" + specaug_transform_func.format(**specaug_str_func_opts_)
        elif version == 2:
            extra_python_code += "\n" + specaug_transform_func_v2.format(**specaug_str_func_opts_)
        else:
            raise ValueError("Invalid specaug version")
    else:
        if specaug_version == 1:
            python_prolog = specaugment.specaug_tf2.get_funcs()  # type: list
        elif specaug_version == 2:
            python_prolog = specaugment.specaug_v2.get_funcs()
        elif specaug_version == 3:
            python_prolog = specaugment.specaug_v3.get_funcs()
        elif specaug_version == 4:
            python_prolog = specaugment.specaug_v4.get_funcs()
        else:
            raise ValueError("Invalid specaug_version")

    if speed_pert:
        if speed_pert_version == 1:
            python_prolog += [data_aug.speed_pert]
        elif speed_pert_version == 2:
            python_prolog += [data_aug.speed_pert_v2]
        elif speed_pert_version == 3:
            python_prolog += [data_aug.speed_pert_v3]
        elif speed_pert_version == 4:
            python_prolog += [data_aug.speed_pert_v4]
        elif isinstance(speed_pert_version, dict):
            # generic
            version = speed_pert_version["version"]
            if version == 1:
                speed_pert_generic_str = data_aug.speed_pert_generic
            else:
                assert "sample_rate" in speed_pert_version
                speed_pert_generic_str = data_aug.speed_pert_generic_v2
            assert isinstance(speed_pert_generic_str, str)
            python_prolog += [speed_pert_generic_str.format(**speed_pert_version)]
        else:
            raise ValueError("Invalid speed_pert_version")

    if feature_extraction_net:
        if global_stats:
            add_global_stats_norm(global_stats, exp_config["network"])
        # else:
        #     # use per-seq norm
        #     add_per_seq_norm(exp_config["network"])

    if mixup_aug_opts:
        add_mixup_layers(
            exp_config["network"],
            feature_extraction_net,
            mixup_aug_opts,
            is_recog=(not with_pretrain and not staged_hyperparams)
            or is_recog,  # add get_global_config() in network to access mixup funcs
        )

    staged_network_dict = None

    # add pretraining
    if with_pretrain and ext_lm_opts is None and retrain_checkpoint is None and is_recog is False:
        if with_staged_network:
            staged_network_dict = {}
            idx = 0
            while True:
                if mixup_aug_opts and not enable_mixup_in_pretrain:
                    encoder_args["input"] = encoder_args_input_
                net = pretrain_layers_and_dims(
                    idx, exp_config["network"], encoder_type, decoder_type, encoder_args, decoder_args, **pretrain_opts
                )
                if not net:
                    break
                net["#copy_param_mode"] = "subset"
                if feature_extraction_net:
                    net.update(feature_extraction_net)
                    if global_stats:
                        add_global_stats_norm(global_stats, net)
                    # else:
                    #     add_per_seq_norm(net)

                if (mixup_aug_opts and enable_mixup_in_pretrain) or (
                    global_stats and not global_stats.get("use_legacy_version", False)
                ):
                    net_as_str = ""
                    if mixup_aug_opts and enable_mixup_in_pretrain:
                        add_mixup_layers(net, feature_extraction_net, mixup_aug_opts, is_recog)
                        net_as_str += "from returnn.config import get_global_config\n"
                    if global_stats and not global_stats.get("use_legacy_version", False):
                        net_as_str += "import numpy\n"

                    net_as_str += "network = %s" % str(net)
                    staged_network_dict[(idx * pretrain_reps) + 1] = net_as_str
                else:
                    staged_network_dict[(idx * pretrain_reps) + 1] = net
                idx += 1
            if mixup_aug_opts or (global_stats and not global_stats.get("use_legacy_version", False)):
                net_as_str = ""
                if mixup_aug_opts:
                    net_as_str += "from returnn.config import get_global_config\n"
                if global_stats and not global_stats.get("use_legacy_version", False):
                    net_as_str += "import numpy\n"

                net_as_str += "network = %s" % str(exp_config["network"])  # mixup already added
                staged_network_dict[(idx * pretrain_reps) + 1] = net_as_str
            else:
                staged_network_dict[(idx * pretrain_reps) + 1] = exp_config["network"]
            exp_config.pop("network")
        else:
            if pretrain_opts is None:
                pretrain_opts = {}

            pretrain_networks = []
            idx = 0
            while True:
                net = pretrain_layers_and_dims(
                    idx, exp_config["network"], encoder_type, decoder_type, encoder_args, decoder_args, **pretrain_opts
                )
                if not net:
                    break
                pretrain_networks.append(net)
                idx += 1

            exp_config["pretrain_nets_lookup"] = {k: v for k, v in enumerate(pretrain_networks)}

            exp_config["pretrain"] = {
                "repetitions": pretrain_reps,
                "copy_param_mode": "subset",
                "construction_algo": CodeWrapper("custom_construction_algo"),
            }

            pretrain_algo_str = (
                "def custom_construction_algo(idx, net_dict):\n\treturn pretrain_nets_lookup.get(idx, None)"
            )
            python_prolog += [pretrain_algo_str]

    if recog_epochs:
        assert isinstance(recog_epochs, list)
        exp_post_config["cleanup_old_models"] = {"keep": recog_epochs}

    if keep_all_epochs:
        exp_post_config["cleanup_old_models"] = False

    # it is 4 by default
    if keep_best_n:
        exp_post_config["keep_best_n"] = keep_best_n

    if extra_str:
        extra_python_code += "\n" + extra_str

    if config_override:
        exp_config.update(config_override)

    if joint_ctc_att_decode_args:
        python_prolog += ["from returnn.tf.compat import v1 as tf_v1"]
        if joint_ctc_att_decode_args.get("remove_eos", False):
            python_prolog += [update_tensor_entry]

    if mixup_aug_opts:
        from i6_experiments.users.zeineldeen.data_aug.mixup.tf_mixup import (
            _mixup_eval_layer_func,
            _mixup_eval_layer_out_type_func,
            _get_raw_func,
        )

        python_prolog += ["from typing import Union, Dict, Any"]
        python_prolog += [
            _mixup_eval_layer_func,
            _mixup_eval_layer_out_type_func,
            _get_raw_func,
        ]

    if (global_stats and not global_stats.get("use_legacy_version", False)) or ctc_log_prior_file:
        python_prolog += ["import numpy"]

    # modify hyperparameters based on epoch (only used in training)
    if staged_hyperparams:
        if staged_network_dict is None:
            staged_network_dict = {}
            max_ep = 0
        else:
            max_ep = max(staged_network_dict.keys())
        for ep, v in staged_hyperparams.items():
            # assume always that the last staged network is same as the base network (e.g last network of pretraining)
            base_net = copy.deepcopy(exp_config["network"])
            assert isinstance(v, dict)
            base_net["#config"] = v
            assert ep not in staged_network_dict, f"{ep} already exists in staged_network_dict?"
            assert ep > max_ep, f"Latest epoch is {max_ep} but current epoch is {ep}"
            staged_network_dict[ep] = base_net

    if seq_train_opts:
        from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.librispeech_960.seq_train_helpers import (
            add_double_softmax,
            add_min_wer,
            add_mmi,
        )

        assert retrain_checkpoint, "seq train requires retrain checkpoint"
        seq_train_type = seq_train_opts["type"]
        assert seq_train_type in ["mmi", "min_wer", "double_softmax"], f"Unknown seq train type {seq_train_type}"
        opts = copy.deepcopy(seq_train_opts)
        del opts["type"]
        if seq_train_type == "mmi":
            add_mmi(net=exp_config["network"], **opts)
        elif seq_train_type == "min_wer":
            add_min_wer(net=exp_config["network"], **opts)
        elif seq_train_type == "double_softmax":
            add_double_softmax(net=exp_config["network"], **opts)

    if horovod_params:
        exp_config.update(horovod_params)

    if extra_prolog:
        assert isinstance(extra_prolog, list)
        python_prolog += extra_prolog

    returnn_config = ReturnnConfig(
        exp_config,
        staged_network_dict=staged_network_dict,
        post_config=exp_post_config,
        python_prolog=python_prolog,
        python_epilog=extra_python_code,
        hash_full_python_code=True,
        pprint_kwargs={"sort_dicts": False},
    )

    return returnn_config


def add_per_seq_norm(net):
    net["log10_"] = copy.deepcopy(net["log10"])
    net["log10"] = {"class": "norm", "from": "log10_", "axis": "T"}


def add_global_stats_norm(global_stats, net):
    if isinstance(global_stats, dict):
        from sisyphus.delayed_ops import DelayedFormat

        net["log10_"] = copy.deepcopy(net["log10"])

        use_legacy_version = global_stats.get("use_legacy_version", False)

        if use_legacy_version:
            # note: kept to not break hashes
            global_mean_delayed = DelayedFormat("{}", global_stats["mean"])
            global_stddev_delayed = DelayedFormat("{}", global_stats["stddev"])
            global_mean_value = CodeWrapper(
                f"eval(\"exec('import numpy') or numpy.loadtxt('{global_mean_delayed}', dtype='float32')\")"
            )
            global_stddev_value = CodeWrapper(
                f"eval(\"exec('import numpy') or numpy.loadtxt('{global_stddev_delayed}', dtype='float32')\")"
            )
        else:
            global_mean_delayed = DelayedFormat("numpy.loadtxt('{}', dtype='float32')", global_stats["mean"])
            global_stddev_delayed = DelayedFormat("numpy.loadtxt('{}', dtype='float32')", global_stats["stddev"])
            global_mean_value = CodeWrapper(global_mean_delayed)
            global_stddev_value = CodeWrapper(global_stddev_delayed)

        net["global_mean"] = {
            "class": "constant",
            "value": global_mean_value,
            "dtype": "float32",
        }
        net["global_stddev"] = {
            "class": "constant",
            "value": global_stddev_value,
            "dtype": "float32",
        }
        net["log10"] = {
            "class": "eval",
            "from": ["log10_", "global_mean", "global_stddev"],
            "eval": "(source(0) - source(1)) / source(2)",
        }
    else:
        # NOTE: old way. should not be used anymore but only kept to not break hashes
        print("WARNING: Using legacy way to add global stats normalization into the config.")
        net["log10_"] = copy.deepcopy(net["log10"])
        net["global_mean"] = {
            "class": "eval",
            "eval": f"exec('import numpy') or numpy.loadtxt('{global_stats[0]}', dtype='float32') + (source(0) - source(0))",
            "from": "log10_",
        }
        net["global_stddev"] = {
            "class": "eval",
            "eval": f"exec('import numpy') or numpy.loadtxt('{global_stats[1]}', dtype='float32') + (source(0) - source(0))",
            "from": "log10_",
        }
        net["log10"] = {
            "class": "eval",
            "from": ["log10_", "global_mean", "global_stddev"],
            "eval": "(source(0) - source(1)) / source(2)",
        }


def add_mixup_layers(net, feature_extraction_net, mixup_aug_opts, is_recog):
    from i6_experiments.users.zeineldeen.data_aug.mixup.tf_mixup import make_mixup_layer_dict

    assert feature_extraction_net
    # TODO: this is just to test the effect of doing mixup on features in log-space or not
    assert "log_mel_features" in feature_extraction_net, "currently mixup is only supported for log-mel features"
    use_log10_features = mixup_aug_opts.get("use_log10_features", False)
    net.update(
        make_mixup_layer_dict(
            src="log_mel_features" if use_log10_features else "mel_filterbank",
            dim=feature_extraction_net["mel_filterbank"]["n_out"],
            opts=mixup_aug_opts,
            is_recog=is_recog,
        )
    )
    if not use_log10_features:
        # mixup features are not in log10 space so we need to convert
        net["log"] = {
            "from": "mixup",
            "class": "activation",
            "activation": "safe_log",
            "opts": {"eps": 1e-10},
        }
        # this layer is fed as input to SpecAugment layer
        net["mixup_features"] = {"from": "log", "class": "eval", "eval": "source(0) / 2.3026"}
    else:
        net["mixup_features"] = {"class": "copy", "from": "mixup"}


def add_ctc_decoding(config, beam_size, ctc_prior_scale, ctc_remove_eos, ext_lm_opts, ctc_blank_idx):
    # create bpe labels with blank extern data
    config["extern_data"]["bpe_labels_w_blank"] = copy.deepcopy(config["extern_data"]["bpe_labels"])
    config["extern_data"]["bpe_labels_w_blank"].pop("vocab", None)  # vocab is with blank now
    config["extern_data"]["bpe_labels_w_blank"]["dim"] += 1

    create_ctc_decoder(config["network"], beam_size, ctc_prior_scale, ctc_remove_eos)

    # filter out blanks from best hyp
    # TODO: we might want to also dump blank for analysis, however, this needs some fix to work.
    add_filter_blank_and_merge_labels_layers(config["network"], blank_idx=ctc_blank_idx)
    config["network"].pop(config["search_output_layer"], None)
    config["search_output_layer"] = "out_best_wo_blank"


def add_att_ctc_joint_decoding(config, joint_ctc_att_decode_args, ctc_blank_idx):
    # create bpe labels with blank extern data
    config["extern_data"]["bpe_labels_w_blank"] = copy.deepcopy(config["extern_data"]["bpe_labels"])
    config["extern_data"]["bpe_labels_w_blank"]["dim"] += 1

    add_joint_ctc_att_subnet(config["network"], **joint_ctc_att_decode_args)
    joint_ctc_scale = joint_ctc_att_decode_args["ctc_scale"]
    if joint_ctc_scale > 0.0:
        add_filter_blank_and_merge_labels_layers(config["network"], blank_idx=ctc_blank_idx)
        config["network"].pop(config["search_output_layer"], None)
        config["search_output_layer"] = "out_best_wo_blank"
    else:
        pass  # use decision layer as before


def add_ctc_log_prior(config, ctc_log_prior_file):
    from sisyphus.delayed_ops import DelayedFormat

    config["network"]["ctc_log_prior"] = {
        "class": "constant",
        "value": CodeWrapper(DelayedFormat("numpy.loadtxt('{}', dtype='float32')", ctc_log_prior_file)),
    }
