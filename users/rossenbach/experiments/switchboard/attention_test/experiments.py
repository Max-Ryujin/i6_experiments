import copy

from .attention_asr_config import create_config, ConformerEncoderArgs, TransformerDecoderArgs, RNNDecoderArgs
from .additional_config import apply_fairseq_init_to_conformer_encoder, apply_fairseq_init_to_transformer_decoder
from .data import build_training_datasets, build_test_datasets
from ..default_tools import RETURNN_EXE, RETURNN_ROOT
from .feature_extraction_net import log10_net_10ms
from .pipeline import training, search, get_average_checkpoint, get_best_checkpoint


def conformer_baseline():
    prefix_name = "experiments/switchboard/attention_test/conformer_baseline_2023"

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    train_data = build_training_datasets(
        bpe_size=1000,
        use_raw_features=True,
        link_speed_perturbation=True
    )

    test_data, eval_object = build_test_datasets(
        bpe_size=1000,
        use_raw_features=True,
    )

    # ---------------------------------------------------------------------------------------------------------------- #
    # local experiment function

    def run_exp(ft_name, feature_extraction_net, datasets, train_args, search_args=None):
        search_args = search_args if search_args is not None else train_args

        returnn_config = create_config(training_datasets=datasets, **train_args, feature_extraction_net=feature_extraction_net)
        returnn_search_config = create_config(training_datasets=datasets, **search_args, feature_extraction_net=feature_extraction_net, is_recog=True)

        train_job = training(ft_name, returnn_config, RETURNN_EXE, RETURNN_ROOT, num_epochs=250)

        #averaged_checkpoint = get_average_checkpoint(train_job, num_average=4)
        #best_checkpoint = get_best_checkpoint(train_job)

        search(ft_name + "/ep80", returnn_search_config, train_job.out_checkpoints[80], {"hub5e00": (test_data, eval_object)}, RETURNN_EXE, RETURNN_ROOT)
        search(ft_name + "/ep160", returnn_search_config, train_job.out_checkpoints[160], {"hub5e00": (test_data, eval_object)}, RETURNN_EXE, RETURNN_ROOT)
        search(ft_name + "/ep250", returnn_search_config, train_job.out_checkpoints[250], {"hub5e00": (test_data, eval_object)}, RETURNN_EXE, RETURNN_ROOT)
        #search(ft_name + "/default_last", returnn_search_config, train_job.out_checkpoints[250], test_dataset_tuples, RETURNN_EXE, RETURNN_ROOT)
        #search(ft_name + "/default_best", returnn_search_config, best_checkpoint, test_dataset_tuples, RETURNN_EXE, RETURNN_ROOT)
        #search(ft_name + "/average_4", returnn_search_config, averaged_checkpoint, test_dataset_tuples, RETURNN_EXE, RETURNN_ROOT)

        return train_job


    conformer_enc_args = ConformerEncoderArgs(
        num_blocks=12, input_layer='lstm-6', att_num_heads=8, ff_dim=2048, enc_key_dim=512, conv_kernel_size=32,
        pos_enc='rel', dropout=0.1, att_dropout=0.1, l2=0.0001)

    apply_fairseq_init_to_conformer_encoder(conformer_enc_args)
    conformer_enc_args.ctc_loss_scale = 1.0
    # overwrite BN params
    conformer_enc_args.batch_norm_opts = {
        'momentum': 0.1,
        'epsilon': 1e-3,
        'update_sample_only_in_training': True,
        'delay_sample_update': True
    }

    rnn_dec_args = RNNDecoderArgs()

    trafo_dec_args = TransformerDecoderArgs(
        num_layers=6, embed_dropout=0.1, label_smoothing=0.1,
        apply_embed_weight=True,
        pos_enc='rel',
    )
    apply_fairseq_init_to_transformer_decoder(trafo_dec_args)

    training_args = {}

    # LR scheduling
    training_args['const_lr'] = [42, 100]  # use const LR during pretraining
    training_args['wup_start_lr'] = 0.0002
    training_args['wup'] = 20
    training_args['with_staged_network'] = True
    training_args['speed_pert'] = True



    name = 'tf_feature_conformer_12l_lstm_1l'
    exp_prefix = prefix_name + "/" + name

    lstm_training_args = copy.deepcopy(training_args)
    lstm_training_args['pretrain_opts'] = {'variant': 3, "initial_batch_size": 22500*80}
    lstm_training_args['pretrain_reps'] = 5
    lstm_training_args['batch_size'] = 15000*80  # frames * samples per frame
    lstm_training_args['name'] = name

    exp_args = copy.deepcopy({**lstm_training_args, "encoder_args": conformer_enc_args, "decoder_args": rnn_dec_args})
    train_job_base = run_exp(exp_prefix + "/" + "raw_log10", log10_net_10ms, datasets=train_data, train_args=exp_args)
