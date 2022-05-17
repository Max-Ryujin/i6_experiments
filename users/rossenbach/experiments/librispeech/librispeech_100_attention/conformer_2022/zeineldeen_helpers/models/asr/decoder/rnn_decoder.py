from recipe.crnn.helpers.zeineldeen.network import ReturnnNetwork
from recipe.crnn.helpers.zeineldeen.modules.attention import AttentionMechanism


class RNNDecoder:
  """
  Represents RNN LSTM Attention-based decoder

  Related:
    * Single headed attention based sequence-to-sequence model for state-of-the-art results on Switchboard
      ref: https://arxiv.org/abs/2001.07263
  """

  def __init__(self, base_model, source=None, dropout=0.3, label_smoothing=0.1, target='bpe',
               beam_size=12, embed_dim=621, embed_dropout=0., dec_lstm_num_units=1000,
               dec_output_num_units=1000, l2=None, att_dropout=None, rec_weight_dropout=None, dec_zoneout=False,
               ff_init=None, add_lstm_lm=False, lstm_lm_dim=1000, loc_conv_att_filter_size=None,
               loc_conv_att_num_channels=None, reduceout=True, att_num_heads=1, embed_weight_init=None,
               lstm_weights_init=None):
    """
    :param base_model: base/encoder model instance
    :param str source: input to decoder subnetwork
    :param float dropout: Dropout applied to the softmax input
    :param float label_smoothing: label smoothing value applied to softmax
    :param str target: target data key name
    :param int beam_size: value of the beam size
    :param int embed_dim: target embedding dimension
    :param float|None embed_dropout: dropout to be applied on the target embedding
    :param int dec_lstm_num_units: the number of hidden units for the decoder LSTM
    :param int dec_output_num_units: the number of hidden dimensions for the last layer before softmax
    :param float|None l2: weight decay with l2 norm
    :param float|None att_dropout: dropout applied to attention weights
    :param float|None rec_weight_dropout: dropout applied to weight paramters
    :param bool dec_zoneout: if set, zoneout LSTM cell is used in the decoder instead of nativelstm2
    :param str|None ff_init: feed-forward weights initialization
    :param bool add_lstm_lm: add separate LSTM layer that acts as LM-like model
      same as here: https://arxiv.org/abs/2001.07263
    :param float lstm_lm_dim:
    :param int|None loc_conv_att_filter_size:
    :param int|None loc_conv_att_num_channels:
    :param bool reduceout: if set to True, maxout layer is used
    :param int att_num_heads: number of attention heads
    """

    self.base_model = base_model

    self.source = source

    self.dropout = dropout
    self.label_smoothing = label_smoothing

    self.enc_key_dim = base_model.enc_key_dim
    self.enc_value_dim = base_model.enc_value_dim
    self.att_num_heads = att_num_heads

    self.target = target

    self.beam_size = beam_size

    self.embed_dim = embed_dim
    self.embed_dropout = embed_dropout

    self.dec_lstm_num_units = dec_lstm_num_units
    self.dec_output_num_units = dec_output_num_units

    self.ff_init = ff_init

    self.decision_layer_name = None  # this is set in the end-point config

    self.l2 = l2
    self.att_dropout = att_dropout
    self.rec_weight_dropout = rec_weight_dropout
    self.dec_zoneout = dec_zoneout

    self.add_lstm_lm = add_lstm_lm
    self.lstm_lm_dim = lstm_lm_dim

    self.loc_conv_att_filter_size = loc_conv_att_filter_size
    self.loc_conv_att_num_channels = loc_conv_att_num_channels

    self.embed_weight_init = embed_weight_init
    self.lstm_weights_init = lstm_weights_init

    self.reduceout = reduceout

    self.network = ReturnnNetwork()
    self.subnet_unit = ReturnnNetwork()
    self.dec_output = None

  def add_decoder_subnetwork(self, subnet_unit: ReturnnNetwork):

    subnet_unit.add_compare_layer('end', source='output', value=0)  # sentence end token

    # target embedding
    subnet_unit.add_linear_layer(
      'target_embed0', 'output', n_out=self.embed_dim, initial_output=0, with_bias=False, l2=self.l2,
      forward_weights_init=self.embed_weight_init)

    subnet_unit.add_dropout_layer(
      'target_embed', 'target_embed0', dropout=self.embed_dropout, dropout_noise_shape={'*': None})

    # attention
    att = AttentionMechanism(
      enc_key_dim=self.enc_key_dim, att_num_heads=self.att_num_heads, att_dropout=self.att_dropout, l2=self.l2,
      loc_filter_size=self.loc_conv_att_filter_size, loc_num_channels=self.loc_conv_att_num_channels)
    subnet_unit.update(att.create())

    # LM-like component same as here https://arxiv.org/pdf/2001.07263.pdf
    lstm_lm_component = None
    if self.add_lstm_lm:
      lstm_lm_component = subnet_unit.add_rnn_cell_layer(
        'lm_like_s', 'prev:target_embed', n_out=self.lstm_lm_dim, l2=self.l2)

    lstm_inputs = []
    if lstm_lm_component:
      lstm_inputs += [lstm_lm_component]
    else:
      lstm_inputs += ['prev:target_embed']
    lstm_inputs += ['prev:att']

    # LSTM decoder (or decoder state)
    if self.dec_zoneout:
      subnet_unit.add_rnn_cell_layer(
        's', lstm_inputs, n_out=self.dec_lstm_num_units, l2=self.l2, weights_init=self.lstm_weights_init,
        unit='zoneoutlstm', unit_opts={'zoneout_factor_cell': 0.15, 'zoneout_factor_output': 0.05})
    else:
      if self.rec_weight_dropout:
        # a rec layer with unit nativelstm2 is required to use rec_weight_dropout
        subnet_unit.add_rec_layer(
          's', lstm_inputs, n_out=self.dec_lstm_num_units, l2=self.l2, unit='NativeLSTM2',
          rec_weight_dropout=self.rec_weight_dropout, weights_init=self.lstm_weights_init)
      else:
        subnet_unit.add_rnn_cell_layer(
          's', lstm_inputs, n_out=self.dec_lstm_num_units, l2=self.l2, weights_init=self.lstm_weights_init)

    # ASR softmax output layer
    subnet_unit.add_linear_layer(
      'readout_in', ["s", "prev:target_embed", "att"], n_out=self.dec_output_num_units, l2=self.l2)

    if self.reduceout:
      subnet_unit.add_reduceout_layer('readout', 'readout_in')
    else:
      subnet_unit.add_copy_layer('readout', 'readout_in')

    output_prob = subnet_unit.add_softmax_layer(
      'output_prob', 'readout', l2=self.l2, loss='ce', loss_opts={'label_smoothing': self.label_smoothing},
      target=self.target, dropout=self.dropout)

    subnet_unit.add_choice_layer(
      'output', output_prob, target=self.target, beam_size=self.beam_size, initial_output=0)

    # recurrent subnetwork
    dec_output = self.network.add_subnet_rec_layer(
      'output', unit=subnet_unit.get_net(), target=self.target, source=self.source)

    return dec_output

  def create_network(self):
    self.dec_output = self.add_decoder_subnetwork(self.subnet_unit)

    # Add to Base/Encoder network

    if hasattr(self.base_model, 'enc_proj_dim') and self.base_model.enc_proj_dim:
      self.base_model.network.add_copy_layer('enc_ctx', 'encoder_proj')
      self.base_model.network.add_split_dim_layer(
        'enc_value', 'encoder_proj', dims=(self.att_num_heads, self.enc_value_dim // self.att_num_heads))
    else:
      self.base_model.network.add_linear_layer(
        'enc_ctx', 'encoder', with_bias=True, n_out=self.enc_key_dim, l2=self.base_model.l2)
      self.base_model.network.add_split_dim_layer(
        'enc_value', 'encoder', dims=(self.att_num_heads, self.enc_value_dim // self.att_num_heads))

    self.base_model.network.add_linear_layer(
      'inv_fertility', 'encoder', activation='sigmoid', n_out=self.att_num_heads, with_bias=False)

    decision_layer_name = self.base_model.network.add_decide_layer('decision', self.dec_output, target=self.target)
    self.decision_layer_name = decision_layer_name

    return self.dec_output
