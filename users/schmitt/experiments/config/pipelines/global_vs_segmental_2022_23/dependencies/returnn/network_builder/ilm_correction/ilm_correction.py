from typing import Optional, Dict, List

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.network_builder.network_builder import add_is_last_frame_condition


def add_ilm_correction(
        network: Dict,
        rec_layer_name: str,
        target_num_labels: int,
        opts: Dict,
        label_prob_layer: str,
        att_layer_name: str,
        static_att: bool
):
  network[rec_layer_name]["unit"].update({
    "prior_readout_in": {
      "class": "linear",
      "activation": None,
      "with_bias": True,
      "from": ["prior_s", "prev:target_embed", att_layer_name],
      "n_out": 1024,
      "reuse_params": "readout_in"
    },
    "prior_readout": {
      "class": "reduce_out",
      "from": ["prior_readout_in"],
      "num_pieces": 2,
      "mode": "max",
    },
    "prior_label_prob": {
      "class": "softmax",
      "from": ["prior_readout"],
      "n_out": target_num_labels,
    },
  })

  prior_s_dict = {
    "prior_s": {
      "class": "rnn_cell",
      "unit": "zoneoutlstm",
      "n_out": 1024,
      "from": ["prev:target_embed", att_layer_name if static_att else f"prev:{att_layer_name}"],
      "unit_opts": {
        "zoneout_factor_cell": 0.15,
        "zoneout_factor_output": 0.05,
      },
    },
  }

  if label_prob_layer == "label_log_prob":
    network[rec_layer_name]["unit"].update({
      "prior_s": {
        "class": "unmask",
        "from": "prior_s_masked",
        "mask": "prev:output_emit",
      },
      "prior_s_masked": {
        "class": "masked_computation",
        "from": ["prev:target_embed"],
        "mask": "prev:output_emit",
        "unit": {
          "class": "subnetwork",
          "from": "data",
          "subnetwork": {
            **prior_s_dict,
            "output": {
              "class": "copy",
              "from": "prior_s",
            }
          }
        }
      }
    })
    network[rec_layer_name]["unit"]["prior_s_masked"]["unit"]["subnetwork"]["prior_s"]["from"] = ["data", f"base:{att_layer_name}" if static_att else f"base:prev:{att_layer_name}"]
    network[rec_layer_name]["unit"]["prior_s_masked"]["unit"]["subnetwork"]["prior_s"]["name_scope"] = "/output/rec/s/rec"
    network[rec_layer_name]["unit"]["prior_label_prob"]["reuse_params"] = "label_log_prob"
  else:
    assert label_prob_layer == "output_prob"
    network[rec_layer_name]["unit"].update(prior_s_dict)
    network[rec_layer_name]["unit"]["prior_s"]["reuse_params"] = "s"
    network[rec_layer_name]["unit"]["prior_label_prob"]["reuse_params"] = "output_prob"

  combo_label_prob_layer = f"combo_{label_prob_layer}"
  if combo_label_prob_layer in network[rec_layer_name]["unit"]:
    network[rec_layer_name]["unit"][combo_label_prob_layer]["from"].append("prior_label_prob")
    network[rec_layer_name]["unit"][combo_label_prob_layer]["eval"] += f" - {opts['scale']} * safe_log(source(2))"
  else:
    network[rec_layer_name]["unit"].update({
      combo_label_prob_layer: {
        "class": "eval",
        "from": [label_prob_layer, "prior_label_prob"],
      },
    })
    if label_prob_layer == "label_log_prob":
      network[rec_layer_name]["unit"][combo_label_prob_layer][
        "eval"] = f"source(0) - {opts['scale']} * safe_log(source(1))"
      network[rec_layer_name]["unit"]["label_log_prob_plus_emit"]["from"] = [combo_label_prob_layer, "emit_log_prob"]
    else:
      assert label_prob_layer == "output_prob"
      network[rec_layer_name]["unit"][combo_label_prob_layer][
        "eval"] = f"safe_log(source(0)) - {opts['scale']} * safe_log(source(1))"
      network[rec_layer_name]["unit"]["output"]["from"] = combo_label_prob_layer
      network[rec_layer_name]["unit"]["output"]["input_type"] = "log_prob"

  # special eos handling only for segmental models
  if label_prob_layer == "label_log_prob":
    if opts["correct_eos"]:
      add_is_last_frame_condition(network, rec_layer_name)  # adds layer "is_last_frame"
      network[rec_layer_name]["unit"].update({
        "ilm_eos_prob": {
          "class": "gather",
          "from": "prior_label_prob",
          "position": opts["eos_idx"],
          "axis": "f",
        },
        "ilm_eos_log_prob0": {
          "class": "eval",
          "eval": "safe_log(source(0))",
          "from": "ilm_eos_prob",
        },
        "ilm_eos_log_prob": {  # this layer is only non-zero for the last frame
          "class": "switch",
          "condition": "is_last_frame",
          "true_from": "ilm_eos_log_prob0",
          "false_from": 0.0,
        }
      })

      assert network[rec_layer_name]["unit"]["blank_log_prob"]["from"] == "emit_prob0" \
          or network[rec_layer_name]["unit"]["blank_log_prob"]["from"] == ["emit_prob0", "lm_eos_log_prob"] \
          and network[rec_layer_name]["unit"]["blank_log_prob"]["eval"].startswith("tf.math.log_sigmoid(-source(0))"), (
        "blank_log_prob layer is not as expected"
      )
      blank_log_prob_layer = network[rec_layer_name]["unit"]["blank_log_prob"]
      if type(blank_log_prob_layer["from"]) is str:
        blank_log_prob_layer["from"] = [blank_log_prob_layer["from"]]

      # in the last frame, we want to subtract the ilm eos log prob from the blank log prob
      blank_log_prob_layer["from"].append("ilm_eos_log_prob")
      blank_log_prob_layer["eval"] += f" - source({len(blank_log_prob_layer['from']) - 1})"


def add_se_loss(
        network: Dict,
        rec_layer_name: str,
        att_layer_name: str
):
  network[rec_layer_name]["unit"].update({
    "se_loss": {
        "class": "eval",
        "eval": "(source(0) - source(1)) ** 2",
        "from": ["att", att_layer_name],
    },
    "att_loss": {
        "class": "reduce",
        "mode": "mean",
        "axis": "F",
        "from": "se_loss",
        "loss": "as_is",
        "loss_scale": 0.05,
    },
  })

  network[rec_layer_name]["unit"]["att"]["axes"] = "except_time"
