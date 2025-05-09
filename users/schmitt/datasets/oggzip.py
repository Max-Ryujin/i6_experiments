from i6_core.returnn.config import CodeWrapper

from . import hdf

from typing import List, Optional, Dict, Union
from sisyphus import Path


def get_dataset_dict(
        oggzip_path_list: List[Path],
        bpe_file: Optional[Path],
        vocab_file: Optional[Path],
        segment_file: Optional[Path],
        fixed_random_subset: Optional[int],
        partition_epoch: int,
        pre_process: Optional[CodeWrapper],
        seq_ordering: str,
        epoch_wise_filter: Optional[Dict],
        hdf_targets: Optional[Union[Path, List[Path]]] = None,
        seq_postfix: Optional[int] = 0,
        use_targets: bool = True,
        peak_normalization: bool = True,
        model_file: Optional[Path] = None,
        post_process: Optional[CodeWrapper] = None,
        text_only: bool = False,
        hdf_features: Optional[Union[Path, List[Path]]] = None,
        seq_order_control_dataset: str = "zip_dataset",
):
  # either not use targets or pass arguments for either BPE or SentencePieces
  assert not use_targets or ((bpe_file is not None and vocab_file is not None) or model_file is not None)
  dataset_dict = {
    "class": "MetaDataset",
    "data_map": {
      "data": ("zip_dataset", "data")
    },
    "datasets": {
      "zip_dataset": {
        "class": "OggZipDataset",
        "path": oggzip_path_list,
        "use_cache_manager": True,
        "audio": {
          "features": "raw",
          "peak_normalization": peak_normalization,
          "preemphasis": None,
          "pre_process": pre_process
        },
        "segment_file": segment_file,
        "partition_epoch": partition_epoch,
        "fixed_random_subset": fixed_random_subset,
        "seq_ordering": seq_ordering,
        "epoch_wise_filter": epoch_wise_filter
      }
    },
    "seq_order_control_dataset": seq_order_control_dataset,
  }

  if post_process is not None:
    dataset_dict["datasets"]["zip_dataset"]["audio"]["post_process"] = post_process

  if use_targets:
    if model_file is not None:
      dataset_dict["datasets"]["zip_dataset"]["targets"] = {
        "class": "SentencePieces",
        "alpha": 0.7,  # hard coded for now (Albert's best setting)
        "enable_sampling": True,
        "model_file": model_file,
        # "seq_postfix": [seq_postfix] if seq_postfix is not None else None,  # does not work for sentencepiece?
      }
    else:
      dataset_dict["datasets"]["zip_dataset"]["targets"] = {
        "class": "BytePairEncoding",
        "bpe_file": bpe_file,
        "vocab_file": vocab_file,
        "unknown_label": None,
        "seq_postfix": [seq_postfix] if seq_postfix is not None else None,
      }
    if text_only:
      dataset_dict["data_map"]["data"] = ("zip_dataset", "classes")
    else:
      dataset_dict["data_map"]["targets"] = ("zip_dataset", "classes")
  else:
    dataset_dict["datasets"]["zip_dataset"]["targets"] = None

  if hdf_targets is not None:
    dataset_dict["datasets"]["align"] = hdf.get_dataset_dict(
      hdf_files=hdf_targets if isinstance(hdf_targets, list) else [hdf_targets],
      partition_epoch=partition_epoch,
      segment_file=segment_file
    )

    dataset_dict["data_map"]["targets"] = ("align", "data")

  if hdf_features is not None:
    dataset_dict["datasets"]["features"] = hdf.get_dataset_dict(
      hdf_files=hdf_features if isinstance(hdf_features, list) else [hdf_features],
      partition_epoch=partition_epoch,
      segment_file=segment_file
    )

    dataset_dict["data_map"]["data"] = ("features", "data")

  return dataset_dict
