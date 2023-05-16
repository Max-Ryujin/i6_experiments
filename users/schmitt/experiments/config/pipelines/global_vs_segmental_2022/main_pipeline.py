import copy
import os

import numpy
import numpy as np

from i6_private.users.schmitt.returnn.tools import DumpForwardJob, CompileTFGraphJob, RASRDecodingJob, \
  CombineAttentionPlotsJob, DumpPhonemeAlignJob, AugmentBPEAlignmentJob, FindSegmentsToSkipJob, ModifySeqFileJob, \
  ConvertCTMBPEToWordsJob, RASRLatticeToCTMJob, CompareAlignmentsJob, DumpAttentionWeightsJob, PlotAttentionWeightsJob, \
  DumpNonBlanksFromAlignmentJob, CalcSearchErrorJob, RemoveLabelFromAlignmentJob, WordsToCTMJob, SwitchLabelInAlignmentJob

from recipe.i6_core.corpus import *
from recipe.i6_core.bpe.apply import ApplyBPEModelToLexiconJob
from recipe.i6_core.tools.git import CloneGitRepositoryJob
from recipe.i6_core.corpus.convert import CorpusToTxtJob
from recipe.i6_core.returnn.training import Checkpoint
from recipe.i6_experiments.users.schmitt.experiments.config.general.concat_seqs import run_concat_seqs
from recipe.i6_experiments.users.schmitt.experiments.config.concat_seqs.concat_seqs import MergeSeqTagFiles

# from sisyphus import *
from sisyphus.delayed_ops import DelayedFormat, DelayedReplace
from i6_experiments.users.schmitt.experiments.swb.transducer.config import \
  SegmentalSWBExtendedConfig, SegmentalSWBAlignmentConfig
from i6_experiments.users.schmitt.experiments.swb.global_enc_dec.config import \
  GlobalEncoderDecoderConfig
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022.sub_pipelines import run_eval, run_training, run_bpe_returnn_decoding, \
  run_rasr_decoding, calculate_search_errors, run_rasr_realignment, calc_rasr_search_errors, run_training_from_file, \
  run_search_from_file, run_rasr_realignment_parallel, run_rasr_decoding_parallel, calc_rasr_search_errors_parallel

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022.model_variants import build_alias, model_variants, model_variants_fixed_chunking
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022.build_rasr_configs import build_returnn_train_config, build_returnn_train_feature_flow, \
  build_phon_align_extraction_config, write_config, build_decoding_config
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022.miscellaneous import find_seqs_to_skip, update_seq_list_file, dump_phoneme_align, calc_align_stats, \
  augment_bpe_align_with_sil, alignment_split_silence, reduce_alignment, convert_phon_json_vocab_to_rasr_vocab, \
  convert_phon_json_vocab_to_allophones, convert_phon_json_vocab_to_state_tying, \
  convert_phon_json_vocab_to_rasr_formats, convert_bpe_json_vocab_to_rasr_formats

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022.alignments import \
  create_swb_alignments, create_librispeech_alignments
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022.recognition import \
  start_rasr_recog_pipeline, start_analysis_pipeline


def run_pipeline():
  stm_jobs = {}
  cv_corpus_job = FilterCorpusBySegmentsJob(
    bliss_corpus=Path("/work/asr3/irie/data/switchboard/corpora/train.corpus.gz"),
    segment_file=Path("/u/schmitt/experiments/transducer/config/dependencies/seg_cv_head3000"))
  train_corpus_job = FilterCorpusBySegmentsJob(
    bliss_corpus=Path("/work/asr3/irie/data/switchboard/corpora/train.corpus.gz"),
    segment_file=Path("/u/schmitt/experiments/transducer/config/dependencies/seg_train"))
  bliss_corpora = [
    Path("/work/asr3/irie/data/switchboard/corpora/dev.corpus.gz"),
    Path("/work/asr3/irie/data/switchboard/corpora/hub5e_01.corpus.gz"),
    Path("/work/asr3/irie/data/switchboard/corpora/rt03s.corpus.gz"),
    cv_corpus_job.out_corpus, train_corpus_job.out_corpus]

  for corpus_path, corpus_alias in zip(bliss_corpora, ["hub5_00", "hub5_01", "rt03s", "cv", "train"]):
    stm_jobs[corpus_alias] = CorpusToStmJob(bliss_corpus=corpus_path)
    stm_jobs[corpus_alias].add_alias("stm_files" + "/" + corpus_alias)
    alias = stm_jobs[corpus_alias].get_one_alias()
    tk.register_output(alias + "/stm_corpus", stm_jobs[corpus_alias].out_stm_path)

  eval_ref_files = {
    "dev": Path("/u/tuske/bin/switchboard/hub5e_00.2.stm"), "hub5e_01": Path("/u/tuske/bin/switchboard/hub5e_01.2.stm"),
    "rt03s": Path("/u/tuske/bin/switchboard/rt03s_ctsonly.stm"), }

  concat_jobs = run_concat_seqs(
    ref_stm_paths={
      "hub5e_00": eval_ref_files["dev"], "hub5e_01": eval_ref_files["hub5e_01"],
      "rt03s": eval_ref_files["rt03s"], "train": stm_jobs["train"].out_stm_path, "cv": stm_jobs["cv"].out_stm_path},
    glm_path=Path("/work/asr2/oberdorfer/kaldi-stable/egs/swbd/s5/data/eval2000/glm"),
    concat_nums=[1, 2, 4, 10, 20])

  merge_train_concat_1_2_job = MergeSeqTagFiles([
    concat_jobs[1]["train"].out_concat_seq_tags, concat_jobs[2]["train"].out_concat_seq_tags])
  tk.register_output("merge_train_seqs_concat_1_2", merge_train_concat_1_2_job.out_seq_tags)

  allophone_path = Path("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/tuske-phoneme-align/allophones")
  allophone_path_librispeech = Path("/work/asr_archive/assis/luescher/best-models/librispeech/960h_2019-04-10/StoreAllophones.34VPSakJyy0U/output/allophones")

  phon_lexicon_path = Path("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/zhou-phoneme-transducer/lexicon/train.lex.v1_0_3.ci.gz")
  phon_lexicon_path_librispeech = Path("/u/corpora/speech/LibriSpeech/lexicon/original.lexicon.golik.xml.gz")
  phon_lexicon_w_blank_path = Path("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/phonemes/lex_with_blank")
  phon_lexicon_wei = Path("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/phonemes/lexicon_wei")

  bpe_sil_lexicon_path = Path("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/bpe/irie/bpe-with-sil_lexicon")
  bpe_sil_phon_lexicon_path = Path("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/bpe/irie/bpe_lexicon_phon")
  bpe_phon_lexicon_path = Path("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/bpe/irie/my_new_lex")

  phon_state_tying_mono_eow_3_states_path = Path("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/tuske-phoneme-align/state-tying_mono-eow_3-states")
  zoltan_phon_align_cache = Path("/work/asr4/zhou/asr-exps/swb1/dependencies/zoltan-alignment/train.alignment.cache.bundle", cached=True)
  chris_phon_align_librispeech = Path("/work/asr3/luescher/setups-data/librispeech/best-model/960h_2019-04-10/AlignmentJob.uPtxlMbFI4lx/output/alignment.cache.bundle", cached=True)

  rasr_nn_trainer = Path("/u/schmitt/src/rasr/arch/linux-x86_64-standard/nn-trainer.linux-x86_64-standard")
  rasr_flf_tool = Path("/u/schmitt/src/rasr/arch/linux-x86_64-standard/flf-tool.linux-x86_64-standard")
  rasr_am_trainer = Path("/u/schmitt/src/rasr/arch/linux-x86_64-standard/acoustic-model-trainer.linux-x86_64-standard")
  rasr_am_trainer_dev = Path("/u/schmitt/src/rasr-dev/arch/linux-x86_64-standard/acoustic-model-trainer.linux-x86_64-standard")

  bpe_vocab = {
    "bpe_file": Path('/work/asr3/irie/data/switchboard/subword_clean/ready/swbd_clean.bpe_code_1k'),
    "vocab_file": Path('/work/asr3/irie/data/switchboard/subword_clean/ready/vocab.swbd_clean.bpe_code_1k')
  }

  corpus_files = {
    "train": Path("/work/asr3/irie/data/switchboard/corpora/train.corpus.gz", cached=True),
    "dev": Path("/work/asr3/irie/data/switchboard/corpora/dev.corpus.gz", cached=True),
    "hub5e_01": Path("/work/asr3/irie/data/switchboard/corpora/hub5e_01.corpus.gz", cached=True),
    "rt03s": Path("/work/asr3/irie/data/switchboard/corpora/rt03s.corpus.gz", cached=True),
  }

  dev_corpus_splits_job = ShuffleAndSplitSegmentsJob(
    Path("/work/asr3/zeyer/schmitt/tests/swb1/bpe-transducer_decoding-test/hub5_00_all_seqs"),
    split={0: 0.1, 1: 0.9}
  )
  dev_corpus_splits_job.add_alias("dev_corpus_split_segments")
  tk.register_output(dev_corpus_splits_job.get_one_alias(), dev_corpus_splits_job.out_segments[0])

  corpus_files_librispeech = {
    "train-merged-960": Path("/work/speech/golik/setups/librispeech/resources/corpus/corpus.train-merged-960.corpus.gz", cached=True),
  }

  feature_cache_files = {
    "train": Path("/u/tuske/work/ASR/switchboard/feature.extraction/gt40_40/data/gt.train.bundle", cached=True),
    "dev": Path("/u/tuske/work/ASR/switchboard/feature.extraction/gt40_40/data/gt.dev.bundle", cached=True),
    "hub5e_01": Path("/u/tuske/work/ASR/switchboard/feature.extraction/gt40_40/data/gt.hub5e_01.bundle", cached=True),
    "rt03s": Path("/u/tuske/work/ASR/switchboard/feature.extraction/gt40_40/data/gt.rt03s.bundle", cached=True)
  }

  feature_cache_files_librispeech = {
    "train-merged-960": Path("/work/asr3/luescher/setups-data/librispeech/best-model/960h_2019-04-10/FeatureExtraction.Gammatone.de79otVcMWSK/output/gt.cache.bundle", cached=True),
  }

  bpe_standard_aligns = {
    "train": Path(
      "/work/asr3/zeyer/schmitt/sisyphus_work_dirs/merboldt_swb_transducer/rna-tf2.blank0.enc6l-grow2l.scratch-lm.rdrop02.lm1-1024.attwb5-drop02.l2_1e_4.mlr50.epoch-150.swap.data-train.hdf",
      cached=True),
    "cv": Path(
      "/work/asr3/zeyer/schmitt/sisyphus_work_dirs/merboldt_swb_transducer/rna-tf2.blank0.enc6l-grow2l.scratch-lm.rdrop02.lm1-1024.attwb5-drop02.l2_1e_4.mlr50.epoch-150.swap.data-dev.hdf",
      cached=True)}

  bpe_global_att_realigns = {
    "train": Path(
      "/work/asr3/zeyer/schmitt/sisyphus_work_dirs/merboldt_swb_transducer/global-att-realign-train.hdf",
      cached=True), "cv": Path(
      "/work/asr3/zeyer/schmitt/sisyphus_work_dirs/merboldt_swb_transducer/global-att-realign-cv.hdf",
      cached=True)}

  bpe_global_att_realigns_wo_eos = {
    "train": Path(
      "/work/asr3/zeyer/schmitt/sisyphus_work_dirs/merboldt_swb_transducer/global-att-realign-wo-eos-train.hdf",
      cached=True), "cv": Path(
      "/work/asr3/zeyer/schmitt/sisyphus_work_dirs/merboldt_swb_transducer/global-att-realign-wo-eos-cv.hdf",
      cached=True)}

  bpe_seg_att_realigns = {
    "train": Path(
      "/work/asr3/zeyer/schmitt/sisyphus_work_dirs/merboldt_swb_transducer/seg-att-realign-train.hdf",
      cached=True), "cv": Path(
      "/work/asr3/zeyer/schmitt/sisyphus_work_dirs/merboldt_swb_transducer/seg-att-realign-cv.hdf",
      cached=True)}

  bpe_seg_att_wo_length_model_realigns = {
    "train": Path("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/merboldt_swb_transducer/seg-att-wo-length-model-realign-train.hdf",
      cached=True),
    "cv": Path("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/merboldt_swb_transducer/seg-att-wo-length-model-realign-cv.hdf",
      cached=True)}

  bpe_seg_att_wo_length_model_retrain1_realigns = {"train": Path(
    "/work/asr3/zeyer/schmitt/sisyphus_work_dirs/merboldt_swb_transducer/seg-att-wo-length-model-retrain1-realign-train.hdf",
    cached=True), "cv": Path(
    "/work/asr3/zeyer/schmitt/sisyphus_work_dirs/merboldt_swb_transducer/seg-att-wo-length-model-retrain1-realign-cv.hdf",
    cached=True)}

  seq_filter_files_standard = {
    "train": Path("/u/schmitt/experiments/transducer/config/dependencies/seg_train", cached=True),
    "devtrain": Path("/u/schmitt/experiments/transducer/config/dependencies/seg_train_head3000", cached=True),
    "cv": Path("/u/schmitt/experiments/transducer/config/dependencies/seg_cv_head3000", cached=True)}

  seq_filter_files_standard_librispeech = {
    "train": Path("/u/luescher/setups/librispeech/2019-07-23--data-augmentation/work/corpus/ShuffleAndSplitSegments.HwFoXgqL5gha/output/train.segments", cached=True),
  }

  dev_debug_segment_file = Path(
    "/work/asr3/zeyer/schmitt/tests/swb1/bpe-transducer_decoding-test/segments.1"
  )
  cv_debug_segment_file = Path(
    "/work/asr3/zeyer/schmitt/tests/swb1/bpe-transducer_decoding-test/cv_test_segments1"
  )

  zoltan_4gram_lm = {
    "image": Path("/u/zhou/asr-exps/swb1/dependencies/zoltan_4gram.image"),
    "image_wei": Path("/u/zhou/asr-exps/swb1/dependencies/monophone-eow/clean-all/zoltan_4gram.image"),
    "file": Path("/work/asr4/zhou/asr-exps/swb1/dependencies/zoltan_4gram.gz")
  }

  concat_seq_tags_examples = {
    1: [
      "switchboard-1/sw02022A/sw2022A-ms98-a-0002"
    ],
    2: [
      "switchboard-1/sw02022A/sw2022A-ms98-a-0002;switchboard-1/sw02022A/sw2022A-ms98-a-0003"
    ],
    4: [
      "switchboard-1/sw02022A/sw2022A-ms98-a-0002;switchboard-1/sw02022A/sw2022A-ms98-a-0003;switchboard-1/sw02022A/sw2022A-ms98-a-0004;switchboard-1/sw02022A/sw2022A-ms98-a-0005"
    ],
    10: [
      "switchboard-1/sw02022A/sw2022A-ms98-a-0002;switchboard-1/sw02022A/sw2022A-ms98-a-0003;switchboard-1/sw02022A/sw2022A-ms98-a-0004;switchboard-1/sw02022A/sw2022A-ms98-a-0005;switchboard-1/sw02022A/sw2022A-ms98-a-0006;switchboard-1/sw02022A/sw2022A-ms98-a-0007;switchboard-1/sw02022A/sw2022A-ms98-a-0009;switchboard-1/sw02022A/sw2022A-ms98-a-0011;switchboard-1/sw02022A/sw2022A-ms98-a-0013;switchboard-1/sw02022A/sw2022A-ms98-a-0015"
    ],
    20: [
      "switchboard-1/sw02022A/sw2022A-ms98-a-0002;switchboard-1/sw02022A/sw2022A-ms98-a-0003;switchboard-1/sw02022A/sw2022A-ms98-a-0004;switchboard-1/sw02022A/sw2022A-ms98-a-0005;switchboard-1/sw02022A/sw2022A-ms98-a-0006;switchboard-1/sw02022A/sw2022A-ms98-a-0007;switchboard-1/sw02022A/sw2022A-ms98-a-0009;switchboard-1/sw02022A/sw2022A-ms98-a-0011;switchboard-1/sw02022A/sw2022A-ms98-a-0013;switchboard-1/sw02022A/sw2022A-ms98-a-0015;switchboard-1/sw02022A/sw2022A-ms98-a-0017;switchboard-1/sw02022A/sw2022A-ms98-a-0019;switchboard-1/sw02022A/sw2022A-ms98-a-0021;switchboard-1/sw02022A/sw2022A-ms98-a-0023;switchboard-1/sw02022A/sw2022A-ms98-a-0024;switchboard-1/sw02022A/sw2022A-ms98-a-0025;switchboard-1/sw02022A/sw2022A-ms98-a-0027;switchboard-1/sw02022A/sw2022A-ms98-a-0028;switchboard-1/sw02022A/sw2022A-ms98-a-0030;switchboard-1/sw02022A/sw2022A-ms98-a-0032"
    ]
  }

  build_returnn_train_feature_flow(feature_cache_path=feature_cache_files["train"])

  phon_extraction_rasr_configs = {
    "train": write_config(*build_phon_align_extraction_config(
      corpus_path=corpus_files["train"], feature_cache_path=feature_cache_files["train"],
      segment_path=seq_filter_files_standard["train"],
      lexicon_path=phon_lexicon_path, allophone_path=allophone_path, alignment_cache_path=zoltan_phon_align_cache
    ), alias="phon_extract_train_rasr_config"),
    "cv": write_config(
      *build_phon_align_extraction_config(corpus_path=corpus_files["train"],
        feature_cache_path=feature_cache_files["train"], segment_path=seq_filter_files_standard["cv"],
        lexicon_path=phon_lexicon_path, allophone_path=allophone_path, alignment_cache_path=zoltan_phon_align_cache),
      alias="phon_extract_cv_rasr_config")
  }

  phon_extraction_rasr_configs_librispeech = {
    "train": write_config(*build_phon_align_extraction_config(
      corpus_path=corpus_files_librispeech["train-merged-960"],
      feature_cache_path=feature_cache_files_librispeech["train-merged-960"],
      segment_path=seq_filter_files_standard_librispeech["train"],
      lexicon_path=phon_lexicon_path_librispeech, allophone_path=allophone_path_librispeech,
      alignment_cache_path=chris_phon_align_librispeech
    ), alias="phon_extract_librispeech_train_rasr_config")
  }

  corpus_to_txt_job = CorpusToTxtJob(corpus_files["train"])
  corpus_to_txt_job.add_alias("train_corpus_to_txt")
  tk.register_output("train_corpus_to_txt", corpus_to_txt_job.out_txt)

  corpus_to_txt_job = CorpusToTxtJob(corpus_files["dev"])
  corpus_to_txt_job.add_alias("dev_corpus_to_txt")
  tk.register_output("dev_corpus_to_txt", corpus_to_txt_job.out_txt)

  segment_corpus_job = SegmentCorpusJob(corpus_files["dev"], num_segments=1)
  segment_corpus_job.add_alias("segments_dev")
  tk.register_output("segments_dev", segment_corpus_job.out_single_segment_files[1])

  total_data = {
    "bpe": {}, "phonemes": {}, "phonemes-split-sil": {}, "bpe-with-sil": {}, "bpe-with-sil-split-sil": {},
    "bpe-sil-wo-sil": {}, "bpe-sil-wo-sil-in-middle": {}, "bpe-with-sil-split-silv2": {}, "bpe-center-positions": {},
    "bpe-sil-wo-sil-center-positions": {}, "bpe-sil-center-positions": {}, "bpe-sil-center-pos-wo-sil": {},
    "bpe-sil-center-pos-wo-sil-in-middle": {}, "bpe-eos": {}}

  total_data_ls = {
    "phonemes"
  }

  create_swb_alignments(
    data_dict=total_data, seq_filter_files_standard=seq_filter_files_standard, bpe_standard_aligns=bpe_standard_aligns,
    bpe_vocab=bpe_vocab, rasr_nn_trainer=rasr_nn_trainer, phon_extraction_rasr_configs=phon_extraction_rasr_configs)

  total_data["bpe-global-realign"] = copy.deepcopy(total_data["bpe"])
  total_data["bpe-global-realign"]["train"]["time-red-6"]["align"] = bpe_global_att_realigns["train"]
  total_data["bpe-global-realign"]["cv"]["time-red-6"]["align"] = bpe_global_att_realigns["cv"]
  total_data["bpe-global-realign-wo-eos"] = copy.deepcopy(total_data["bpe"])
  total_data["bpe-global-realign-wo-eos"]["train"]["time-red-6"]["align"] = bpe_global_att_realigns_wo_eos["train"]
  total_data["bpe-global-realign-wo-eos"]["cv"]["time-red-6"]["align"] = bpe_global_att_realigns_wo_eos["cv"]
  total_data["bpe-seg-realign"] = copy.deepcopy(total_data["bpe"])
  total_data["bpe-seg-realign"]["train"]["time-red-6"]["align"] = bpe_seg_att_realigns["train"]
  total_data["bpe-seg-realign"]["cv"]["time-red-6"]["align"] = bpe_seg_att_realigns["cv"]
  total_data["bpe-seg-wo-length-realign"] = copy.deepcopy(total_data["bpe"])
  total_data["bpe-seg-wo-length-realign"]["train"]["time-red-6"]["align"] = bpe_seg_att_wo_length_model_realigns["train"]
  total_data["bpe-seg-wo-length-realign"]["cv"]["time-red-6"]["align"] = bpe_seg_att_wo_length_model_realigns["cv"]
  total_data["bpe-seg-wo-length-retrain1-realign"] = copy.deepcopy(total_data["bpe"])
  total_data["bpe-seg-wo-length-retrain1-realign"]["train"]["time-red-6"]["align"] = bpe_seg_att_wo_length_model_retrain1_realigns["train"]
  total_data["bpe-seg-wo-length-retrain1-realign"]["cv"]["time-red-6"]["align"] = bpe_seg_att_wo_length_model_retrain1_realigns["cv"]

  # not working yet
  # create_librispeech_alignments(
  #   data_dict=total_data_ls, rasr_nn_trainer=rasr_nn_trainer, phon_extraction_rasr_configs=phon_extraction_rasr_configs_librispeech)

  search_aligns = {}
  search_labels = {}
  for i in range(1):
    for variant_name, params in model_variants.items():
      name = "%s" % variant_name
      check_name = "" + build_alias(**params["config"])
      # check if name is according to my naming conventions
      assert name == check_name, "\n{} \t should be \n{}".format(name, check_name)

      if name in [
        "seg.bpe-with-sil-split-sil.full-ctx.time-red6.fast-rec.fast-rec-full.seg.pooling-att.ctx-w-bias.am2048.frame-length-model-in_am+prev-out-embed.bpe-sil-segs",
        "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs",
        "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.frame-length-model-in_am+prev-out-embed.prev-target-in-readout.all-segs",
        "glob.best-model.bpe-with-sil.time-red6.am2048.1pretrain-reps.ctx-use-bias.pretrain-like-seg.bpe-sil-segs",
        "glob.best-model.bpe-with-sil.time-red6.am2048.1pretrain-reps.no-weight-feedback.ctx-use-bias.pretrain-like-seg.bpe-sil-segs",
        "glob.best-model.bpe.time-red6.am2048.1pretrain-reps.no-weight-feedback.ctx-use-bias.pretrain-like-seg.all-segs/search_cv_120_debug",
        "glob.best-model.bpe-with-sil-split-sil.time-red6.am2048.1pretrain-reps.no-weight-feedback.ctx-use-bias.pretrain-like-seg.bpe-sil-segs",
        "glob.best-model.bpe-with-sil-split-sil.time-red6.am2048.1pretrain-reps.ctx-use-bias.pretrain-like-seg.bpe-sil-segs",
        "glob.best-model.bpe.time-red6.am2048.6pretrain-reps.no-state-vector.no-weight-feedback.no-l2.ctx-use-bias.all-segs",
        "glob.best-model.bpe.time-red6.am2048.6pretrain-reps.no-weight-feedback.no-l2.ctx-use-bias.all-segs",
        # "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs",
        "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.frame-length-model-in_am+prev-out-embed.all-segs",
        "glob.best-model.bpe.time-red6.am2048.6pretrain-reps.no-state-vector.no-l2.ctx-use-bias.all-segs",
        "glob.best-model.bpe.time-red6.am2048.1pretrain-reps.no-weight-feedback.ctx-use-bias.pretrain-like-seg.all-segs"
      ]:
        num_epochs = [150]
      else:
        num_epochs = [40, 80, 120, 150]

      if name in [
        "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs",
        "seg.bpe.full-ctx.time-red6.conf.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs",
        "seg.bpe.full-ctx.time-red6.conf-tim.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs",
      ] or "conf-wei" in name:
        num_epochs.append(300)

      if name in [
        "seg.bpe.mlr-2e-05.gc-0.gn-0.0.adam.bs-10000.sa-albert.static-lr.nb-lrd-0.7.full-ctx.time-red6.conf-tim.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-pt.no-ctx-reg.all-segs"
      ]:
        num_epochs = [40, 80, 120, 150, 175, 200, 225, 250, 275, 300]


      # if name == "seg.bpe.concat.ep-split-12.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs":
      #   num_epochs.append(59)

      # Currently different segments, depending on the label type
      segment_selection = params["config"].pop("segment_selection")
      time_red = int(np.prod(params["config"]["time_red"]))
      if segment_selection == "bpe-sil":
        train_segments = total_data["bpe-with-sil"]["train"]["time-red-%d" % time_red]["seq_filter_file"]
        cv_segments = total_data["bpe-with-sil"]["cv"]["time-red-%d" % time_red]["seq_filter_file"]
        devtrain_segments = total_data["bpe-with-sil"]["devtrain"]["time-red-%d" % time_red]["seq_filter_file"]
      elif segment_selection == "all":
        train_segments = total_data["bpe"]["train"]["time-red-%d" % time_red]["seq_filter_file"]
        cv_segments = total_data["bpe"]["cv"]["time-red-%d" % time_red]["seq_filter_file"]
        devtrain_segments = total_data["bpe"]["devtrain"]["time-red-%d" % time_red]["seq_filter_file"]
      elif segment_selection == "phonemes":
        train_segments = total_data["phonemes"]["train"]["time-red-%d" % time_red]["seq_filter_file"]
        cv_segments = total_data["phonemes"]["cv"]["time-red-%d" % time_red]["seq_filter_file"]
        devtrain_segments = total_data["phonemes"]["devtrain"]["time-red-%d" % time_red]["seq_filter_file"]
      elif segment_selection == "bpe-eos":
        train_segments = total_data["bpe-eos"]["train"]["time-red-%d" % time_red]["seq_filter_file"]
        cv_segments = total_data["bpe-eos"]["cv"]["time-red-%d" % time_red]["seq_filter_file"]
        devtrain_segments = total_data["bpe-eos"]["devtrain"]["time-red-%d" % time_red]["seq_filter_file"]
      else:
        raise NotImplementedError

      returnn_train_rasr_configs = {}
      for config_alias, corpus_name, corpus_segments in zip(
        ["train", "cv", "devtrain", "dev", "hub5e_01", "rt03s"],
        ["train", "train", "train", "dev", "hub5e_01", "rt03s"],
        [train_segments, cv_segments, devtrain_segments, None, None, None]):

        returnn_train_rasr_configs[config_alias] = write_config(*build_returnn_train_config(
          segment_file=corpus_segments, corpus_file=corpus_files[corpus_name],
          feature_cache_path=feature_cache_files[corpus_name]),
          alias="returnn_%s_rasr_config" % config_alias)

      # General data opts, which apply for all models
      train_data_opts = {
        "data": "train", "rasr_config_path": returnn_train_rasr_configs["train"],
        "rasr_nn_trainer_exe": rasr_nn_trainer, "epoch_split": params["config"]["epoch_split"]}
      if params["config"]["model_type"] == "seg":
        train_data_opts["correct_concat_ep_split"] = params["config"].pop("correct_concat_ep_split")
      cv_data_opts = {
        "data": "cv", "rasr_config_path": returnn_train_rasr_configs["cv"],
        "rasr_nn_trainer_exe": rasr_nn_trainer}
      devtrain_data_opts = {
        "data": "devtrain", "rasr_config_path": returnn_train_rasr_configs["devtrain"],
        "rasr_nn_trainer_exe": rasr_nn_trainer}
      dev_data_opts = {
        "data": "dev", "rasr_config_path": returnn_train_rasr_configs["dev"],
        "rasr_nn_trainer_exe": rasr_nn_trainer}
      hub5e_01_data_opts = {
        "data": "hub5e_01", "rasr_config_path": returnn_train_rasr_configs["hub5e_01"], "rasr_nn_trainer_exe": rasr_nn_trainer}
      rt03s_data_opts = {
        "data": "rt03s", "rasr_config_path": returnn_train_rasr_configs["rt03s"], "rasr_nn_trainer_exe": rasr_nn_trainer}

      rasr_decoding_opts = dict(
        corpus_path=corpus_files["dev"],
        reduction_factors=int(np.prod(params["config"]["time_red"])),
        feature_cache_path=feature_cache_files["dev"], skip_silence=False, name=name)

      # Set more specific data opts for the individual model and label types
      if params["config"]["model_type"] == "glob":
        if params["config"]["label_type"].startswith("bpe"):
          sos_idx = 0 if params["config"]["label_type"] in ["bpe", "bpe-eos"] else 1030
          sil_idx = None if params["config"]["label_type"] in ["bpe", "bpe-eos"] else 0
          target_num_labels = 1030 if params["config"]["label_type"] in ["bpe", "bpe-eos"] else 1031
          vocab = bpe_vocab
          vocab["seq_postfix"] = [0]
          dev_data_opts["vocab"] = bpe_vocab
          hub5e_01_data_opts["vocab"] = bpe_vocab
          rt03s_data_opts["vocab"] = bpe_vocab
          train_data_opts.update({
            "label_hdf": total_data[params["config"]["label_type"]]["train"]["label_seqs"],
            "label_name": "bpe", "segment_file": train_segments, "concat_seqs": params["config"].pop("concat_seqs"),
            "concat_seq_tags": merge_train_concat_1_2_job.out_seq_tags})
          cv_data_opts.update({
            "label_hdf": total_data[params["config"]["label_type"]]["cv"]["label_seqs"],
            "label_name": "bpe", "segment_file": cv_segments})
          devtrain_data_opts.update({
            "label_hdf": total_data[params["config"]["label_type"]]["train"]["label_seqs"],
            "label_name": "bpe", "segment_file": devtrain_segments})
          params["config"]["label_name"] = "bpe"
        else:
          assert params["config"]["label_type"] in ["phonemes", "phonemes-split-sil"]
          train_data_opts.update({
            "label_hdf": total_data[params["config"]["label_type"]]["train"]["label_seqs"],
            "label_name": "phonemes",
            "segment_file": train_segments})
          cv_data_opts.update({
            "label_hdf": total_data[params["config"]["label_type"]]["cv"]["label_seqs"],
            "label_name": "phonemes",
            "segment_file": cv_segments})
          devtrain_data_opts.update({
            "label_hdf": total_data[params["config"]["label_type"]]["train"]["label_seqs"],
            "label_name": "phonemes",
            "segment_file": devtrain_segments})
          params["config"]["label_name"] = "phonemes"
          sos_idx = 88
          sil_idx = 0
          target_num_labels = 89
        rasr_decoding_opts.update(
          dict(
            lexicon_path=bpe_sil_lexicon_path, label_unit="word", label_scorer_type="tf-attention",
            label_file_path=total_data["bpe"]["rasr_label_file"], lm_type="simple-history", use_lm_score=False,
            lm_scale=None, lm_file=None, lm_image=None, label_pruning=10.0, label_pruning_limit=12,
            word_end_pruning_limit=12, word_end_pruning=10.0, lm_lookahead_cache_size_high=None,
            lm_lookahead_cache_size_low=None, lm_lookahead_history_limit=None, lm_lookahead_scale=None, lm_lookahead=False,
            blank_label_index=1031, label_recombination_limit=-1))
      else:
        assert params["config"]["model_type"] == "seg"
        rasr_decoding_opts["label_recombination_limit"] = params["config"]["ctx_size"] if params["config"]["ctx_size"] != "inf" else -1
        if params["config"]["label_type"].startswith("bpe"):
          sos_idx = 0 if params["config"]["label_type"] in ["bpe", "bpe-eos", "bpe-global-realign", "bpe-global-realign-wo-eos", "bpe-seg-realign", "bpe-seg-wo-length-realign", "bpe-seg-wo-length-retrain1-realign"] else 1030
          sil_idx = None if params["config"]["label_type"] in ["bpe", "bpe-eos", "bpe-global-realign", "bpe-global-realign-wo-eos", "bpe-seg-realign", "bpe-seg-wo-length-realign", "bpe-seg-wo-length-retrain1-realign"] else 0
          target_num_labels = 1030 if params["config"]["label_type"] in ["bpe", "bpe-eos", "bpe-global-realign", "bpe-global-realign-wo-eos", "bpe-seg-realign", "bpe-seg-wo-length-realign", "bpe-seg-wo-length-retrain1-realign"] else 1031
          targetb_blank_idx = 1030 if params["config"]["label_type"] in ["bpe", "bpe-eos", "bpe-global-realign", "bpe-global-realign-wo-eos", "bpe-seg-realign", "bpe-seg-wo-length-realign", "bpe-seg-wo-length-retrain1-realign"] else 1031
          vocab = bpe_vocab
          dev_data_opts["vocab"] = vocab if params["config"]["label_type"] in ["bpe", "bpe-eos", "bpe-global-realign", "bpe-global-realign-wo-eos", "bpe-seg-realign", "bpe-seg-wo-length-realign", "bpe-seg-wo-length-retrain1-realign"] else {
            "bpe_file": bpe_vocab["bpe_file"], "vocab_file": total_data["bpe-with-sil"]["json_vocab"]}
          hub5e_01_data_opts["vocab"] = vocab
          rt03s_data_opts["vocab"] = vocab
          train_align = total_data[params["config"]["label_type"]]["train"]["time-red-%d" % time_red]["align"] if params["config"]["label_type"] != "bpe-with-sil-split-silv2" else Path("/work/asr3/zeyer/schmitt/old_models_and_analysis/old_bpe_sil_split_sil_aligns/train/AlignmentSplitSilenceJob.4dfiua41gqWb/output/out_align")
          cv_align = total_data[params["config"]["label_type"]]["cv"]["time-red-%d" % time_red]["align"] if params["config"]["label_type"] != "bpe-with-sil-split-silv2" else Path("/work/asr3/zeyer/schmitt/old_models_and_analysis/old_bpe_sil_split_sil_aligns/cv/AlignmentSplitSilenceJob.p0VY0atlAdkq/output/out_align")

          train_data_opts.update({
            "segment_file": train_segments, "alignment": train_align,
            "concat_seqs": params["config"].pop("concat_seqs"),
            "concat_seq_tags": merge_train_concat_1_2_job.out_seq_tags})
          cv_data_opts.update({
            "segment_file": cv_segments, "alignment": cv_align})
          devtrain_data_opts.update({
            "segment_file": devtrain_segments, "alignment": train_align})
          rasr_decoding_opts.update(dict(lexicon_path=bpe_sil_lexicon_path, label_unit="word",
            label_file_path=total_data[params["config"]["label_type"]]["rasr_label_file"],
            lm_type="simple-history", use_lm_score=False,
            lm_scale=None, lm_file=None, lm_image=None, label_pruning=10.0, label_pruning_limit=128,
            word_end_pruning_limit=128, word_end_pruning=10.0, lm_lookahead_cache_size_high=None,
            lm_lookahead_cache_size_low=None, lm_lookahead_history_limit=None, lm_lookahead_scale=None,
            lm_lookahead=False))
        else:
          assert params["config"]["label_type"] in ["phonemes", "phonemes-split-sil"]
          train_align = total_data[params["config"]["label_type"]]["train"]["time-red-%d" % time_red]["align"]
          cv_align = total_data[params["config"]["label_type"]]["cv"]["time-red-%d" % time_red]["align"]
          sos_idx = 88
          sil_idx = 0
          target_num_labels = 89
          targetb_blank_idx = 89
          vocab = bpe_vocab
          train_data_opts.update({
            "segment_file": train_segments, "alignment": train_align})
          cv_data_opts.update({
            "segment_file": cv_segments, "alignment": cv_align})
          devtrain_data_opts.update({
            "segment_file": devtrain_segments, "alignment": train_align})
          rasr_decoding_opts.update(dict(
            lexicon_path=phon_lexicon_wei, label_unit="phoneme",
            label_file_path=total_data["phonemes"]["rasr_label_file"], lm_type="ARPA",
            lm_file=zoltan_4gram_lm["file"], lm_image=zoltan_4gram_lm["image_wei"], lm_scale=0.8, use_lm_score=True,
            label_pruning=12.0, label_pruning_limit=50000, word_end_pruning_limit=5000, word_end_pruning=0.5,
            lm_lookahead_cache_size_high=None, lm_lookahead_cache_size_low=None,
            lm_lookahead_history_limit=None,
            lm_lookahead_scale=None, lm_lookahead=False))

      # update the config params with the specific info
      params["config"].update({
        "sos_idx": sos_idx, "target_num_labels": target_num_labels, "vocab": vocab, "sil_idx": sil_idx
      })
      rasr_decoding_opts.update(dict(start_label_index=sos_idx))
      # in case of segmental/transducer model, we need to set the blank index
      if params["config"]["model_type"] == "seg":
        params["config"].update({
          "targetb_blank_idx": targetb_blank_idx,
        })
        rasr_decoding_opts.update(dict(blank_label_index=targetb_blank_idx))
      # choose the config class depending on the model type
      config_class = {
        "seg": SegmentalSWBExtendedConfig, "glob": GlobalEncoderDecoderConfig}.get(params["config"]["model_type"])
      config_params = copy.deepcopy(params["config"])

      # these parameters are not needed for the config class
      del config_params["label_type"]
      del config_params["model_type"]

      # initialize returnn config
      train_config_obj = config_class(
        task="train",
        post_config={"cleanup_old_models": {"keep_last_n": 1, "keep_best_n": 1, "keep": num_epochs}},
        train_data_opts=train_data_opts,
        cv_data_opts=cv_data_opts,
        devtrain_data_opts=devtrain_data_opts,
        **config_params).get_config()

      if name in model_variants_fixed_chunking:
        if "chunking" in train_config_obj.config:
          from i6_experiments.users.schmitt.chunking import chunking
          chunk_size_data = train_config_obj.config["chunking"][0]["data"]
          chunk_size_align = train_config_obj.config["chunking"][0]["alignment"]
          chunk_step_data = train_config_obj.config["chunking"][1]["data"]
          chunk_step_align = train_config_obj.config["chunking"][1]["alignment"]
          del train_config_obj.config["chunking"]
          train_config_obj.python_prolog += [
            "from returnn.util.basic import NumbersDict",
            "chunk_size = NumbersDict({'alignment': %s, 'data': %s})" % (chunk_size_align, chunk_size_data),
            "chunk_step = NumbersDict({'alignment': %s, 'data': %s})" % (chunk_step_align, chunk_step_data),
            chunking]

      # start standard returnn training
      checkpoints, train_config, train_job = run_training(train_config_obj, mem_rqmt=24, time_rqmt=30, num_epochs=num_epochs,
                                               name=name, alias_suffix="train")

      # for each previously specified epoch, run decoding
      # if name == "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.frame-length-model-in_am+prev-out-embed.all-segs":
      #   num_epochs = [150]
      #   checkpoints = {150: Checkpoint(index_path=Path("/u/schmitt/experiments/transducer/alias/glob.best-model.bpe.time-red6.am2048.6pretrain-reps.no-weight-feedback.no-l2.ctx-use-bias.all-segs/train/output/models/epoch.150.index"))}
      for epoch in num_epochs:
        if epoch in checkpoints and "no-time-loop" not in name or \
                (name == "seg.win-size520.no-time-loop.use-eos.bpe.lr-meas-lab.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.chunk-size-None.all-segs" and epoch == 150):
          checkpoint = checkpoints[epoch]
          if name == "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.all-segs":
            checkpoint = Checkpoint(
              index_path=Path(
                "/work/asr3/zeyer/schmitt/old_models_and_analysis/seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.all-segs/train/output/models/epoch.150.index",
                hash_overwrite=(train_job, "models/epoch.150.index"), creator=train_job
              ))

          if params["config"]["model_type"] == "seg":
            # for bpe + sil model use additional RETURNN decoding as sanity check
            if params["config"]["label_type"].startswith("bpe"):
              if params["config"]["label_type"].startswith("bpe-with-sil") or params["config"]["label_type"].startswith("bpe-sil-wo-sil"):
                config_params["vocab"]["vocab_file"] = total_data["bpe-with-sil"]["json_vocab"]

              if name in [
                "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs",
                "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.frame-length-model-in_am+prev-out-embed.all-segs",
                "seg.bpe.concat-wrong.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs",
                "seg.bpe.concat.ep-split-12.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs",
                "seg.bpe-with-sil-split-sil.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.ctx-w-bias.am2048.frame-length-model-in_am+prev-out-embed.no-ctx-reg.bpe-sil-segs",
                "seg.bpe-with-sil-split-sil.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.ctx-w-bias.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.bpe-sil-segs"] and epoch == 150:
                for concat_num in concat_jobs:
                  for corpus_name in concat_jobs[concat_num]:
                    if corpus_name == "hub5e_00":
                      data_opts = copy.deepcopy(dev_data_opts)
                      stm_job = stm_jobs["hub5_00"]
                    elif corpus_name == "hub5e_01":
                      data_opts = copy.deepcopy(hub5e_01_data_opts)
                      stm_job = stm_jobs["hub5_01"]
                    elif corpus_name == "rt03s":
                      data_opts = copy.deepcopy(rt03s_data_opts)
                      stm_job = stm_jobs["rt03s"]
                    else:
                      continue
                    data_opts.update({
                      "concat_seqs": True, "concat_seq_tags": concat_jobs[concat_num][corpus_name].out_concat_seq_tags,
                      "concat_seq_lens": concat_jobs[concat_num][corpus_name].out_orig_seq_lens_py})
                    search_config = config_class(
                      task="search", beam_size=12, search_data_opts=data_opts, search_use_recomb=False,
                      target="bpe", length_scale=1., **config_params)

                    ctm_results = run_bpe_returnn_decoding(
                      returnn_config=search_config.get_config(),
                      checkpoint=checkpoint, bliss_corpus=stm_job.bliss_corpus, num_epochs=epoch,
                      name=name, dataset_key=corpus_name, concat_seqs=True,
                      alias_addon="concat-%s_beam-%s" % (concat_num, 12), mem_rqmt=4,
                      stm_path=concat_jobs[concat_num][corpus_name].out_stm)
                    run_eval(
                      ctm_file=ctm_results, reference=concat_jobs[concat_num][corpus_name].out_stm, name=name,
                      dataset_key="%s_concat-%s" % (corpus_name, concat_num), num_epochs=epoch, alias_addon="_beam-%s" % 12)

              if "length-loss-scale-0.0" not in name:
                for beam_size in [12]:
                  for use_recomb in [False]:
                    length_scales = [1.]
                    if name == "seg.win-size520.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs" and epoch == 150:
                      length_scales += [0.]
                    for length_scale in length_scales:
                      alias_addon = "returnn_%srecomb_length-scale-%s_beam-%s" % ("" if use_recomb else "no-", length_scale, beam_size)
                      # standard returnn decoding
                      search_config = config_class(
                        task="search", search_data_opts=dev_data_opts, target="bpe", search_use_recomb=use_recomb,
                        beam_size=beam_size, length_scale=length_scale, **config_params)
                      ctm_results = run_bpe_returnn_decoding(
                        returnn_config=search_config.get_config(), checkpoint=checkpoint,
                        bliss_corpus=stm_jobs["hub5_00"].bliss_corpus, num_epochs=epoch, name=name,
                        dataset_key="dev", alias_addon=alias_addon)
                      run_eval(ctm_file=ctm_results, reference=Path("/u/tuske/bin/switchboard/hub5e_00.2.stm"), name=name,
                               dataset_key="dev", num_epochs=epoch, alias_addon=alias_addon)

                      search_error_data_opts = copy.deepcopy(cv_data_opts)
                      alignment_hdf = search_error_data_opts.pop("alignment")
                      segment_file = search_error_data_opts.pop("segment_file")
                      search_error_data_opts["vocab"] = dev_data_opts["vocab"]
                      dump_search_config = config_class(search_use_recomb=True if use_recomb else False, task="search", target="bpe", beam_size=beam_size,
                        search_data_opts=search_error_data_opts, dump_output=True, length_scale=length_scale, **config_params)
                      feed_config_load = config_class(
                        task="train",
                        length_scale=length_scale,
                        post_config={"cleanup_old_models": {"keep_last_n": 1, "keep_best_n": 1, "keep": num_epochs}},
                        train_data_opts=train_data_opts,
                        cv_data_opts=cv_data_opts,
                        devtrain_data_opts=devtrain_data_opts,
                        **config_params).get_config()
                      feed_config_load.config["load"] = checkpoint
                      # alias_addon = "_returnn_search_errors_%srecomb_length-scale-%s_beam-%s" % ("" if use_recomb else "no-", length_scale, beam_size)
                      search_align, ctm_results = calculate_search_errors(checkpoint=checkpoint, search_config=dump_search_config, bliss_corpus=stm_jobs["cv"].bliss_corpus,
                        train_config=feed_config_load, name=name, segment_path=segment_file, ref_targets=alignment_hdf,
                        label_name="alignment", model_type="seg", blank_idx=targetb_blank_idx, rasr_nn_trainer_exe=rasr_nn_trainer,
                        rasr_config=returnn_train_rasr_configs["cv"], alias_addon=alias_addon, epoch=epoch, dataset_key="cv", length_norm=False)

                      if name == "seg.win-size4.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs" and epoch == 150:
                        for seq_tag in [
                          "switchboard-1/sw02102A/sw2102A-ms98-a-0092",
                          # "switchboard-1/sw02102A/sw2102A-ms98-a-0090", "switchboard-1/sw02025A/sw2025A-ms98-a-0035",
                          # "switchboard-1/sw02102A/sw2102A-ms98-a-0002", "switchboard-1/sw02038B/sw2038B-ms98-a-0069"
                        ]:

                          group_alias = name + "/returnn-neural-length_analysis_epoch-%s_length-scale-%s_%s-recomb_max-seg-len-%s%s" % (
                          epoch, length_scale, "no" if not vit_recomb else "vit", max_seg_len,
                          "_length-norm" if length_norm else "")

                          for align_alias, align in zip(["ground-truth", "search", "realign"],
                                                        [alignment_hdf, search_align]):
                            vocab_file = bpe_vocab["vocab_file"] if params["config"]["label_type"] == "bpe" else \
                              total_data["bpe-with-sil"]["json_vocab"]
                            dump_att_weights_job = DumpAttentionWeightsJob(returnn_config=feed_config_load,
                                                                           model_type="seg",
                                                                           rasr_config=returnn_train_rasr_configs["cv"],
                                                                           blank_idx=1030,
                                                                           label_name="alignment",
                                                                           rasr_nn_trainer_exe=rasr_nn_trainer,
                                                                           hdf_targets=align, seq_tag=seq_tag, )
                            dump_att_weights_job.add_alias(
                              group_alias + "/" + seq_tag.replace("/", "_") + "/att_weights_%s_%s" % (
                                align_alias, epoch))
                            tk.register_output(dump_att_weights_job.get_one_alias(), dump_att_weights_job.out_data)

                            plot_weights_job = PlotAttentionWeightsJob(data_path=dump_att_weights_job.out_data,
                                                                       blank_idx=1030,
                                                                       json_vocab_path=vocab_file, time_red=6,
                                                                       seq_tag=seq_tag)
                            plot_weights_job.add_alias(
                              group_alias + "/" + seq_tag.replace("/", "_") + "/plot_att_weights_%s_%s" % (
                                align_alias, epoch))
                            tk.register_output(plot_weights_job.get_one_alias(), plot_weights_job.out_plot)

                          compare_aligns_job = CompareAlignmentsJob(hdf_align1=cv_align, hdf_align2=search_align,
                                                                    seq_tag=seq_tag, blank_idx1=targetb_blank_idx,
                                                                    blank_idx2=targetb_blank_idx,
                                                                    vocab1=vocab_file, vocab2=vocab_file,
                                                                    name1="ground_truth", name2="search_alignment")
                          compare_aligns_job.add_alias(
                            group_alias + "/" + seq_tag.replace("/", "_") + "/search-align-compare")
                          tk.register_output(compare_aligns_job.get_one_alias(), compare_aligns_job.out_align)

                      if name == "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs" and length_scale == 1. and epoch == 150 and use_recomb:
                        for concat_num in concat_seq_tags_examples:
                          search_error_data_opts.update({
                            "concat_seqs": True, "concat_seq_tags": concat_jobs[concat_num]["cv"].out_concat_seq_tags,
                            "concat_seq_lens": concat_jobs[concat_num]["cv"].out_orig_seq_lens_py})
                          dump_search_concat_config = config_class(
                            search_use_recomb=True if use_recomb else False, task="search",
                            search_data_opts=search_error_data_opts, target="bpe", beam_size=beam_size,
                            length_scale=length_scale, dump_output=True, import_model=checkpoint, **config_params)
                          search_targets_concat_hdf, ctm_results = calculate_search_errors(
                            checkpoint=checkpoint, search_config=dump_search_concat_config, train_config=feed_config_load,
                            name=name, segment_path=segment_file, ref_targets=alignment_hdf, label_name="alignment",
                            model_type="seg", blank_idx=targetb_blank_idx,  rasr_nn_trainer_exe=rasr_nn_trainer,
                            rasr_config=returnn_train_rasr_configs["cv"], alias_addon="_cv_concat_%d" % concat_num,
                            epoch=epoch, dataset_key="cv", bliss_corpus=stm_jobs["cv"].bliss_corpus, length_norm=False,
                            concat_seqs=True, stm_path=concat_jobs[concat_num]["cv"].out_stm)
                          # run_eval(
                          #   ctm_file=ctm_results, reference=stm_jobs["cv"].out_stm_path, name=name,
                          #   dataset_key="cv_concat_%d" % concat_num,
                          #   num_epochs=epoch, alias_addon="_beam-%s" % beam_size)
                          vocab_file = bpe_vocab["vocab_file"] if params["config"]["label_type"] == "bpe" else \
                            total_data["bpe-with-sil"]["json_vocab"]
                          hdf_aliases = ["ground-truth-concat", "search-concat"]
                          hdf_targetss = [alignment_hdf, search_targets_concat_hdf]
                          for hdf_alias, hdf_targets in zip(hdf_aliases, hdf_targetss):
                            for seq_tag in concat_seq_tags_examples[concat_num]:
                              dump_att_weights_job = DumpAttentionWeightsJob(
                                returnn_config=feed_config_load,  model_type="seg",
                                rasr_config=returnn_train_rasr_configs["cv"], blank_idx=targetb_blank_idx,
                                label_name="alignment", rasr_nn_trainer_exe=rasr_nn_trainer, hdf_targets=hdf_targets,
                                concat_seqs=True, seq_tag=seq_tag,
                                concat_seq_tags_file=concat_jobs[concat_num]["cv"].out_concat_seq_tags,
                                concat_hdf=False if hdf_alias == "search-concat" else True)
                              seq_tags = seq_tag.split(";")
                              seq_tag_alias = seq_tags[0].replace("/", "_") + "-" + seq_tags[-1].replace("/", "_")
                              dump_att_weights_job.add_alias(
                                name + "/" + seq_tag_alias + "/att_weights_%s-labels" % (hdf_alias,))
                              tk.register_output(dump_att_weights_job.get_one_alias(), dump_att_weights_job.out_data)

                              plot_weights_job = PlotAttentionWeightsJob(
                                data_path=dump_att_weights_job.out_data, blank_idx=targetb_blank_idx,
                                json_vocab_path=vocab_file, time_red=6, seq_tag=seq_tag, mem_rqmt=12)
                              plot_weights_job.add_alias(
                                name + "/" + seq_tag_alias + "/plot_att_weights_%s-labels" % (hdf_alias,))
                              tk.register_output(plot_weights_job.get_one_alias(), plot_weights_job.out_plot)

              if epoch in [150, 300] and not "conf-wei" in name and not "no-time-loop" in name:
                length_scales = [1.]
                if name in [
                  "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.all-segs",
                  "seg.bpe-with-sil-split-sil.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.ctx-w-bias.am2048.frame-length-model-in_am+prev-out-embed.no-ctx-reg.bpe-sil-segs",
                  "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs"] and epoch == 150:
                  length_scales += [.0, .01, .1, .3, .5, .7, .9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.]
                if name in [
                  "seg.win-size520.bpe.lr-meas-lab.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.chunk-size-None.all-segs",
                  "seg.win-size4.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs",
                  "seg.win-size8.bpe.lr-meas-lab.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs"
                  ] and epoch == 150:
                  length_scales += [0.]
                if "bpe-eos" in name or "bpe-seg-wo-length-realign" in name or "bpe-seg-wo-length-retrain1-realign" in name:
                  length_scales = [0.]
                if name in [
                  "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs",
                  "seg.bpe.lr-meas-lab.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs",
                  "seg.win-size8.bpe.lr-meas-lab.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs"
                ]:
                  length_scales += [.7]
                for length_scale in length_scales:
                  max_seg_lens = [20]
                  # if name in [
                  #   "seg.win-size4.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs"
                  # ]:
                  #   max_seg_lens += [20]
                  # if name in [
                  #   "seg.bpe-with-sil-split-sil.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.ctx-w-bias.am2048.frame-length-model-in_am+prev-out-embed.no-ctx-reg.bpe-sil-segs",
                  #   "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs"] and length_scale == 1. and epoch == 150:
                  #   max_seg_lens += [10, 15, 25, 30]
                  for max_seg_len in max_seg_lens:
                    vit_recombs = [True]
                    if name == "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs":
                      vit_recombs = [True]
                    if max_seg_len != 20 or length_scale != 1.:
                      vit_recombs = [True]
                    for vit_recomb in vit_recombs:
                      length_norms = [False]
                      if length_scale in [.0]:
                        length_norms = [True]

                      for length_norm in length_norms:
                        if name == "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs":
                          eval_corpus_names = ["dev", "rt03s", "hub5e_01"] if length_scale in [1., 0.7] and max_seg_len == 20 else ["dev"]

                        else:
                          eval_corpus_names = ["dev"]
                        for eval_corpus_name in eval_corpus_names:
                          # Config for compiling model for RASR
                          compile_config = config_class(
                            task="eval", feature_stddev=3., length_scale=length_scale,
                            force_eos=True if "bpe-eos" in name else False,
                            **config_params)

                          if length_scale == 0.0:
                            mem_rqmt = 24
                          else:
                            mem_rqmt = 12

                          if vit_recomb:
                            blank_update_history = True
                            allow_label_recombination = True
                            allow_word_end_recombination = True
                          else:
                            blank_update_history = True
                            allow_label_recombination = False
                            allow_word_end_recombination = False

                          # RASR NEURAL LENGTH DECODING

                          decoding_alias_addon = "rasr_decoding_limit12_pruning12.0_%s-recomb_neural-length_max-seg-len-%s_length-scale-%s%s" % ("vit" if vit_recomb else "no", 20, length_scale, "length-norm" if length_norm else "")

                          if "bpe-eos" not in name:
                            new_rasr_decoding_opts = copy.deepcopy(rasr_decoding_opts)
                            new_rasr_decoding_opts.update(
                              dict(
                                word_end_pruning_limit=12, word_end_pruning=12.0,
                                label_pruning_limit=12, label_pruning=12.0,
                                corpus_path=corpus_files[eval_corpus_name], feature_cache_path=feature_cache_files[eval_corpus_name]))
                            if "bpe-eos" in name:
                              new_rasr_decoding_opts["label_file_path"] = Path("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/bpe/irie/bpe_label_file_w_add_eos")

                            ctm_results = run_rasr_decoding(
                              segment_path=None, mem_rqmt=mem_rqmt, simple_beam_search=True, length_norm=length_norm,
                              full_sum_decoding=False, blank_update_history=blank_update_history,
                              allow_word_end_recombination=allow_word_end_recombination, loop_update_history=True,
                              allow_label_recombination=allow_label_recombination, max_seg_len=20, debug=False,
                              compile_config=compile_config.get_config(), alias_addon=decoding_alias_addon,
                              rasr_exe_path=rasr_flf_tool, model_checkpoint=checkpoint, num_epochs=epoch,
                              time_rqmt=30 if "bpe-eos" in name else 24, gpu_rqmt=1, **new_rasr_decoding_opts)
                            run_eval(ctm_file=ctm_results, reference=eval_ref_files[eval_corpus_name], name=name,
                                     dataset_key=eval_corpus_name, num_epochs=epoch, alias_addon=decoding_alias_addon)

                            realign_alias_addon = "rasr_realign_limit12_pruning12.0_%s-recomb_neural-length_max-seg-len-%s_length-scale-%s" % (
                            "vit" if vit_recomb else "no", max_seg_len, length_scale)

                            if epoch == 150 and length_scale in [1., .0]:
                              cv_realignment = run_rasr_realignment(
                                compile_config=compile_config.get_config(),
                                alias_addon=realign_alias_addon,
                                segment_path=cv_segments, loop_update_history=True, blank_update_history=True, name=name,
                                corpus_path=corpus_files["train"], lexicon_path=bpe_phon_lexicon_path if params["config"][
                                                                                                           "label_type"] in ["bpe", "bpe-seg-wo-length-realign", "bpe-seg-wo-length-retrain1-realign"] else bpe_sil_phon_lexicon_path,
                                allophone_path=total_data["bpe"]["allophones"],
                                state_tying_path=total_data["bpe"]["state_tying"],
                                feature_cache_path=feature_cache_files["train"], num_epochs=epoch,
                                label_file=total_data["bpe"]["rasr_label_file"],
                                label_pruning=12.0,
                                label_pruning_limit=5000, label_recombination_limit=-1, blank_label_index=targetb_blank_idx,
                                model_checkpoint=checkpoint, context_size=-1, reduction_factors=time_red,
                                rasr_nn_trainer_exe_path=rasr_nn_trainer, start_label_index=sos_idx,
                                rasr_am_trainer_exe_path=rasr_am_trainer, num_classes=targetb_blank_idx + 1, time_rqmt=4,
                                blank_allophone_state_idx=4119 if params["config"]["label_type"] in ["bpe", "bpe-seg-wo-length-realign", "bpe-seg-wo-length-retrain1-realign"] else 4123,
                                max_segment_len=max_seg_len, mem_rqmt=8, length_norm=False, data_key="cv")

                            search_error_opts = copy.deepcopy(cv_data_opts)
                            cv_align = search_error_opts.pop("alignment")
                            cv_segments = search_error_opts.pop("segment_file")
                            feed_config_load = config_class(task="train", length_scale=length_scale,
                                                            post_config={
                                                              "cleanup_old_models": {"keep_last_n": 1, "keep_best_n": 1,
                                                                                     "keep": num_epochs}},
                                                            train_data_opts=train_data_opts, cv_data_opts=cv_data_opts,
                                                            devtrain_data_opts=devtrain_data_opts,
                                                            **config_params).get_config()
                            feed_config_load.config["load"] = checkpoint
                            if name == "seg.bpe-with-sil-split-silv2.full-ctx.time-red6.fast-rec.fast-rec-full.sep-sil-model-pooling.seg.mlp-att.ctx-w-bias.am2048.frame-length-model-in_am+prev-out-embed.no-ctx-reg.bpe-sil-segs":
                              feed_config_load.config["network"]["label_model"]["unit"].update({
                                "label_log_prob": {
                                  "class": "copy", "from": ["sil_log_prob", "label_log_prob1"], },
                                "label_log_prob1": {
                                  "class": "combine", "from": ["label_log_prob0", "non_sil_log_prob"], "kind": "add", },
                                "label_prob": {
                                  "activation": "exp", "class": "activation", "from": ["label_log_prob"],
                                  "is_output_layer": True, "loss": "ce",
                                  "loss_opts": {"focal_loss_factor": 0.0, "label_smoothing": 0.1},
                                  "target": "label_ground_truth", },
                              })
                            new_rasr_decoding_opts = copy.deepcopy(rasr_decoding_opts)
                            new_rasr_decoding_opts.update(
                              dict(
                                word_end_pruning_limit=12, word_end_pruning=12.0,
                                label_pruning_limit=12, label_pruning=12.0,
                                corpus_path=corpus_files["train"], feature_cache_path=feature_cache_files["train"]))
                            if "bpe-eos" in name:
                              new_rasr_decoding_opts["label_file_path"] = Path(
                                "/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/bpe/irie/bpe_label_file_w_add_eos")
                            search_align, ctm_results = calc_rasr_search_errors(
                              segment_path=cv_segments, mem_rqmt=mem_rqmt, simple_beam_search=True,
                              length_norm=length_norm,
                              ref_align=cv_align, num_classes=targetb_blank_idx + 1, num_epochs=epoch,
                              blank_idx=targetb_blank_idx, rasr_nn_trainer_exe=rasr_nn_trainer,
                              extern_sprint_rasr_config=returnn_train_rasr_configs["cv"],
                              train_config=feed_config_load, loop_update_history=True,
                              full_sum_decoding=False, blank_update_history=blank_update_history,
                              allow_word_end_recombination=allow_word_end_recombination,
                              allow_label_recombination=allow_label_recombination,
                              max_seg_len=20, debug=False,
                              compile_config=compile_config.get_config(),
                              alias_addon=decoding_alias_addon, rasr_exe_path=rasr_flf_tool,
                              model_checkpoint=checkpoint, time_rqmt=48, gpu_rqmt=1,
                              model_type="seg", label_name="alignment",
                              orig_idx=1031 if "bpe-eos" in name else None, new_idx=0 if "bpe-eos" in name else None,
                              **new_rasr_decoding_opts)

                            if "bpe-eos" in name:
                              cv_segments = Path(
                                "/u/schmitt/experiments/transducer/config/dependencies/seg_cv_head3000")

                            if name in [
                              "seg.win-size8.bpe.lr-meas-lab.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs",
                              "seg.bpe.lr-meas-lab.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs",
                              "seg.win-size520.bpe.lr-meas-lab.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.chunk-size-None.all-segs"
                            ]:
                              run_eval(ctm_file=ctm_results, reference=stm_jobs["cv"].out_stm_path, name=name,
                                       dataset_key="cv", num_epochs=epoch, alias_addon=decoding_alias_addon)

                          else:
                            search_beam_size = 3000

                            new_rasr_decoding_opts = copy.deepcopy(rasr_decoding_opts)
                            new_rasr_decoding_opts.update(
                              dict(
                                word_end_pruning_limit=search_beam_size,
                                word_end_pruning=12.0,
                                label_pruning_limit=search_beam_size,
                                label_pruning=12.0,
                                corpus_path=corpus_files["dev"], feature_cache_path=feature_cache_files["dev"],
                                label_file_path=Path(
                                "/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/bpe/irie/bpe_label_file_w_add_eos"))
                            )

                            ctm_400 = run_rasr_decoding_parallel(
                              segment_path=Path("/work/asr3/zeyer/schmitt/tests/swb1/bpe-transducer_decoding-test/dev_non_empty_segments"),
                              mem_rqmt=64, simple_beam_search=True,
                              length_norm=length_norm, concurrent=8,
                              full_sum_decoding=False, blank_update_history=blank_update_history,
                              allow_word_end_recombination=allow_word_end_recombination, loop_update_history=True,
                              allow_label_recombination=allow_label_recombination, max_seg_len=20, debug=False,
                              compile_config=compile_config.get_config(), alias_addon=decoding_alias_addon + "_dev_400" + "_beam_" + str(search_beam_size) + "parallel_8-gpu",
                              rasr_exe_path=rasr_flf_tool, model_checkpoint=checkpoint, num_epochs=epoch,
                              time_rqmt=30, **new_rasr_decoding_opts)
                            run_eval(
                              ctm_file=ctm_400,
                              reference=Path(
                                "/work/asr3/zeyer/schmitt/tests/swb1/bpe-transducer_decoding-test/dev_non_empty.stm"),
                              name=name,
                              dataset_key="dev_400", num_epochs=epoch,
                              alias_addon=decoding_alias_addon + "_dev_400" + "_beam_" + str(search_beam_size) + "parallel_8-gpu")

                            new_rasr_decoding_opts = copy.deepcopy(rasr_decoding_opts)
                            new_rasr_decoding_opts.update(
                              dict(
                                word_end_pruning_limit=12,
                                word_end_pruning=12.0,
                                label_pruning_limit=12,
                                label_pruning=12.0,
                                corpus_path=corpus_files["train"], feature_cache_path=feature_cache_files["train"],
                                label_file_path=Path(
                                  "/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/bpe/irie/bpe_label_file_w_add_eos"))
                            )
                            cv_corpus_splits_job = ShuffleAndSplitSegmentsJob(
                              cv_segments,
                              split={i: 0.1 for i in range(10)}
                            )
                            search_align, ctm_results = calc_rasr_search_errors_parallel(
                              segment_path=cv_corpus_splits_job.out_segments[0],
                              mem_rqmt=64, simple_beam_search=True, concurrent=8,
                              length_norm=length_norm,
                              ref_align=cv_align, num_classes=targetb_blank_idx + 1, num_epochs=epoch,
                              blank_idx=targetb_blank_idx, rasr_nn_trainer_exe=rasr_nn_trainer,
                              extern_sprint_rasr_config=returnn_train_rasr_configs["cv"],
                              train_config=feed_config_load, loop_update_history=True,
                              full_sum_decoding=False, blank_update_history=blank_update_history,
                              allow_word_end_recombination=allow_word_end_recombination,
                              allow_label_recombination=allow_label_recombination,
                              max_seg_len=20, debug=False,
                              compile_config=compile_config.get_config(),
                              alias_addon=decoding_alias_addon + "_cv_300" + "_beam_" + str(search_beam_size) + "parallel_8-gpu",
                              rasr_exe_path=rasr_flf_tool,
                              model_checkpoint=checkpoint, time_rqmt=30,
                              model_type="seg", label_name="alignment",
                              orig_idx=None, new_idx=None,
                              **new_rasr_decoding_opts)


                          if (name == "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs" and length_scale in [1.] and epoch == 150) or \
                                  name == "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs" and length_scale == 0.0 and epoch == 150 or \
                                  (name == "seg.bpe-seg-wo-length-realign.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.length-loss-scale-0.0.no-ctx-reg.all-segs" and length_scale == 0.0 and epoch == 150):
                            for use_gpu in (True,):
                              nums_concurrent = (10,) if not use_gpu else (8,)
                              for num_concurrent in nums_concurrent:
                                parallel_alias_addon = "_%d-concurrent_%sgpu" % (num_concurrent, "" if use_gpu else "no-")
                                train_realignment = run_rasr_realignment_parallel(
                                  compile_config=compile_config.get_config(),
                                  alias_addon=realign_alias_addon + parallel_alias_addon, concurrent=num_concurrent, use_gpu=use_gpu,
                                  segment_path=train_segments, loop_update_history=True, blank_update_history=True, name=name,
                                  corpus_path=corpus_files["train"], lexicon_path=bpe_phon_lexicon_path if params["config"][
                                                                                                             "label_type"] in ["bpe", "bpe-seg-wo-length-realign", "bpe-seg-wo-length-retrain1-realign"] else bpe_sil_phon_lexicon_path,
                                  allophone_path=total_data["bpe"]["allophones"],
                                  state_tying_path=total_data["bpe"]["state_tying"],
                                  feature_cache_path=feature_cache_files["train"], num_epochs=epoch,
                                  label_file=total_data["bpe"]["rasr_label_file"],
                                  label_pruning=12.0,
                                  label_pruning_limit=5000, label_recombination_limit=-1,
                                  blank_label_index=targetb_blank_idx,
                                  model_checkpoint=checkpoint, context_size=-1, reduction_factors=time_red,
                                  rasr_nn_trainer_exe_path=rasr_nn_trainer, start_label_index=sos_idx,
                                  rasr_am_trainer_exe_path=rasr_am_trainer, num_classes=targetb_blank_idx + 1, time_rqmt=24,
                                  blank_allophone_state_idx=4119 if params["config"]["label_type"] in ["bpe", "bpe-seg-wo-length-realign", "bpe-seg-wo-length-retrain1-realign"] else 4123,
                                  max_segment_len=20, mem_rqmt=12, length_norm=False, data_key="train")

                          calc_align_stats(
                            alignment=search_align, blank_idx=targetb_blank_idx, seq_filter_file=cv_segments,
                            sil_idx=99999,  # set arbitrarily if there is no silence
                            alias=name + "/" + decoding_alias_addon + "/cv_search_align_stats_epoch-%s" % epoch)

                          seq_tags = [
                              "switchboard-1/sw02102A/sw2102A-ms98-a-0092",
                              "switchboard-1/sw02102A/sw2102A-ms98-a-0090", "switchboard-1/sw02025A/sw2025A-ms98-a-0035",
                              "switchboard-1/sw02102A/sw2102A-ms98-a-0002", "switchboard-1/sw02038B/sw2038B-ms98-a-0069"
                            ]
                          if name in [
                            "seg.win-size8.bpe.lr-meas-lab.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs",
                            "seg.bpe.lr-meas-lab.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs"]:
                            seq_tags += [
                              "switchboard-1/sw02102A/sw2102A-ms98-a-0053", "switchboard-1/sw02102A/sw2102A-ms98-a-0003"
                            ]

                          if name in [
                            "seg.win-size520.bpe.lr-meas-lab.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.chunk-size-None.all-segs"
                          ]:
                            seq_tags += [
                              "switchboard-1/sw02252A/sw2252A-ms98-a-0094",
                              "switchboard-1/sw02273B/sw2273B-ms98-a-0113"
                            ]

                          if "pooling-att" not in name:
                            for seq_tag in seq_tags:

                              group_alias = name + "/neural-length_analysis_epoch-%s_length-scale-%s_%s-recomb_max-seg-len-%s%s" % (epoch, length_scale, "no" if not vit_recomb else "vit", max_seg_len, "_length-norm" if length_norm else "")

                              alignments = [cv_align, search_align]
                              if epoch == 150 and length_scale in [1., .0] and "bpe-eos" not in name:
                                alignments.append(cv_realignment)

                              for align_alias, align in zip(["ground-truth", "search", "realign"],
                                                            alignments):
                                vocab_file = bpe_vocab["vocab_file"] if params["config"]["label_type"] == "bpe" else \
                                total_data["bpe-with-sil"]["json_vocab"]
                                dump_att_weights_job = DumpAttentionWeightsJob(returnn_config=feed_config_load,
                                  model_type="seg", rasr_config=returnn_train_rasr_configs["cv"],
                                  blank_idx=targetb_blank_idx, label_name="alignment", rasr_nn_trainer_exe=rasr_nn_trainer,
                                  hdf_targets=align, seq_tag=seq_tag, )
                                dump_att_weights_job.add_alias(
                                  group_alias + "/" + seq_tag.replace("/", "_") + "/att_weights_%s_%s" % (
                                  align_alias, epoch))
                                tk.register_output(dump_att_weights_job.get_one_alias(), dump_att_weights_job.out_data)

                                plot_weights_job = PlotAttentionWeightsJob(data_path=dump_att_weights_job.out_data,
                                                                           blank_idx=targetb_blank_idx,
                                                                           json_vocab_path=vocab_file, time_red=6,
                                                                           seq_tag=seq_tag)
                                plot_weights_job.add_alias(
                                  group_alias + "/" + seq_tag.replace("/", "_") + "/plot_att_weights_%s_%s" % (
                                  align_alias, epoch))
                                tk.register_output(plot_weights_job.get_one_alias(), plot_weights_job.out_plot)

                              compare_aligns_job = CompareAlignmentsJob(hdf_align1=cv_align, hdf_align2=search_align,
                                seq_tag=seq_tag, blank_idx1=targetb_blank_idx, blank_idx2=targetb_blank_idx,
                                vocab1=vocab_file, vocab2=vocab_file, name1="ground_truth", name2="search_alignment")
                              compare_aligns_job.add_alias(
                                group_alias + "/" + seq_tag.replace("/", "_") + "/search-align-compare")
                              tk.register_output(compare_aligns_job.get_one_alias(), compare_aligns_job.out_align)

                              if epoch == 150 and length_scale in [1., .0] and "bpe-eos" not in name:
                                compare_aligns_job = CompareAlignmentsJob(
                                  hdf_align1=cv_align, hdf_align2=cv_realignment,
                                  seq_tag=seq_tag, blank_idx1=targetb_blank_idx,
                                  blank_idx2=targetb_blank_idx,
                                  vocab1=vocab_file, vocab2=vocab_file,
                                  name1="ground_truth", name2="realignment")
                                compare_aligns_job.add_alias(
                                  group_alias + "/" + seq_tag.replace("/", "_") + "/realign-compare")
                                tk.register_output(compare_aligns_job.get_one_alias(), compare_aligns_job.out_align)

              if epoch in [150] and (name == "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs" or "bpe-eos" in name or "bpe-seg-wo-length-realign" in name or "bpe-seg-wo-length-retrain1-realign" in name) and params["config"]["length_model_type"] == "frame":
                length_scales = [1., 0.0]
                if name == "seg.bpe-with-sil-split-sil.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.ctx-w-bias.am2048.frame-length-model-in_am+prev-out-embed.no-ctx-reg.bpe-sil-segs":
                  length_scales += [.01, .1, .3, .5, .7, .9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.]
                elif "bpe-eos" in name:
                  length_scales = [.0]
                for length_scale in length_scales:
                  if params["config"]["label_type"] in ["bpe", "bpe_sil_wo_sil"]:
                    max_seg_lens = [25]
                  else:
                    max_seg_lens = [20]
                  if name == "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs" and length_scale == 0.0:
                    max_seg_lens = [25]
                  for max_seg_len in max_seg_lens:
                    limits = [12]
                    for limit in limits:
                      vit_recombs = [True, False]
                      if max_seg_len in [10, 30, 40]:
                        vit_recombs = [True]
                      if limit == 5000:
                        vit_recombs = [True]
                      if "bpe-eos" in name:
                        vit_recombs = [True]
                      for vit_recomb in vit_recombs:
                        if length_scale == 0.0:
                          length_norms = [True]
                          if name == "seg.bpe-with-sil-split-sil.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.ctx-w-bias.am2048.frame-length-model-in_am+prev-out-embed.no-ctx-reg.bpe-sil-segs":
                            length_norms += [False]
                          beam_searches = [True]
                        elif max_seg_len in [10, 30, 40]:
                          length_norms = [False]
                          beam_searches = [True]
                        else:
                          length_norms = [True, False]
                          beam_searches = [True]
                        for length_norm in length_norms:
                          for beam_search in beam_searches:
                            seg_selections = ["all"]
                            for seg_selection in seg_selections:
                              global_length_vars = [None]
                              silence_splits = [False]
                              net_types = ["default"]
                              if name == "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs" and length_scale == 0.0:
                                net_types += ["global_import"]
                              elif name == "seg.bpe-with-sil-split-sil.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.ctx-w-bias.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.bpe-sil-segs" and length_scale == 0.0:
                                net_types += ["global_import"]
                                silence_splits += [True]
                              elif "bpe-eos" in name:
                                net_types = ["global_import"]
                              for net_type in net_types:
                                if name == "seg.bpe-with-sil-split-sil.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.ctx-w-bias.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.bpe-sil-segs" and net_type != "global_import" and limit == 5000:
                                  continue
                                for glob_length_var in global_length_vars:
                                  for silence_split in silence_splits:
                                    # parameters to use label dep length model
                                    label_dep_params = copy.deepcopy(config_params)
                                    label_dep_params.pop("length_model_type")
                                    label_dep_params.pop("max_seg_len")
                                    label_dep_params.pop("pretrain")
                                    label_dep_params.update(dict(
                                      length_scale=length_scale, label_dep_length_model=True, length_model_type="seg-static",
                                      label_dep_means=total_data[params["config"]["label_type"]]["train"]["time-red-%s" % time_red]["label_dep_mean_lens"]))
                                    # compile config for RASR
                                    compile_config = config_class(
                                      task="eval", feature_stddev=3., max_seg_len=max_seg_len,
                                      network_type=net_type,
                                      global_length_var=glob_length_var,
                                      **label_dep_params).get_config()
                                    feed_config_load_new = config_class(
                                      task="eval", pretrain=False, max_seg_len=max_seg_len,
                                      network_type=net_type,
                                      global_length_var=glob_length_var,
                                      **label_dep_params).get_config()

                                    # print(name)

                                    if net_type == "global_import":
                                      if name == "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs" or "bpe-eos" in name:
                                        checkpoint = Checkpoint(index_path=Path(
                                          "/u/schmitt/experiments/transducer/alias/glob.best-model.bpe.time-red6.am2048.1pretrain-reps.no-weight-feedback.ctx-use-bias.pretrain-like-seg.all-segs/train/output/models/epoch.150.index"))
                                      else:
                                        if silence_split:
                                          checkpoint = Checkpoint(index_path=Path(
                                            "/u/schmitt/experiments/transducer/alias/glob.best-model.bpe-with-sil-split-sil.time-red6.am2048.1pretrain-reps.ctx-use-bias.pretrain-like-seg.bpe-sil-segs/train/output/models/epoch.150.index"))
                                        else:
                                          checkpoint = Checkpoint(index_path=Path(
                                            "/u/schmitt/experiments/transducer/alias/glob.best-model.bpe-with-sil.time-red6.am2048.1pretrain-reps.no-weight-feedback.ctx-use-bias.pretrain-like-seg.bpe-sil-segs/train/output/models/epoch.150.index"))
                                    else:
                                      checkpoint = checkpoints[epoch]

                                    feed_config_load_new.config["load"] = checkpoint

                                    # RASR LABEL DEP LENGTH DECODING

                                    if vit_recomb:
                                      blank_update_history = True
                                      allow_label_recombination = True
                                      allow_word_end_recombination = True
                                      mem_rqmt = 64
                                    else:
                                      blank_update_history = True
                                      allow_label_recombination = False
                                      allow_word_end_recombination = False
                                      mem_rqmt = 64

                                    if not vit_recomb and length_scale in [0.0, 0.01, 0.1, 0.5] and max_seg_len == 20:
                                      if not length_norm:
                                        label_pruning = 1.0
                                      else:
                                        label_pruning = 0.1
                                      mem_rqmt = 64
                                    else:
                                      label_pruning = 12.0

                                    alias_addon = "rasr_limit%s_pruning%s_%s-recomb_label-dep-length-%s_max-seg-len-%s_length-scale-%s%s%s_%s-segments%s%s" % (limit, label_pruning, "vit" if vit_recomb else "no", "glob-var-%s" % glob_length_var, max_seg_len, length_scale, "" if not length_norm else "_length-norm", "" if not beam_search else "_beam_search", "all" if seg_selection == "all" else "red", "" if net_type == "default" else "_" + net_type, "" if not silence_split else "_split-sil")

                                    if "bpe-eos" in name or "bpe-seg-wo-length-realign" in name or "bpe-seg-wo-length-retrain1-realign" in name:
                                      search_beam_size = 3000

                                      new_rasr_decoding_opts = copy.deepcopy(rasr_decoding_opts)
                                      new_rasr_decoding_opts.update(
                                        dict(word_end_pruning_limit=search_beam_size, word_end_pruning=12.0,
                                          label_pruning_limit=search_beam_size, label_pruning=12.0,
                                          corpus_path=corpus_files["dev"],
                                          feature_cache_path=feature_cache_files["dev"], label_file_path=Path(
                                            "/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/bpe/irie/bpe_label_file_w_add_eos")))

                                      ctm_400 = run_rasr_decoding_parallel(segment_path=Path(
                                        "/work/asr3/zeyer/schmitt/tests/swb1/bpe-transducer_decoding-test/dev_non_empty_segments"),
                                        mem_rqmt=64, simple_beam_search=True, length_norm=length_norm, concurrent=8,
                                        full_sum_decoding=False, blank_update_history=blank_update_history,
                                        allow_word_end_recombination=allow_word_end_recombination,
                                        loop_update_history=True, allow_label_recombination=allow_label_recombination,
                                        max_seg_len=20, debug=False, compile_config=compile_config,
                                        alias_addon=alias_addon + "_dev_400" + "_beam_" + str(
                                          search_beam_size) + "parallel_8-gpu", rasr_exe_path=rasr_flf_tool,
                                        model_checkpoint=checkpoint, num_epochs=epoch, time_rqmt=30,
                                        **new_rasr_decoding_opts)
                                      run_eval(ctm_file=ctm_400, reference=Path(
                                        "/work/asr3/zeyer/schmitt/tests/swb1/bpe-transducer_decoding-test/dev_non_empty.stm"),
                                        name=name, dataset_key="dev_400", num_epochs=epoch,
                                        alias_addon=alias_addon + "_dev_400" + "_beam_" + str(
                                          search_beam_size) + "parallel_8-gpu")

                                      # new_rasr_decoding_opts = copy.deepcopy(rasr_decoding_opts)
                                      # new_rasr_decoding_opts.update(
                                      #   dict(word_end_pruning_limit=12, word_end_pruning=12.0, label_pruning_limit=12,
                                      #     label_pruning=12.0, corpus_path=corpus_files["train"],
                                      #     feature_cache_path=feature_cache_files["train"], label_file_path=Path(
                                      #       "/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/bpe/irie/bpe_label_file_w_add_eos")))
                                      # cv_corpus_splits_job = ShuffleAndSplitSegmentsJob(cv_segments,
                                      #   split={i: 0.1 for i in range(10)})
                                      # search_align, ctm_results = calc_rasr_search_errors_parallel(
                                      #   segment_path=cv_corpus_splits_job.out_segments[0], mem_rqmt=64,
                                      #   simple_beam_search=True, concurrent=8, length_norm=length_norm,
                                      #   ref_align=cv_align, num_classes=targetb_blank_idx + 1, num_epochs=epoch,
                                      #   blank_idx=targetb_blank_idx, rasr_nn_trainer_exe=rasr_nn_trainer,
                                      #   extern_sprint_rasr_config=returnn_train_rasr_configs["cv"],
                                      #   train_config=feed_config_load, loop_update_history=True,
                                      #   full_sum_decoding=False, blank_update_history=blank_update_history,
                                      #   allow_word_end_recombination=allow_word_end_recombination,
                                      #   allow_label_recombination=allow_label_recombination, max_seg_len=20,
                                      #   debug=False, compile_config=compile_config,
                                      #   alias_addon=alias_addon + "_cv_300" + "_beam_" + str(
                                      #     search_beam_size) + "parallel_8-gpu", rasr_exe_path=rasr_flf_tool,
                                      #   model_checkpoint=checkpoint, time_rqmt=30, model_type="seg",
                                      #   label_name="alignment", orig_idx=None, new_idx=None, **new_rasr_decoding_opts)
                                    else:
                                      new_rasr_decoding_opts = copy.deepcopy(rasr_decoding_opts)
                                      new_rasr_decoding_opts.update(
                                        dict(
                                          word_end_pruning_limit=limit,
                                          word_end_pruning=label_pruning,
                                          label_pruning_limit=limit,
                                          label_pruning=label_pruning))
                                      if net_type != "default":
                                        if "sil" not in params["config"]["label_type"]:
                                          new_rasr_decoding_opts.update(dict(
                                            label_file_path=Path(
                                              "/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/bpe/irie/bpe_label_file_w_add_eos")
                                          ))
                                        else:
                                          new_rasr_decoding_opts.update(dict(label_file_path=Path(
                                            "/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/bpe/irie/bpe_sil_label_file_w_add_eos")))

                                      new_rasr_decoding_opts.update(dict(
                                        segment_path=None if seg_selection == "all" else Path(
                                          "/work/asr3/zeyer/schmitt/tests/swb1/bpe-transducer_decoding-test/hub5_00_10div"),
                                        mem_rqmt=128 if limit == 5000 else mem_rqmt,
                                        simple_beam_search=False if not beam_search else True, length_norm=length_norm,
                                        full_sum_decoding=False, blank_update_history=blank_update_history,
                                        allow_word_end_recombination=allow_word_end_recombination,
                                        loop_update_history=True, allow_label_recombination=allow_label_recombination,
                                        max_seg_len=max_seg_len, debug=False, compile_config=compile_config,
                                        alias_addon=alias_addon, rasr_exe_path=rasr_flf_tool, model_checkpoint=checkpoint,
                                        num_epochs=epoch, time_rqmt=72 if limit == 5000 else 24, gpu_rqmt=1
                                      ))

                                      cv_realignment = None
                                      if name == "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs" and net_type == "global_import" or \
                                              name == "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs" and length_scale == 0.0:
                                        cv_realignment = run_rasr_realignment(
                                          compile_config=compile_config,
                                          alias_addon=alias_addon + "_realign",
                                          segment_path=cv_segments,
                                          loop_update_history=True, blank_update_history=True,
                                          name=name,
                                          corpus_path=corpus_files["train"],
                                          lexicon_path=Path("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/bpe/irie/bpe_lexicon_w_eos"),
                                          allophone_path=Path("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/bpe/irie/bpe_allophones_w_eos"),
                                          state_tying_path=Path("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/bpe/irie/bpe_state_tying_w_eos"),
                                          feature_cache_path=feature_cache_files["train"], num_epochs=epoch,
                                          label_file=Path("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/bpe/irie/bpe_label_file_w_add_eos"),
                                          label_pruning=20.0,
                                          label_pruning_limit=5000, label_recombination_limit=-1,
                                          blank_label_index=targetb_blank_idx,
                                          model_checkpoint=checkpoint, context_size=-1, reduction_factors=time_red,
                                          rasr_nn_trainer_exe_path=rasr_nn_trainer, start_label_index=sos_idx,
                                          rasr_am_trainer_exe_path=rasr_am_trainer, num_classes=targetb_blank_idx + 1,
                                          time_rqmt=4,
                                          blank_allophone_state_idx=4119 if params["config"][
                                                                              "label_type"] == "bpe" else 4123,
                                          max_segment_len=20, mem_rqmt=8, length_norm=False, data_key="cv")

                                        switch_label_job = SwitchLabelInAlignmentJob(alignment=cv_realignment,
                                          new_idx=1030, orig_idx=1031)
                                        tk.register_output("glob-att-realign/cv", switch_label_job.out_alignment)
                                        cv_realignment = switch_label_job.out_alignment

                                        train_realignment = run_rasr_realignment_parallel(
                                          use_gpu=True, concurrent=8,
                                          compile_config=compile_config,
                                          alias_addon=alias_addon + "_realign",
                                          segment_path=train_segments,
                                          loop_update_history=True, blank_update_history=True,
                                          name=name,
                                          corpus_path=corpus_files["train"],
                                          lexicon_path=Path(
                                            "/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/bpe/irie/bpe_lexicon_w_eos"),
                                          allophone_path=Path(
                                            "/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/bpe/irie/bpe_allophones_w_eos"),
                                          state_tying_path=Path(
                                            "/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/bpe/irie/bpe_state_tying_w_eos"),
                                          feature_cache_path=feature_cache_files["train"], num_epochs=epoch,
                                          label_file=Path(
                                            "/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/bpe/irie/bpe_label_file_w_add_eos"),
                                          label_pruning=20.0,
                                          label_pruning_limit=5000, label_recombination_limit=-1,
                                          blank_label_index=targetb_blank_idx,
                                          model_checkpoint=checkpoint, context_size=-1, reduction_factors=time_red,
                                          rasr_nn_trainer_exe_path=rasr_nn_trainer, start_label_index=sos_idx,
                                          rasr_am_trainer_exe_path=rasr_am_trainer, num_classes=targetb_blank_idx + 1,
                                          time_rqmt=24,
                                          blank_allophone_state_idx=4119 if params["config"][
                                                                              "label_type"] == "bpe" else 4123,
                                          max_segment_len=20, mem_rqmt=12, length_norm=False, data_key="train")

                                        switch_label_job = SwitchLabelInAlignmentJob(alignment=train_realignment,
                                                                                     new_idx=1030, orig_idx=1031)
                                        tk.register_output("glob-att-realign/train", switch_label_job.out_alignment)
                                        train_realignment = switch_label_job.out_alignment

                                      # ctm_results = run_rasr_decoding(
                                      #   segment_path=None if seg_selection == "all" else Path("/work/asr3/zeyer/schmitt/tests/swb1/bpe-transducer_decoding-test/hub5_00_10div"),
                                      #   mem_rqmt=128 if limit == 5000 else mem_rqmt, simple_beam_search=False if not beam_search else True, length_norm=length_norm,
                                      #   full_sum_decoding=False, blank_update_history=blank_update_history,
                                      #   allow_word_end_recombination=allow_word_end_recombination, loop_update_history=True,
                                      #   allow_label_recombination=allow_label_recombination, max_seg_len=max_seg_len, debug=False,
                                      #   compile_config=compile_config, alias_addon=alias_addon,
                                      #   rasr_exe_path=rasr_flf_tool, model_checkpoint=checkpoint,
                                      #   num_epochs=epoch, time_rqmt=72 if limit == 5000 else 24, gpu_rqmt=1, **new_rasr_decoding_opts)
                                      # run_eval(ctm_file=ctm_results,
                                      #          reference=Path("/u/tuske/bin/switchboard/hub5e_00.2.stm") if seg_selection == "all" else Path("/work/asr3/zeyer/schmitt/tests/swb1/bpe-transducer_decoding-test/hub5_00_stm_10div"),
                                      #          name=name,
                                      #          dataset_key="dev", num_epochs=epoch, alias_addon=alias_addon)

                                      search_error_opts = copy.deepcopy(cv_data_opts)
                                      cv_align = search_error_opts.pop("alignment")
                                      cv_segments = search_error_opts.pop("segment_file")
                                      rasr_search_error_decoding_opts = copy.deepcopy(rasr_decoding_opts)
                                      rasr_search_error_decoding_opts.update(
                                        dict(word_end_pruning_limit=limit, word_end_pruning=label_pruning, label_pruning_limit=limit, label_pruning=label_pruning,
                                          corpus_path=corpus_files["train"], feature_cache_path=feature_cache_files["train"]))
                                      if net_type != "default":
                                        if "sil" not in params["config"]["label_type"]:
                                          rasr_search_error_decoding_opts.update(dict(label_file_path=Path(
                                            "/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/bpe/irie/bpe_label_file_w_add_eos")))
                                        else:
                                          rasr_search_error_decoding_opts.update(dict(label_file_path=Path(
                                            "/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/bpe/irie/bpe_sil_label_file_w_add_eos")))

                                      rasr_search_error_decoding_opts.update(dict(
                                        segment_path=cv_segments,
                                        mem_rqmt=128 if limit == 5000 else mem_rqmt,
                                        simple_beam_search=False if not beam_search else True, length_norm=length_norm,
                                        ref_align=cv_realignment if name == "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs" and net_type == "global_import" else cv_align,
                                        num_classes=targetb_blank_idx + 1, num_epochs=epoch,
                                        blank_idx=targetb_blank_idx, rasr_nn_trainer_exe=rasr_nn_trainer,
                                        extern_sprint_rasr_config=returnn_train_rasr_configs["cv"],
                                        train_config=feed_config_load_new, loop_update_history=True, full_sum_decoding=False,
                                        blank_update_history=blank_update_history,
                                        allow_word_end_recombination=allow_word_end_recombination,
                                        allow_label_recombination=allow_label_recombination, max_seg_len=max_seg_len,
                                        debug=False, compile_config=compile_config, alias_addon=alias_addon,
                                        rasr_exe_path=rasr_flf_tool, model_checkpoint=checkpoint,
                                        time_rqmt=72 if limit == 5000 else 24, gpu_rqmt=1,
                                        model_type="seg_lab_dep" if net_type == "default" else "global-import",
                                        label_name="alignment"
                                      ))

                                      search_align, ctm_results = start_rasr_recog_pipeline(
                                        ref_stm_path=Path("/u/tuske/bin/switchboard/hub5e_00.2.stm") if seg_selection == "all" else Path("/work/asr3/zeyer/schmitt/tests/swb1/bpe-transducer_decoding-test/hub5_00_stm_10div"),
                                        model_name=name, recog_epoch=epoch, rasr_decoding_opts=new_rasr_decoding_opts,
                                        rasr_search_error_decoding_opts=rasr_search_error_decoding_opts,
                                        blank_idx=targetb_blank_idx, cv_segments=cv_segments,
                                        recog_corpus_name="dev", alias_addon=alias_addon
                                      )
                                      if name == "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs" and net_type == "global_import":
                                        run_eval(ctm_file=ctm_results, reference=stm_jobs["cv"].out_stm_path, name=name,
                                                 dataset_key="cv", num_epochs=epoch, alias_addon=alias_addon)

                                      # alias_addon = "_rasr_search_errors_limit12_pruning12.0_%s-recomb_label-dep-length_max-seg-len-%s_length-scale-%s" % ("vit" if vit_recomb else "no", max_seg_len, length_scale)
                                      # search_align, ctm_results = calc_rasr_search_errors(
                                      #   segment_path=cv_segments, mem_rqmt=128 if limit == 5000 else mem_rqmt,
                                      #   simple_beam_search=False if not beam_search else True, length_norm=length_norm,
                                      #   ref_align=cv_align, num_classes=targetb_blank_idx+1, num_epochs=epoch, blank_idx=targetb_blank_idx,
                                      #   rasr_nn_trainer_exe=rasr_nn_trainer,
                                      #   extern_sprint_rasr_config=returnn_train_rasr_configs["cv"],
                                      #   train_config=feed_config_load, loop_update_history=True, full_sum_decoding=False,
                                      #   blank_update_history=blank_update_history, allow_word_end_recombination=allow_word_end_recombination,
                                      #   allow_label_recombination=allow_label_recombination, max_seg_len=max_seg_len, debug=False,
                                      #   compile_config=compile_config, alias_addon=alias_addon,
                                      #   rasr_exe_path=rasr_flf_tool, model_checkpoint=checkpoint, time_rqmt=72 if limit == 5000 else 24,
                                      #   gpu_rqmt=1, model_type="seg_lab_dep" if net_type == "default" else "global-import", label_name="alignment", **rasr_search_error_decoding_opts)
                                      # # run_eval(ctm_file=ctm_results, reference=stm_jobs["cv"].out_stm_path, name=name,
                                      # #          dataset_key="cv", num_epochs=epoch, alias_addon=alias_addon)
                                      #
                                      # calc_align_stats(
                                      #   alignment=search_align, blank_idx=targetb_blank_idx,
                                      #   seq_filter_file=cv_segments, alias=name + "/" + alias_addon + "/cv_search_align_stats_epoch-%s" % epoch)

                                      # if name == "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.all-segs":
                                      #   if alias_addon == "rasr_limit12_pruning12.0_no-recomb_label-dep-length-glob-var-None_max-seg-len-25_length-scale-0.0_length-norm_beam_search_all-segments_global_import":
                                      #     dump_non_blanks_job = DumpNonBlanksFromAlignmentJob(search_align,
                                      #       blank_idx=targetb_blank_idx)
                                      #     dump_non_blanks_job.add_alias("dump_non_blanks_" + alias_addon)
                                      #     search_aligns["global_import_segmental"] = search_align
                                      #     search_labels["global_import_segmental"] = dump_non_blanks_job.out_labels
                                      # if name == "seg.bpe-with-sil-split-sil.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.ctx-w-bias.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.bpe-sil-segs":
                                      #   if alias_addon == "rasr_limit12_pruning12.0_vit-recomb_label-dep-length-glob-var-None_max-seg-len-20_length-scale-0.0_length-norm_beam_search_all-segments_global_import_split-sil":
                                      #     dump_non_blanks_job = DumpNonBlanksFromAlignmentJob(search_align,
                                      #       blank_idx=targetb_blank_idx)
                                      #     dump_non_blanks_job.add_alias("dump_non_blanks_" + alias_addon)
                                      #     search_aligns["global_import_segmental_w_split_sil"] = search_align
                                      #     search_labels["global_import_segmental_w_split_sil"] = dump_non_blanks_job.out_labels

                                      # if alias_addon == "rasr_limit12_pruning0.1_no-recomb_label-dep-length-glob-var-None_max-seg-len-20_length-scale-0.0_length-norm_beam_search_all-segments_global_import_w_feedback":
                                      #   for search_align_alias, other_search_align in search_aligns.items():
                                      #     calc_search_err_job = CalcSearchErrorJob(
                                      #       returnn_config=train_config,
                                      #       rasr_config=returnn_train_rasr_configs["cv"],
                                      #       rasr_nn_trainer_exe=rasr_nn_trainer,
                                      #       segment_file=cv_segments,
                                      #       blank_idx=targetb_blank_idx,
                                      #       search_targets=other_search_align,
                                      #       ref_targets=cv_align, label_name="alignment",
                                      #       model_type="seg_lab_dep",
                                      #       max_seg_len=max_seg_len if max_seg_len is not None else -1,
                                      #       length_norm=length_norm)
                                      #     calc_search_err_job.add_alias(
                                      #       name + ("/%s/search_errors_%s_%d" % (alias_addon, search_align_alias, epoch)))
                                      #     alias = calc_search_err_job.get_one_alias()
                                      #     tk.register_output(alias + "search_errors", calc_search_err_job.out_search_errors)

                                      seq_tags = [
                                        "switchboard-1/sw02102A/sw2102A-ms98-a-0092",
                                        "switchboard-1/sw02022A/sw2022A-ms98-a-0002",
                                        "switchboard-1/sw02102A/sw2102A-ms98-a-0090",
                                        "switchboard-1/sw02025A/sw2025A-ms98-a-0035",
                                        "switchboard-1/sw02102A/sw2102A-ms98-a-0002",
                                        "switchboard-1/sw02038B/sw2038B-ms98-a-0069"
                                      ]

                                      if name == "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs" and net_type == "global_import" and vit_recomb:
                                        seq_tags += [
                                          "switchboard-1/sw02121A/sw2121A-ms98-a-0096",  # both correct
                                          "switchboard-1/sw02375B/sw2375B-ms98-a-0015",  # both correct
                                          "switchboard-1/sw02248A/sw2248A-ms98-a-0076",  # both correct
                                          "switchboard-1/sw02242B/sw2242B-ms98-a-0106",  # both correct
                                          "switchboard-1/sw02252A/sw2252A-ms98-a-0088",  # both correct
                                          "switchboard-1/sw02306B/sw2306B-ms98-a-0059",  # only segmental correct
                                          "switchboard-1/sw02107A/sw2107A-ms98-a-0042",  # only segmental correct
                                          "switchboard-1/sw02064A/sw2064A-ms98-a-0038",  # only segmental correct
                                          "switchboard-1/sw02230B/sw2230B-ms98-a-0075",  # only segmental correct
                                          "switchboard-1/sw02252A/sw2252A-ms98-a-0030",  # only segmental correct
                                          "switchboard-1/sw02102A/sw2102A-ms98-a-0039",  # only global correct
                                          "switchboard-1/sw02038B/sw2038B-ms98-a-0017",  # only global correct
                                          "switchboard-1/sw02107A/sw2107A-ms98-a-0085",  # only global correct
                                          "switchboard-1/sw02141A/sw2141A-ms98-a-0121",  # only global correct
                                          "switchboard-1/sw02161B/sw2161B-ms98-a-0059",  # only global correct
                                        ]

                                      if limit == 12 and seg_selection == "all" and "pooling-att" not in name:
                                        for seq_tag in seq_tags:
                                          group_alias = name + "/analysis_epoch-%s_length-scale-%s-%s_%s-recomb_max-seg-len-%s%s%s%s%s" % (epoch, length_scale, "glob-var-%s" % glob_length_var, "no" if not vit_recomb else "vit", max_seg_len,"" if not length_norm else "_length-norm", "" if not beam_search else "_beam_search", "" if net_type == "default" else "_" + net_type, "" if not silence_split else "_split-sil")

                                          # # cv_realignment = run_rasr_realignment(
                                          # #   compile_config=compile_config,
                                          # #   alias_addon="", segment_path=Path("/work/asr3/zeyer/schmitt/tests/swb1/bpe-transducer_decoding-test/cv_test_segments1"),
                                          # #   loop_update_history=True, blank_update_history=True if not vit_recomb else False, name=group_alias,
                                          # #   corpus_path=corpus_files["train"], lexicon_path=bpe_phon_lexicon_path if params["config"]["label_type"] == "bpe" else bpe_sil_phon_lexicon_path,
                                          # #   allophone_path=total_data[params["config"]["label_type"]]["allophones"],
                                          # #   state_tying_path=total_data[params["config"]["label_type"]]["state_tying"],
                                          # #   feature_cache_path=feature_cache_files["train"], num_epochs=epoch,
                                          # #   label_file=total_data[params["config"]["label_type"]]["rasr_label_file"], label_pruning=12.0,
                                          # #   label_pruning_limit=5000, label_recombination_limit=-1, blank_label_index=targetb_blank_idx,
                                          # #   model_checkpoint=checkpoint, context_size=-1, reduction_factors=time_red,
                                          # #   rasr_nn_trainer_exe_path=rasr_nn_trainer, start_label_index=sos_idx,
                                          # #   rasr_am_trainer_exe_path=rasr_am_trainer, num_classes=targetb_blank_idx + 1, time_rqmt=2,
                                          # #   blank_allophone_state_idx=4119 if params["config"]["label_type"] == "bpe" else 4123,
                                          # #   max_segment_len=max_seg_len, mem_rqmt=16)
                                          feed_config_load_new = config_class(task="train",
                                                                          post_config={
                                                                            "cleanup_old_models": {"keep_last_n": 1, "keep_best_n": 1,
                                                                                                   "keep": num_epochs}},
                                                                          network_type=net_type,
                                                                          train_data_opts=train_data_opts, cv_data_opts=cv_data_opts,
                                                                          devtrain_data_opts=devtrain_data_opts,
                                                                          max_seg_len=max_seg_len, **label_dep_params).get_config()
                                          feed_config_load_new.config["load"] = checkpoint
                                          feed_config_load_new.config["network"]["label_model"]["unit"]["label_prob"]["loss"] = None
                                          feed_config_load_new.config["network"]["label_model"]["unit"]["label_prob"]["is_output_layer"] = False
                                          feed_config_load_new.config["network"]["output"]["unit"]["emit_blank_prob"]["loss"] = None

                                          start_analysis_pipeline(
                                            group_alias=group_alias, feed_config=feed_config_load_new,
                                            vocab_file=bpe_vocab["vocab_file"] if params["config"]["label_type"] == "bpe" else total_data["bpe-with-sil"]["json_vocab"],
                                            cv_align=cv_align, search_align=search_align, rasr_config=returnn_train_rasr_configs["cv"],
                                            blank_idx=targetb_blank_idx, model_type="seg_lab_dep" if net_type == "default" else "global-import",
                                            rasr_nn_trainer_exe=rasr_nn_trainer, seq_tag=seq_tag, epoch=epoch,
                                            cv_realign=cv_realignment)
                                          #
                                          # for align_alias, align in zip(["ground-truth", "search", "realign"], [cv_align, search_align]):
                                          #   vocab_file = bpe_vocab["vocab_file"] if params["config"]["label_type"] == "bpe" else total_data["bpe-with-sil"]["json_vocab"]
                                          #   dump_att_weights_job = DumpAttentionWeightsJob(
                                          #     returnn_config=feed_config_load, model_type="seg_lab_dep" if net_type == "default" else "global-import",
                                          #     rasr_config=returnn_train_rasr_configs["cv"], blank_idx=targetb_blank_idx,
                                          #     label_name="alignment", rasr_nn_trainer_exe=rasr_nn_trainer, hdf_targets=align,
                                          #     seq_tag=seq_tag, )
                                          #   dump_att_weights_job.add_alias(group_alias + "/" + seq_tag.replace("/", "_") + "/att_weights_%s_%s" % (align_alias, epoch))
                                          #   tk.register_output(dump_att_weights_job.get_one_alias(), dump_att_weights_job.out_data)
                                          #
                                          #   plot_weights_job = PlotAttentionWeightsJob(data_path=dump_att_weights_job.out_data,
                                          #     blank_idx=targetb_blank_idx,
                                          #     json_vocab_path=vocab_file, time_red=6,
                                          #     seq_tag=seq_tag)
                                          #   plot_weights_job.add_alias(group_alias + "/" + seq_tag.replace("/", "_") + "/plot_att_weights_%s_%s" % (align_alias, epoch))
                                          #   tk.register_output(plot_weights_job.get_one_alias(), plot_weights_job.out_plot)
                                          #
                                          # compare_aligns_job = CompareAlignmentsJob(
                                          #   hdf_align1=cv_align, hdf_align2=search_align, seq_tag=seq_tag,
                                          #   blank_idx1=targetb_blank_idx, blank_idx2=targetb_blank_idx, vocab1=vocab_file, vocab2=vocab_file,
                                          #   name1="ground_truth", name2="search_alignment"
                                          # )
                                          # compare_aligns_job.add_alias(group_alias + "/" + seq_tag.replace("/", "_") + "/search-align-compare")
                                          # tk.register_output(compare_aligns_job.get_one_alias(), compare_aligns_job.out_align)
                                          #
                                          # # compare_aligns_job = CompareAlignmentsJob(hdf_align1=cv_align, hdf_align2=cv_realignment,
                                          # #   seq_tag=seq_tag, blank_idx1=targetb_blank_idx, blank_idx2=targetb_blank_idx,
                                          # #   vocab1=vocab_file, vocab2=vocab_file, name1="ground_truth",
                                          # #   name2="search_alignment")
                                          # # compare_aligns_job.add_alias(group_alias + "/" + seq_tag + "/realignment-compare")
                                          # # tk.register_output(compare_aligns_job.get_one_alias(), compare_aligns_job.out_align)



              # # example for calculating RASR search errors
              # cv_align = cv_data_opts.pop("alignment")
              # cv_segments = cv_data_opts.pop("segment_file")
              # new_rasr_decoding_opts = copy.deepcopy(rasr_decoding_opts)
              # new_rasr_decoding_opts.update(
              #   dict(
              #     word_end_pruning_limit=12, word_end_pruning=10.0, label_pruning_limit=12, label_pruning=10.0,
              #     corpus_path=corpus_files["train"], feature_cache_path=feature_cache_files["train"]))
              # alias_addon = "_rasr_beam_search12_no-recomb"
              # calc_rasr_search_errors(
              #   segment_path=cv_segments, mem_rqmt=8, simple_beam_search=True, ref_align=cv_align,
              #   num_classes=1032, num_epochs=epoch, blank_idx=1031, rasr_nn_trainer_exe=rasr_nn_trainer,
              #   extern_sprint_rasr_config=returnn_train_rasr_configs["cv"],
              #   train_config=train_config, loop_update_history=True,
              #   full_sum_decoding=False, blank_update_history=True,
              #   allow_word_end_recombination=False, allow_label_recombination=False,
              #   max_seg_len=None, debug=False, compile_config=compile_config.get_config(),
              #   alias_addon=alias_addon, rasr_exe_path=rasr_flf_tool,
              #   model_checkpoint=checkpoint, time_rqmt=time_rqmt,
              #   gpu_rqmt=1, **new_rasr_decoding_opts)

              # # example for realignment + retraining
              # cv_realignment = run_rasr_realignment(
              #   compile_config=compile_config.get_config(), alias_addon=alias_addon + "_cv",
              #   segment_path=cv_segments, loop_update_history=True, blank_update_history=True,
              #   name=name, corpus_path=corpus_files["train"], lexicon_path=bpe_sil_phon_lexicon_path,
              #   allophone_path=total_data["bpe-with-sil"]["allophones"],
              #   state_tying_path=total_data["bpe-with-sil"]["state_tying"], feature_cache_path=feature_cache_files["train"],
              #   num_epochs=epoch, label_file=total_data["bpe-with-sil"]["rasr_label_file"],
              #   label_pruning=50.0, label_pruning_limit=1000, label_recombination_limit=-1, blank_label_index=1031,
              #   model_checkpoint=checkpoint, context_size=-1, reduction_factors=time_red, rasr_nn_trainer_exe_path=rasr_nn_trainer,
              #   start_label_index=1030, rasr_am_trainer_exe_path=rasr_am_trainer, num_classes=1032, time_rqmt=3,
              #   blank_allophone_state_idx=4123,
              #   max_segment_len=total_data[params["config"]["label_type"]]["train"]["time-red-%s" % time_red]["95_percentile"]
              # )
              #
              # train_realignment = run_rasr_realignment(compile_config=compile_config.get_config(),
              #   alias_addon=alias_addon + "_train", segment_path=train_data_opts["segment_file"], name=name,
              #   corpus_path=corpus_files["train"], lexicon_path=bpe_sil_phon_lexicon_path,
              #   allophone_path=total_data["bpe-with-sil"]["allophones"], loop_update_history=True,
              #   blank_update_history=True,
              #   state_tying_path=total_data["bpe-with-sil"]["state_tying"], feature_cache_path=feature_cache_files["train"],
              #   num_epochs=epoch, label_file=total_data["bpe-with-sil"]["rasr_label_file"], label_pruning=50.0,
              #   label_pruning_limit=1000, label_recombination_limit=-1, blank_label_index=1031, model_checkpoint=checkpoint,
              #   context_size=-1, reduction_factors=time_red, rasr_nn_trainer_exe_path=rasr_nn_trainer,
              #   start_label_index=1030, rasr_am_trainer_exe_path=rasr_am_trainer, num_classes=1032, time_rqmt=80,
              #   blank_allophone_state_idx=4123,
              #   max_segment_len=total_data[params["config"]["label_type"]]["train"]["time-red-%s" % time_red][
              #     "95_percentile"])
              #



          # for global attention models with BPE labels use RETURNN decoding
          elif params["config"]["label_type"].startswith("bpe"):
            if name == "glob.best-model.bpe.time-red6.am2048.6pretrain-reps.ctx-use-bias.all-segs":
              beam_sizes = [12]
            else:
              beam_sizes = [12]
            for beam_size in beam_sizes:
              if name == "glob.best-model.bpe.time-red6.am2048.1pretrain-reps.no-weight-feedback.ctx-use-bias.pretrain-like-seg.all-segs":
                eval_corpus_names = ["dev", "hub5e_01", "rt03s"]
              else:
                eval_corpus_names = ["dev"]
              for eval_corpus_name in eval_corpus_names:
                if eval_corpus_name == "dev":
                  data_opts = copy.deepcopy(dev_data_opts)
                  stm_job = stm_jobs["hub5_00"]
                elif eval_corpus_name == "hub5e_01":
                  data_opts = copy.deepcopy(hub5e_01_data_opts)
                  stm_job = stm_jobs["hub5_01"]
                else:
                  assert eval_corpus_name == "rt03s"
                  data_opts = copy.deepcopy(rt03s_data_opts)
                  stm_job = stm_jobs["rt03s"]
                search_config = config_class(
                  task="search",
                  beam_size=beam_size,
                  search_data_opts=data_opts,
                  **config_params)
                ctm_results = run_bpe_returnn_decoding(
                  returnn_config=search_config.get_config(), checkpoint=checkpoint, bliss_corpus=stm_job.bliss_corpus,
                  num_epochs=epoch, name=name, dataset_key=eval_corpus_name, alias_addon="_beam-%s" % beam_size,
                  mem_rqmt=4 if beam_size == 12 else 16)
                run_eval(
                  ctm_file=ctm_results,
                  reference=Path("/work/asr3/zeyer/schmitt/tests/swb1/bpe-transducer_decoding-test/dev_non_empty.stm") if name == "glob.best-model.bpe.time-red6.am2048.1pretrain-reps.no-weight-feedback.ctx-use-bias.pretrain-like-seg.all-segs" and eval_corpus_name == "dev" else eval_ref_files[eval_corpus_name],
                  name=name,
                  dataset_key=eval_corpus_name, num_epochs=epoch, alias_addon="_beam-%s" % beam_size)

            search_error_data_opts = copy.deepcopy(cv_data_opts)
            label_hdf = search_error_data_opts.pop("label_hdf")
            label_name = search_error_data_opts.pop("label_name")
            segment_file = search_error_data_opts.pop("segment_file")
            search_error_data_opts["vocab"] = dev_data_opts["vocab"]
            dump_search_config = config_class(
              task="search", search_data_opts=search_error_data_opts, dump_output=True, import_model=checkpoint,
              **config_params)
            train_config_load = copy.deepcopy(train_config_obj)
            train_config_load.config["load"] = checkpoint
            search_targets_hdf, ctm_results = calculate_search_errors(
              checkpoint=checkpoint, search_config=dump_search_config, train_config=train_config_load,
              name=name, segment_path=segment_file, ref_targets=label_hdf, label_name=label_name, model_type="glob",
              blank_idx=0, rasr_nn_trainer_exe=rasr_nn_trainer, rasr_config=returnn_train_rasr_configs["cv"],
              alias_addon="_debug", epoch=epoch, dataset_key="cv", bliss_corpus=stm_jobs["cv"].bliss_corpus, length_norm=True)
            run_eval(ctm_file=ctm_results, reference=stm_jobs["cv"].out_stm_path, name=name,
              dataset_key="cv", num_epochs=epoch, alias_addon="_beam-%s" % beam_size)

            if name in [
              "glob.best-model.bpe.time-red6.am2048.1pretrain-reps.no-weight-feedback.ctx-use-bias.pretrain-like-seg.all-segs",
              "glob.best-model.bpe.concat.ep-split-12.time-red6.am2048.1pretrain-reps.no-weight-feedback.ctx-use-bias.pretrain-like-seg.all-segs"] and epoch == 150:
              for concat_num in concat_seq_tags_examples:
                search_error_data_opts.update({
                  "concat_seqs": True, "concat_seq_tags": concat_jobs[concat_num]["cv"].out_concat_seq_tags,
                  "concat_seq_lens": concat_jobs[concat_num]["cv"].out_orig_seq_lens_py})
                dump_search_concat_config = config_class(task="search", search_data_opts=search_error_data_opts, dump_output=True,
                  import_model=checkpoint, **config_params)
                search_targets_concat_hdf, ctm_results = calculate_search_errors(checkpoint=checkpoint,
                  search_config=dump_search_concat_config, train_config=train_config_load, name=name, segment_path=segment_file,
                  ref_targets=label_hdf, label_name=label_name, model_type="glob", blank_idx=0,
                  rasr_nn_trainer_exe=rasr_nn_trainer, rasr_config=returnn_train_rasr_configs["cv"], alias_addon="_cv_concat%d" % concat_num,
                  epoch=epoch, dataset_key="cv", bliss_corpus=stm_jobs["cv"].bliss_corpus, length_norm=True,
                  concat_seq_tags_file=concat_jobs[concat_num]["cv"].out_concat_seq_tags)
                # run_eval(ctm_file=ctm_results, reference=stm_jobs["cv"].out_stm_path, name=name, dataset_key="cv_concat%d" % concat_num,
                #          num_epochs=epoch, alias_addon="_beam-%s" % beam_size)

                feed_config_load = copy.deepcopy(train_config_obj)
                feed_config_load.config["load"] = checkpoint
                vocab_file = bpe_vocab["vocab_file"] if params["config"]["label_type"] == "bpe" else total_data["bpe-with-sil"]["json_vocab"]
                hdf_aliases = ["ground-truth-concat", "search-concat"]
                hdf_targetss = [label_hdf, search_targets_concat_hdf]
                for hdf_alias, hdf_targets in zip(hdf_aliases, hdf_targetss):
                  for seq_tag in concat_seq_tags_examples[concat_num]:
                    dump_att_weights_job = DumpAttentionWeightsJob(
                      returnn_config=feed_config_load, model_type="glob",
                      rasr_config=returnn_train_rasr_configs["cv"], blank_idx=0, label_name=label_name,
                      rasr_nn_trainer_exe=rasr_nn_trainer, hdf_targets=hdf_targets, concat_seqs=True,
                      seq_tag=seq_tag, concat_hdf=False if hdf_alias == "search-concat" else True,
                      concat_seq_tags_file=concat_jobs[concat_num]["cv"].out_concat_seq_tags)
                    seq_tags = seq_tag.split(";")
                    seq_tag_alias = seq_tags[0].replace("/", "_") + "-" + seq_tags[-1].replace("/", "_")
                    dump_att_weights_job.add_alias(name + "/" + seq_tag_alias + "/att_weights_%s-labels" % (hdf_alias,))
                    tk.register_output(dump_att_weights_job.get_one_alias(), dump_att_weights_job.out_data)

                    plot_weights_job = PlotAttentionWeightsJob(
                      data_path=dump_att_weights_job.out_data,
                      blank_idx=None, json_vocab_path=vocab_file,
                      time_red=6, seq_tag=seq_tag, mem_rqmt=12)
                    plot_weights_job.add_alias(name + "/" + seq_tag_alias + "/plot_att_weights_%s-labels" % (hdf_alias,))
                    tk.register_output(plot_weights_job.get_one_alias(), plot_weights_job.out_plot)


            if epoch == 150 and name == "glob.best-model.bpe.time-red6.am2048.1pretrain-reps.no-weight-feedback.ctx-use-bias.pretrain-like-seg.all-segs":
              feed_config_load = copy.deepcopy(train_config_obj)
              feed_config_load.config["load"] = checkpoint
              vocab_file = bpe_vocab["vocab_file"] if params["config"]["label_type"] == "bpe" else total_data["bpe-with-sil"]["json_vocab"]
              if name.startswith("glob.best-model.bpe."):
                hdf_aliases = ["ground-truth", "search"]
                hdf_targetss = [label_hdf, search_targets_hdf]
                # if name == "glob.best-model.bpe.time-red6.am2048.1pretrain-reps.no-weight-feedback.ctx-use-bias.pretrain-like-seg.all-segs":
                #   hdf_aliases += ["global_import_segmental"]
                #   hdf_targetss += [search_labels["global_import_segmental"]]
              else:
                assert name.startswith("glob.best-model.bpe-with-sil")
                hdf_aliases = ["ground-truth", "search", "global_import_segmental_w_split_sil"]
                hdf_targetss = [label_hdf, search_targets_hdf, search_labels["global_import_segmental_w_split_sil"]]

              seq_tags = [
                "switchboard-1/sw02102A/sw2102A-ms98-a-0092", "switchboard-1/sw02022A/sw2022A-ms98-a-0002",
                "switchboard-1/sw02102A/sw2102A-ms98-a-0090", "switchboard-1/sw02025A/sw2025A-ms98-a-0035",
                "switchboard-1/sw02102A/sw2102A-ms98-a-0002", "switchboard-1/sw02023A/sw2023A-ms98-a-0001",
                "switchboard-1/sw02252A/sw2252A-ms98-a-0094", "switchboard-1/sw02273B/sw2273B-ms98-a-0113"
              ]
              if name == "glob.best-model.bpe.time-red6.am2048.1pretrain-reps.no-weight-feedback.ctx-use-bias.pretrain-like-seg.all-segs":
                seq_tags += [
                  "switchboard-1/sw02121A/sw2121A-ms98-a-0096",  # both correct
                  "switchboard-1/sw02375B/sw2375B-ms98-a-0015",  # both correct
                  "switchboard-1/sw02248A/sw2248A-ms98-a-0076",  # both correct
                  "switchboard-1/sw02242B/sw2242B-ms98-a-0106",  # both correct
                  "switchboard-1/sw02252A/sw2252A-ms98-a-0088",  # both correct
                  "switchboard-1/sw02306B/sw2306B-ms98-a-0059",  # only segmental correct
                  "switchboard-1/sw02107A/sw2107A-ms98-a-0042",  # only segmental correct
                  "switchboard-1/sw02064A/sw2064A-ms98-a-0038",  # only segmental correct
                  "switchboard-1/sw02230B/sw2230B-ms98-a-0075",  # only segmental correct
                  "switchboard-1/sw02252A/sw2252A-ms98-a-0030",  # only segmental correct
                  "switchboard-1/sw02102A/sw2102A-ms98-a-0039",  # only global correct
                  "switchboard-1/sw02038B/sw2038B-ms98-a-0017",  # only global correct
                  "switchboard-1/sw02107A/sw2107A-ms98-a-0085",  # only global correct
                  "switchboard-1/sw02141A/sw2141A-ms98-a-0121",  # only global correct
                  "switchboard-1/sw02161B/sw2161B-ms98-a-0059",  # only global correct
                ]
              for hdf_alias, hdf_targets in zip(hdf_aliases, hdf_targetss):
                for seq_tag in seq_tags:
                  dump_att_weights_job = DumpAttentionWeightsJob(returnn_config=feed_config_load, model_type="glob",
                    rasr_config=returnn_train_rasr_configs["cv"], blank_idx=0, label_name=label_name,
                    rasr_nn_trainer_exe=rasr_nn_trainer, hdf_targets=hdf_targets,
                    seq_tag=seq_tag, )
                  dump_att_weights_job.add_alias(name + "/" + seq_tag.replace("/", "_") + "/att_weights_%s-labels" % (hdf_alias,))
                  tk.register_output(dump_att_weights_job.get_one_alias(), dump_att_weights_job.out_data)

                  plot_weights_job = PlotAttentionWeightsJob(
                    data_path=dump_att_weights_job.out_data,
                    blank_idx=None, json_vocab_path=vocab_file,
                    time_red=6, seq_tag=seq_tag)
                  plot_weights_job.add_alias(name + "/" + seq_tag.replace("/", "_") + "/plot_att_weights_%s-labels" % (hdf_alias,))
                  tk.register_output(plot_weights_job.get_one_alias(), plot_weights_job.out_plot)

              for hdf_alias, hdf_targets in zip(hdf_aliases, hdf_targetss):
                calc_search_err_job = CalcSearchErrorJob(returnn_config=train_config, rasr_config=returnn_train_rasr_configs["cv"],
                  rasr_nn_trainer_exe=rasr_nn_trainer, segment_file=segment_file, blank_idx=0,
                  model_type="glob", label_name=label_name, search_targets=hdf_targets, ref_targets=label_hdf,
                  max_seg_len=-1, length_norm=True)
                calc_search_err_job.add_alias(name + ("/search_errors_%d_%s" % (epoch, hdf_alias)))
                alias = calc_search_err_job.get_one_alias()
                tk.register_output(alias + "search_errors", calc_search_err_job.out_search_errors)

            if name == "glob.best-model.bpe.time-red6.am2048.1pretrain-reps.no-weight-feedback.ctx-use-bias.pretrain-like-seg.all-segs":
              compile_config = config_class(
                task="eval", feature_stddev=3.0, beam_size=beam_size, search_data_opts=dev_data_opts,
                **config_params)
              alias_addon = "rasr_limit12"
              new_rasr_decoding_opts = copy.deepcopy(rasr_decoding_opts)
              new_rasr_decoding_opts.update(
                dict(word_end_pruning_limit=12, word_end_pruning=12.0, label_pruning_limit=12, label_pruning=12.0))
              # ctm_results = run_rasr_decoding(
              #   segment_path=None, mem_rqmt=mem_rqmt, simple_beam_search=True,
              #   length_norm=True, full_sum_decoding=False, blank_update_history=True,
              #   allow_word_end_recombination=False, loop_update_history=True,
              #   allow_label_recombination=False, max_seg_len=None, debug=False,
              #   compile_config=compile_config.get_config(), alias_addon=alias_addon, rasr_exe_path=rasr_flf_tool,
              #   model_checkpoint=checkpoint, num_epochs=epoch, time_rqmt=20, gpu_rqmt=1, **new_rasr_decoding_opts)
              # run_eval(ctm_file=ctm_results, reference=Path("/u/tuske/bin/switchboard/hub5e_00.2.stm"), name=name,
              #          dataset_key="dev", num_epochs=epoch, alias_addon=alias_addon)

            if name in [
              "glob.best-model.bpe.time-red6.am2048.1pretrain-reps.no-weight-feedback.ctx-use-bias.pretrain-like-seg.all-segs",
              "glob.best-model.bpe.concat.ep-split-12.time-red6.am2048.1pretrain-reps.no-weight-feedback.ctx-use-bias.pretrain-like-seg.all-segs"] and epoch == 150:
              for concat_num in concat_jobs:
                for corpus_name in concat_jobs[concat_num]:
                  if corpus_name == "hub5e_00":
                    data_opts = copy.deepcopy(dev_data_opts)
                    stm_job = stm_jobs["hub5_00"]
                  # elif corpus_name == "cv":
                  #   data_opts = copy.deepcopy(cv_data_opts)
                  #   stm_job = stm_jobs["cv"]
                  elif corpus_name == "hub5e_01":
                    continue
                    data_opts = copy.deepcopy(hub5e_01_data_opts)
                    stm_job = stm_jobs["hub5_01"]
                  else:
                    continue
                    data_opts = copy.deepcopy(rt03s_data_opts)
                    stm_job = stm_jobs["rt03s"]
                  data_opts.update({
                    "concat_seqs": True, "concat_seq_tags": concat_jobs[concat_num][corpus_name].out_concat_seq_tags,
                    "concat_seq_lens": concat_jobs[concat_num][corpus_name].out_orig_seq_lens_py
                  })
                  search_config = config_class(
                    task="search", beam_size=12, search_data_opts=data_opts,
                    **config_params)

                  from recipe.i6_experiments.users.schmitt.experiments.config.concat_seqs.scoring import ScliteHubScoreJob
                  ctm_results = run_bpe_returnn_decoding(returnn_config=search_config.get_config(), checkpoint=checkpoint,
                    bliss_corpus=stm_job.bliss_corpus, num_epochs=epoch, name=name, dataset_key=corpus_name, concat_seqs=True,
                    alias_addon="concat-%s_beam-%s" % (concat_num, 12), mem_rqmt=4,
                    stm_path=concat_jobs[concat_num][corpus_name].out_stm)
                  run_eval(ctm_file=ctm_results, reference=concat_jobs[concat_num][corpus_name].out_stm, name=name,
                    dataset_key="%s_concat-%s" % (corpus_name, concat_num), num_epochs=epoch, alias_addon="_beam-%s" % 12)

                  search_error_data_opts = copy.deepcopy(cv_data_opts)
                  label_hdf = search_error_data_opts.pop("label_hdf")
                  label_name = search_error_data_opts.pop("label_name")
                  segment_file = search_error_data_opts.pop("segment_file")
                  search_error_data_opts["vocab"] = dev_data_opts["vocab"]
                  dump_search_config = config_class(
                    task="search", search_data_opts=search_error_data_opts, dump_output=True, import_model=checkpoint,
                    **config_params)
                  train_config_load = copy.deepcopy(train_config_obj)
                  train_config_load.config["load"] = checkpoint
                  search_targets_hdf, ctm_results = calculate_search_errors(
                    checkpoint=checkpoint, search_config=dump_search_config, train_config=train_config_load,
                    name=name, segment_path=segment_file, ref_targets=label_hdf, label_name=label_name,
                    model_type="glob",
                    blank_idx=0, rasr_nn_trainer_exe=rasr_nn_trainer, rasr_config=returnn_train_rasr_configs["cv"],
                    alias_addon="concat-%s_beam-%s_search-errors" % (concat_num, 12), epoch=epoch, dataset_key="cv", bliss_corpus=stm_jobs["cv"].bliss_corpus, length_norm=True)
                  run_eval(ctm_file=ctm_results, reference=stm_jobs["cv"].out_stm_path, name=name,
                           dataset_key="cv", num_epochs=epoch, alias_addon="_beam-%s" % beam_size)


  # for config_file, name in zip([
  #   "/u/schmitt/experiments/transducer/recipe/i6_experiments/users/schmitt/experiments/swb/transducer/returnn_config_files/clamped6.seg.mlp-att.am2048.key1024.query-lm.slow-rnn-in_input-embed+prev-att.fast-rnn-in_am+prev-out-embed.readout-in_lm+att.emit-prob-in_s_no-switchout.no-focal-loss.config",
  #   "/u/schmitt/experiments/transducer/recipe/i6_experiments/users/schmitt/experiments/swb/transducer/returnn_config_files/seg.mlp-att.am2048.key1024.query-lm.slow-rnn-in_input-embed+prev-att.fast-rnn-in_am+prev-out-embed.readout-in_lm+att.emit-prob-in_s_no-switchout.no-focal-loss.config",
  #   "/u/schmitt/experiments/transducer/recipe/i6_experiments/users/schmitt/experiments/swb/transducer/returnn_config_files/seg.mlp-att.am2048.key1024.query-lm.slow-rnn-in_input-embed.fast-rnn-in_am+prev-out-embed.readout-in_lm+att.emit-prob-in_s_no-switchout.no-focal-loss.config",
  #   "/u/schmitt/experiments/transducer/recipe/i6_experiments/users/schmitt/experiments/swb/transducer/returnn_config_files/clamped6.seg.mlp-att.am2048.key1024.query-lm.slow-rnn-in_input-embed.fast-rnn-in_am+prev-out-embed.readout-in_lm+att.emit-prob-in_s_no-switchout.no-focal-loss.config",
  #   "/u/schmitt/experiments/transducer/recipe/i6_experiments/users/schmitt/experiments/swb/transducer/returnn_config_files/seg.mlp-att.am2048.key1024.query-lm.slow-rnn-in_input-embed.fast-rnn-in_am+prev-out-embed.readout-in_lm+att.emit-prob-in_s_no-switchout.no-focal-loss_sep-loops.config",
  #   "/u/schmitt/experiments/transducer/recipe/i6_experiments/users/schmitt/experiments/swb/transducer/returnn_config_files/seg.mlp-att.am2048.key1024.query-lm.slow-rnn-in_input-embed+prev-att.fast-rnn-in_am+prev-out-embed.readout-in_lm+att.emit-prob-in_s_no-switchout.no-focal-loss_sep-loops.config"
  # ], [
  #   "clamped6.seg.mlp-att.am2048.key1024.query-lm.slow-rnn-in_input-embed+prev-att.fast-rnn-in_am+prev-out-embed.readout-in_lm+att.emit-prob-in_s_no-switchout.no-focal-loss",
  #   "seg.mlp-att.am2048.key1024.query-lm.slow-rnn-in_input-embed+prev-att.fast-rnn-in_am+prev-out-embed.readout-in_lm+att.emit-prob-in_s_no-switchout.no-focal-loss",
  #   "seg.mlp-att.am2048.key1024.query-lm.slow-rnn-in_input-embed.fast-rnn-in_am+prev-out-embed.readout-in_lm+att.emit-prob-in_s_no-switchout.no-focal-loss",
  #   "clamped6.seg.mlp-att.am2048.key1024.query-lm.slow-rnn-in_input-embed.fast-rnn-in_am+prev-out-embed.readout-in_lm+att.emit-prob-in_s_no-switchout.no-focal-loss",
  #   "seg.mlp-att.am2048.key1024.query-lm.slow-rnn-in_input-embed.fast-rnn-in_am+prev-out-embed.readout-in_lm+att.emit-prob-in_s_no-switchout.no-focal-loss_sep-loops",
  #   "seg.mlp-att.am2048.key1024.query-lm.slow-rnn-in_input-embed+prev-att.fast-rnn-in_am+prev-out-embed.readout-in_lm+att.emit-prob-in_s_no-switchout.no-focal-loss_sep-loops"
  # ]):
  #
  #   config_path = Path(config_file)
  #
  #   model_dir = run_training_from_file(
  #     config_file_path=config_path, mem_rqmt=24, time_rqmt=30, parameter_dict={},
  #     name="old_" + name, alias_suffix="_train"
  #   )

    # model_dir = Path("/u/schmitt/experiments/transducer/alias/glob.best-model.bpe.time-red6.am2048.6pretrain-reps.ctx-use-bias.all-segs/train/output/models")

    # for epoch in [33, 150]:
    #   ctm_results = run_search_from_file(
    #     config_file_path=config_path, parameter_dict={}, time_rqmt=1, mem_rqmt=4, name="old_" + name,
    #     alias_suffix="search", model_dir=model_dir, load_epoch=epoch, default_model_name="epoch",
    #     stm_job=hub5e_00_stm_job
    #   )
    #   run_eval(ctm_file=ctm_results, reference=Path("/u/tuske/bin/switchboard/hub5e_00.2.stm"), name="old_" + name,
    #     dataset_key="dev", num_epochs=epoch, alias_addon="returnn")




