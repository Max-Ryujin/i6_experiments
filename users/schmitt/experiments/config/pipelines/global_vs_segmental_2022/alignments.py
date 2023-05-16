from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022.miscellaneous import find_seqs_to_skip, update_seq_list_file, dump_phoneme_align, calc_align_stats, \
  augment_bpe_align_with_sil, alignment_split_silence, reduce_alignment, convert_phon_json_vocab_to_rasr_vocab, \
  convert_phon_json_vocab_to_allophones, convert_phon_json_vocab_to_state_tying, \
  convert_phon_json_vocab_to_rasr_formats, convert_bpe_json_vocab_to_rasr_formats, alignment_center_seg_boundaries, \
  alignment_add_eos

from i6_private.users.schmitt.returnn.tools import DumpForwardJob, CompileTFGraphJob, RASRDecodingJob, \
  CombineAttentionPlotsJob, DumpPhonemeAlignJob, AugmentBPEAlignmentJob, FindSegmentsToSkipJob, ModifySeqFileJob, \
  ConvertCTMBPEToWordsJob, RASRLatticeToCTMJob, CompareAlignmentsJob, DumpAttentionWeightsJob, PlotAttentionWeightsJob, \
  DumpNonBlanksFromAlignmentJob, CalcSearchErrorJob, RemoveLabelFromAlignmentJob, WordsToCTMJob

from sisyphus import *


def create_swb_alignments(
  data_dict, seq_filter_files_standard, bpe_standard_aligns, bpe_vocab, rasr_nn_trainer, phon_extraction_rasr_configs):
  for corpus_key in [
    "cv",
    "train"
  ]:
    if corpus_key == "cv":
      time_rqmt = 2
      mem_rqmt = 4
    else:
      time_rqmt = 10
      mem_rqmt = 6

    bpe_seq_filter_file = seq_filter_files_standard[corpus_key]
    # ----------------------- BPE ALIGNMENTS -----------------------------------------
    bpe_label_dep_mean_lens, bpe_mean_non_sil_len, bpe_95_percentile = calc_align_stats(
      alignment=bpe_standard_aligns[corpus_key],
      seq_filter_file=bpe_seq_filter_file, alias="bpe_align_stats/stats_" + corpus_key,
      blank_idx=1030)

    bpe_labels_job = DumpNonBlanksFromAlignmentJob(
      alignment=bpe_standard_aligns[corpus_key], blank_idx=1030, time_rqmt=time_rqmt)
    bpe_labels_job.add_alias("bpe_labels/%s" % corpus_key)
    tk.register_output("bpe_labels/%s" % corpus_key, bpe_labels_job.out_labels)

    seq_filter_files_bpe = update_seq_list_file(seq_list_file=bpe_seq_filter_file,
      seqs_to_skip=bpe_labels_job.out_skipped_seqs_var,
      alias="seq_filter_files_bpe/time-red-%s/%s" % (6, corpus_key))

    bpe_state_tying, bpe_allophones, bpe_rasr_label_file = convert_bpe_json_vocab_to_rasr_formats(
      bpe_vocab["vocab_file"], blank_idx=1030, alias="bpe_rasr_formats")
    data_dict["bpe"].update({
      "json_vocab": bpe_vocab["vocab_file"],
      "state_tying": bpe_state_tying, "allophones": bpe_allophones, "rasr_label_file": bpe_rasr_label_file})
    data_dict["bpe"][corpus_key] = {
      "label_seqs": bpe_labels_job.out_labels,
      "time-red-6": {
        "align": bpe_standard_aligns[corpus_key],
        "seq_filter_file": seq_filter_files_bpe,
        "label_dep_mean_lens": bpe_label_dep_mean_lens, "mean_non_sil_len": bpe_mean_non_sil_len,
        "95_percentile": bpe_95_percentile}
    }
    if corpus_key == "train":
      data_dict["bpe"]["devtrain"] = {
        "time-red-6": {
          "seq_filter_file": seq_filter_files_standard["devtrain"]}}

    # ----------------------- BPE ALIGNMENTS WITH EOS -----------------------------------------
    bpe_eos_align, bpe_eos_seqs, bpe_eos_exclude_seqs = alignment_add_eos(
      blank_idx=1030, eos_idx=0,
      alias="bpe_eos_align/align_%s" % corpus_key,
      alignment=bpe_standard_aligns[corpus_key],
      seq_filter_file=bpe_seq_filter_file)
    bpe_eos_label_dep_mean_lens, bpe_eos_mean_non_sil_len, bpe_eos_95_percentile = calc_align_stats(
      alignment=bpe_eos_align, blank_idx=1030, sil_idx=1031,
      seq_filter_file=bpe_eos_seqs,
      alias="bpe_eos_align_stats/time-red-6/stats_" + corpus_key)

    bpe_eos_labels_job = DumpNonBlanksFromAlignmentJob(alignment=bpe_eos_align, blank_idx=1030,
                                                              time_rqmt=time_rqmt)
    bpe_eos_labels_job.add_alias("bpe_eos_labels/%s" % corpus_key)
    tk.register_output(bpe_eos_labels_job.get_one_alias(), bpe_eos_labels_job.out_labels)

    data_dict["bpe-eos"].update({
      "json_vocab": bpe_vocab, "state_tying": bpe_state_tying, "allophones": bpe_allophones,
      "rasr_label_file": bpe_rasr_label_file})
    data_dict["bpe-eos"][corpus_key] = {
      "label_seqs": None,
      "time-red-6": {
        "align": bpe_eos_align,
        "seq_filter_file": bpe_eos_seqs,
        "label_dep_mean_lens": bpe_eos_label_dep_mean_lens,
        "mean_non_sil_len": bpe_eos_mean_non_sil_len,
        "95_percentile": bpe_eos_95_percentile}}
    if corpus_key == "train":
      seq_filter_files_bpe_eos_devtrain = update_seq_list_file(
        seq_list_file=seq_filter_files_standard["devtrain"],
        seqs_to_skip=bpe_eos_exclude_seqs,
        alias="seq_filter_files_bpe_eos/time-red-%s/%s" % (6, "devtrain"))
      data_dict["bpe-eos"]["devtrain"] = {
        "time-red-6": {
          "seq_filter_file": seq_filter_files_bpe_eos_devtrain}}

    # ----------------------- BPE Center Position ALIGNMENTS -----------------------------------------
    bpe_center_pos_align = alignment_center_seg_boundaries(
      blank_idx=1030,
      alias="bpe_center_pos_align/align_%s" % corpus_key, alignment=bpe_standard_aligns[corpus_key],
      seq_filter_file=bpe_seq_filter_file)
    bpe_center_pos_label_dep_mean_lens, bpe_center_pos_mean_non_sil_len, bpe_center_pos_95_percentile = calc_align_stats(
      alignment=bpe_center_pos_align, blank_idx=1030,
      seq_filter_file=bpe_seq_filter_file,
      alias="bpe_center_pos_align_stats/time-red-6/stats_" + corpus_key)

    bpe_center_pos_labels_job = DumpNonBlanksFromAlignmentJob(alignment=bpe_center_pos_align, blank_idx=1030,
                                                                 time_rqmt=time_rqmt)
    bpe_center_pos_labels_job.add_alias("bpe_center_pos_labels/%s" % corpus_key)
    tk.register_output(bpe_center_pos_labels_job.get_one_alias(), bpe_center_pos_labels_job.out_labels)

    data_dict["bpe-center-positions"].update({
      "json_vocab": bpe_vocab["vocab_file"],
      "state_tying": bpe_state_tying, "allophones": bpe_allophones, "rasr_label_file": bpe_rasr_label_file})
    data_dict["bpe-center-positions"][corpus_key] = {
      "label_seqs": bpe_center_pos_labels_job.out_labels,
      "time-red-6": {
        "align": bpe_center_pos_align, "seq_filter_file": bpe_seq_filter_file,
        "label_dep_mean_lens": bpe_center_pos_label_dep_mean_lens,
        "mean_non_sil_len": bpe_center_pos_mean_non_sil_len,
        "95_percentile": bpe_center_pos_95_percentile}}
    if corpus_key == "train":
      data_dict["bpe-center-positions"]["devtrain"] = {
        "time-red-6": {
          "seq_filter_file": seq_filter_files_standard["devtrain"]}}

    # ----------------------- PHONEME ALIGNMENTS -----------------------------------------
    # extract phoneme alignments
    phoneme_align, phoneme_vocab_path = dump_phoneme_align(
      time_rqmt=time_rqmt, rasr_exe=rasr_nn_trainer, rasr_config=phon_extraction_rasr_configs[corpus_key],
      mem_rqmt=mem_rqmt, time_red=1, state_tying_file=Path("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/tuske-phoneme-align/state-tying_mono-eow_3-states"),
      alias="phon_align/%s/%s" % ("time-red-1", corpus_key))

    phon_state_tying, phon_allophones, phon_rasr_label_file = convert_phon_json_vocab_to_rasr_formats(
      phoneme_vocab_path, blank_idx=89)

    # calculate alignment statistics for phoneme alignment without time reduction
    phoneme_label_dep_mean_lens, phoneme_mean_non_sil_len, phoneme_95_percentile = calc_align_stats(
      alignment=phoneme_align,
      seq_filter_file=bpe_seq_filter_file, alias="phon_align_stats/stats_" + corpus_key)

    phoneme_labels_job = DumpNonBlanksFromAlignmentJob(
      alignment=phoneme_align, blank_idx=89, time_rqmt=time_rqmt
    )
    phoneme_labels_job.add_alias("phoneme_labels/%s" % corpus_key)
    tk.register_output("phoneme_labels/%s" % corpus_key, phoneme_labels_job.out_labels)


    data_dict["phonemes"].update({
      "json_vocab": phoneme_vocab_path,
      "state_tying": phon_state_tying, "allophones": phon_allophones, "rasr_label_file": phon_rasr_label_file})
    data_dict["phonemes"][corpus_key] = {
      "label_seqs": phoneme_labels_job.out_labels,
      "time-red-1": {
        "align": phoneme_align,
        "seq_filter_file": bpe_seq_filter_file, "label_dep_mean_lens": phoneme_label_dep_mean_lens,
        "mean_non_sil_len": phoneme_mean_non_sil_len, "95_percentile": phoneme_95_percentile}}
    if corpus_key == "train":
      data_dict["phonemes"]["devtrain"] = {
        "time-red-1": {
          "seq_filter_file": seq_filter_files_standard["devtrain"]}}

    # ----------------------- PHONEME SPLIT SILENCE ALIGNMENTS -----------------------------------------

    phoneme_split_sil_align = alignment_split_silence(
      sil_idx=0, blank_idx=89, alias="phon_split_sil_align/align_%s" % corpus_key, alignment=phoneme_align,
      seq_filter_file=bpe_seq_filter_file, max_len=phoneme_mean_non_sil_len)
    phoneme_split_sil_label_dep_mean_lens, phoneme_split_sil_mean_non_sil_len, phoneme_split_sil_95_percentile = calc_align_stats(
      alignment=phoneme_split_sil_align, seq_filter_file=bpe_seq_filter_file,
      alias="phon_split_sil_align_stats/stats_" + corpus_key)

    phoneme_split_sil_labels_job = DumpNonBlanksFromAlignmentJob(alignment=phoneme_split_sil_align, blank_idx=89, time_rqmt=time_rqmt)
    phoneme_split_sil_labels_job.add_alias("phoneme-split-sil_labels/%s" % corpus_key)
    tk.register_output(phoneme_split_sil_labels_job.get_one_alias(), phoneme_split_sil_labels_job.out_labels)

    data_dict["phonemes-split-sil"][corpus_key] = {
      "label_seqs": phoneme_split_sil_labels_job.out_labels,
      "time-red-1": {
        "align": phoneme_split_sil_align, "seq_filter_file": bpe_seq_filter_file,
        "label_dep_mean_lens": phoneme_split_sil_label_dep_mean_lens,
        "mean_non_sil_len": phoneme_split_sil_mean_non_sil_len,
        "95_percentile": phoneme_split_sil_95_percentile}}
    if corpus_key == "train":
      data_dict["phonemes-split-sil"]["devtrain"] = {
        "time-red-1": {
          "seq_filter_file": seq_filter_files_standard["devtrain"]}}

    # ----------------------- BPE + SILENCE ALIGNMENTS -----------------------------------------

    bpe_sil_align, bpe_sil_skipped_seqs, bpe_sil_vocab_path = augment_bpe_align_with_sil(
      phon_align=phoneme_align,
      bpe_align=bpe_standard_aligns[corpus_key],
      seq_filter_file=bpe_seq_filter_file,
      phon_vocab=phoneme_vocab_path,
      alias="bpe_sil_align/%s/%s" % ("time-red-1", corpus_key), phon_time_red=1,
      time_rqmt=2 if corpus_key == "dev" else 6, mem_rqmt=mem_rqmt)

    bpe_sil_vocab = {
      "bpe_file": bpe_vocab["bpe_file"],
      "vocab_file": bpe_sil_vocab_path
    }

    seq_filter_files_bpe_sil = update_seq_list_file(
      seq_list_file=bpe_seq_filter_file, seqs_to_skip=bpe_sil_skipped_seqs,
      alias="seq_filter_files_bpe_sil/time-red-%s/%s" % (1, corpus_key))

    # calculate alignment statistics for bpe + silence alignment with time red factor 1
    bpe_sil_label_dep_mean_lens, bpe_sil_mean_non_sil_len, bpe_sil_95_percentile = calc_align_stats(
      alignment=bpe_sil_align,
      seq_filter_file=seq_filter_files_bpe_sil,
      alias="bpe-with-sil_align_stats/stats_" + corpus_key, blank_idx=1031)

    bpe_sil_labels_job = DumpNonBlanksFromAlignmentJob(alignment=bpe_sil_align, blank_idx=1031, time_rqmt=time_rqmt)
    bpe_sil_labels_job.add_alias("bpe-sil_labels/%s" % corpus_key)
    tk.register_output("bpe-sil_labels/%s" % corpus_key, bpe_sil_labels_job.out_labels)

    bpe_sil_state_tying, bpe_sil_allophones, bpe_sil_rasr_label_file = convert_bpe_json_vocab_to_rasr_formats(
      bpe_sil_vocab_path, blank_idx=1031, alias="bpe_sil_rasr_formats")

    data_dict["bpe-with-sil"].update({
      "json_vocab": bpe_sil_vocab_path,
      "state_tying": bpe_sil_state_tying, "allophones": bpe_sil_allophones, "rasr_label_file": bpe_sil_rasr_label_file})
    data_dict["bpe-with-sil"][corpus_key] = {
      "label_seqs": bpe_sil_labels_job.out_labels,
      "time-red-1": {
        "align": bpe_sil_align, "seq_filter_file": seq_filter_files_bpe_sil,
        "label_dep_mean_lens": bpe_sil_label_dep_mean_lens,
        "mean_non_sil_len": bpe_sil_mean_non_sil_len, "95_percentile": bpe_sil_95_percentile}}
    if corpus_key == "train":
      seq_filter_files_bpe_sil_devtrain = update_seq_list_file(
        seq_list_file=seq_filter_files_standard["devtrain"],
        seqs_to_skip=bpe_sil_skipped_seqs, alias="seq_filter_files_bpe_sil/time-red-%s/%s" % (1, "devtrain"))
      data_dict["bpe-with-sil"]["devtrain"] = {
        "time-red-1": {
          "seq_filter_file": seq_filter_files_bpe_sil_devtrain}}

    # ----------------------- BPE + SILENCE SPLIT SILENCE ALIGNMENTS -----------------------------------------

    bpe_sil_split_sil_align = alignment_split_silence(
      sil_idx=0, blank_idx=1031,
      alias="bpe_sil_split_sil_align/align_%s" % corpus_key, alignment=bpe_sil_align,
      seq_filter_file=seq_filter_files_bpe_sil, max_len=bpe_sil_mean_non_sil_len)
    bpe_sil_split_sil_label_dep_mean_lens, bpe_sil_split_sil_mean_non_sil_len, bpe_sil_split_sil_95_percentile = calc_align_stats(
      alignment=bpe_sil_split_sil_align, blank_idx=1031,
      seq_filter_file=seq_filter_files_bpe_sil,
      alias="bpe-with-sil-split-sil_align_stats/time-red-1/stats_" + corpus_key)

    bpe_sil_split_sil_labels_job = DumpNonBlanksFromAlignmentJob(alignment=bpe_sil_split_sil_align, blank_idx=1031, time_rqmt=time_rqmt)
    bpe_sil_split_sil_labels_job.add_alias("bpe-sil-split-sil_labels/%s" % corpus_key)
    tk.register_output(bpe_sil_split_sil_labels_job.get_one_alias(), bpe_sil_split_sil_labels_job.out_labels)

    data_dict["bpe-with-sil-split-sil"].update({
      "json_vocab": bpe_sil_vocab_path,
      "state_tying": bpe_sil_state_tying, "allophones": bpe_sil_allophones, "rasr_label_file": bpe_sil_rasr_label_file})
    data_dict["bpe-with-sil-split-sil"][corpus_key] = {
      "label_seqs": bpe_sil_split_sil_labels_job.out_labels,
      "time-red-1": {
        "align": bpe_sil_split_sil_align, "seq_filter_file": seq_filter_files_bpe_sil,
        "label_dep_mean_lens": bpe_sil_split_sil_label_dep_mean_lens,
        "mean_non_sil_len": bpe_sil_split_sil_mean_non_sil_len,
        "95_percentile": bpe_sil_split_sil_95_percentile}}
    if corpus_key == "train":
      seq_filter_files_bpe_sil_devtrain = update_seq_list_file(
        seq_list_file=seq_filter_files_standard["devtrain"],
        seqs_to_skip=bpe_sil_skipped_seqs, alias="seq_filter_files_bpe_sil/time-red-%s/%s" % (1, "devtrain"))
      data_dict["bpe-with-sil-split-sil"]["devtrain"] = {
        "time-red-1": {
          "seq_filter_file": seq_filter_files_bpe_sil_devtrain}}

  # ----------------------- BPE + SILENCE SPLIT SILENCE V2 label dependent means --------------------------------------

  bpe_sil_split_silv2_label_dep_mean_lens, bpe_sil_split_silv2_mean_non_sil_len, bpe_sil_split_silv2_95_percentile = calc_align_stats(
    alignment=Path("/work/asr3/zeyer/schmitt/old_models_and_analysis/old_bpe_sil_split_sil_aligns/train/AlignmentSplitSilenceJob.4dfiua41gqWb/output/out_align"),
    blank_idx=1031, seq_filter_file=seq_filter_files_bpe_sil,
    alias="bpe-with-sil-split-silv2_align_stats/time-red-6/stats_train")

  data_dict["bpe-with-sil-split-silv2"].update({
    "json_vocab": bpe_sil_vocab_path, "state_tying": bpe_sil_state_tying, "allophones": bpe_sil_allophones,
    "rasr_label_file": bpe_sil_rasr_label_file})
  data_dict["bpe-with-sil-split-silv2"]["train"] = {
    "time-red-6": {
      "seq_filter_file": seq_filter_files_bpe_sil,
      "label_dep_mean_lens": bpe_sil_split_silv2_label_dep_mean_lens,
      "mean_non_sil_len": bpe_sil_split_silv2_mean_non_sil_len, "95_percentile": bpe_sil_split_silv2_95_percentile}}

  for time_red in [2, 3, 6]:
    for label_type in ["bpe-with-sil", "phonemes"]:
      for corpus_key in ["train", "cv"]:
        # get reduce alignment
        align, red_skipped_seqs = reduce_alignment(
          alignment=data_dict[label_type][corpus_key]["time-red-1"]["align"],
          sil_idx=0,
          blank_idx=1031 if label_type == "bpe-with-sil" else 89,
          alias="%s_align/%s/%s" % (label_type, "time-red-%d" % time_red, corpus_key),
          seq_filter_file=data_dict[label_type][corpus_key]["time-red-1"]["seq_filter_file"],
          reduction_factor=time_red)
        # get seq filter file for reduced alignment
        seq_filter_file = update_seq_list_file(
          seq_list_file=data_dict[label_type][corpus_key]["time-red-1"]["seq_filter_file"],
          seqs_to_skip=red_skipped_seqs,
          alias="seq_filter_files_%s/time-red-%s/%s" % (label_type, time_red, corpus_key))
        # get label dependent means and mean non sil len for reduced alignment
        label_dep_mean_lens, mean_non_sil_len, percentile_95 = calc_align_stats(
          alignment=align,
          seq_filter_file=seq_filter_file,
          alias="%s-align_stats/time-red-%d/stats_%s" % (label_type, time_red, corpus_key),
          blank_idx=1031 if label_type == "bpe-with-sil" else 89)

        data_dict[label_type][corpus_key].update({
          "time-red-%s" % time_red: {
            "align": align, "seq_filter_file": seq_filter_file,
            "label_dep_mean_lens": label_dep_mean_lens,
            "mean_non_sil_len": mean_non_sil_len, "95_percentile": percentile_95}})
        if corpus_key == "train":
          seq_filter_file_devtrain = update_seq_list_file(
            seq_list_file=data_dict[label_type]["devtrain"]["time-red-1"]["seq_filter_file"],
            seqs_to_skip=red_skipped_seqs,
            alias="seq_filter_files_%s/time-red-%s/%s" % (label_type, time_red, "devtrain"))
          data_dict[label_type]["devtrain"].update({
            "time-red-%s" % time_red: {
              "seq_filter_file": seq_filter_file_devtrain}})

        # get reduced alignment with split silence
        split_sil_align = alignment_split_silence(
          sil_idx=0,
          blank_idx=1031 if label_type == "bpe-with-sil" else 89,
          alias="%s-split-sil_align/time-red-%d/align_%s" % (label_type, time_red, corpus_key),
          alignment=align,
          seq_filter_file=seq_filter_file,
          max_len=mean_non_sil_len)
        # get label dep means and mean non sil len for reduced split sil alignment
        split_sil_label_dep_mean_lens, split_sil_mean_non_sil_len, split_sil_percentile_95 = calc_align_stats(
          alignment=split_sil_align,
          blank_idx=1031 if label_type == "bpe-with-sil" else 89,
          seq_filter_file=seq_filter_file,
          alias="%s-split-sil_align_stats/time-red-%d/stats_%s" % (label_type, time_red, corpus_key))

        data_dict[label_type + "-split-sil"][corpus_key].update({
          "time-red-%s" % time_red: {
            "align": split_sil_align, "seq_filter_file": seq_filter_file,
            "label_dep_mean_lens": split_sil_label_dep_mean_lens,
            "mean_non_sil_len": split_sil_mean_non_sil_len,
            "95_percentile": split_sil_percentile_95}})
        if corpus_key == "train":
          seq_filter_file_devtrain = update_seq_list_file(
            seq_list_file=data_dict[label_type]["devtrain"]["time-red-1"]["seq_filter_file"],
            seqs_to_skip=red_skipped_seqs,
            alias="seq_filter_files_%s/time-red-%s/%s" % (label_type, time_red, "devtrain"))
          data_dict[label_type + "-split-sil"]["devtrain"].update({
            "time-red-%s" % time_red: {
              "seq_filter_file": seq_filter_file_devtrain}})

  for remove_only_middle in [False]:
    for corpus_key in ["train", "cv"]:
      # ----------------------- BPE + SILENCE WITH CENTERED SEGMENT BOUNDARIES -----------------------------------------
      bpe_center_pos_align = alignment_center_seg_boundaries(
        blank_idx=1031,
        alias="bpe_sil_center_pos_align/align_%s" % corpus_key,
        alignment=data_dict["bpe-with-sil"][corpus_key]["time-red-6"]["align"],
        seq_filter_file=data_dict["bpe-with-sil"][corpus_key]["time-red-6"]["seq_filter_file"])
      bpe_center_pos_label_dep_mean_lens, bpe_center_pos_mean_non_sil_len, bpe_center_pos_95_percentile = calc_align_stats(
        alignment=bpe_center_pos_align, blank_idx=1031,
        seq_filter_file=data_dict["bpe-with-sil"][corpus_key]["time-red-6"]["seq_filter_file"],
        alias="bpe_sil_center_pos_align_stats/time-red-6/stats_" + corpus_key)

      bpe_center_pos_labels_job = DumpNonBlanksFromAlignmentJob(alignment=bpe_center_pos_align, blank_idx=1031,
                                                                time_rqmt=time_rqmt)
      bpe_center_pos_labels_job.add_alias("bpe_sil_center_pos_labels/%s" % corpus_key)
      tk.register_output(bpe_center_pos_labels_job.get_one_alias(), bpe_center_pos_labels_job.out_labels)

      data_dict["bpe-sil-center-positions"].update({
        "json_vocab": bpe_sil_vocab_path, "state_tying": bpe_sil_state_tying, "allophones": bpe_sil_allophones,
        "rasr_label_file": bpe_sil_rasr_label_file})
      data_dict["bpe-sil-center-positions"][corpus_key] = {
        "label_seqs": None,
        "time-red-6": {
          "align": bpe_center_pos_align,
          "seq_filter_file": data_dict["bpe-with-sil"][corpus_key]["time-red-6"]["seq_filter_file"],
          "label_dep_mean_lens": bpe_center_pos_label_dep_mean_lens,
          "mean_non_sil_len": bpe_center_pos_mean_non_sil_len,
          "95_percentile": bpe_center_pos_95_percentile}}
      if corpus_key == "train":
        seq_filter_files_bpe_sil_devtrain = update_seq_list_file(
          seq_list_file=seq_filter_files_standard["devtrain"],
          seqs_to_skip=bpe_sil_skipped_seqs,
          alias="seq_filter_files_bpe_sil/time-red-%s/%s" % (6, "devtrain"))
        data_dict["bpe-sil-center-positions"]["devtrain"] = {
          "time-red-6": {
            "seq_filter_file": seq_filter_files_bpe_sil_devtrain}}

      # ----------------------- BPE + SILENCE WITH CENTERED SEGMENT BOUNDARIES AND SILENCE REMOVED AFTERWARD -----------

      bpe_sil_wo_sil_align_job = RemoveLabelFromAlignmentJob(
        alignment=bpe_center_pos_align, blank_idx=1031,
        remove_idx=0, remove_only_middle=remove_only_middle)
      bpe_sil_wo_sil_align_job.add_alias(
        "bpe-sil-center-pos-wo-sil%s/time-red-6/%s" % ("-in-middle" if remove_only_middle else "", corpus_key))
      tk.register_output(bpe_sil_wo_sil_align_job.get_one_alias(), bpe_sil_wo_sil_align_job.out_alignment)

      bpe_sil_wo_sil_label_dep_mean_lens, bpe_sil_wo_sil_mean_non_sil_len, bpe_sil_wo_sil_95_percentile = calc_align_stats(
        alignment=bpe_sil_wo_sil_align_job.out_alignment, blank_idx=1031,
        seq_filter_file=data_dict["bpe-with-sil"][corpus_key]["time-red-6"]["seq_filter_file"],
        alias="bpe-sil-center-pos-wo-sil%s_align_stats/time-red-6/stats_%s" % (
        "-in-middle" if remove_only_middle else "", corpus_key))

      data_dict["bpe-sil-center-pos-wo-sil%s" % ("-in-middle" if remove_only_middle else "")].update({
        "json_vocab": bpe_sil_vocab_path, "state_tying": bpe_sil_state_tying, "allophones": bpe_sil_allophones,
        "rasr_label_file": bpe_sil_rasr_label_file})
      data_dict["bpe-sil-center-pos-wo-sil%s" % ("-in-middle" if remove_only_middle else "")][corpus_key] = {
        "label_seqs": None,
        "time-red-6": {
          "align": bpe_sil_wo_sil_align_job.out_alignment,
          "seq_filter_file": data_dict["bpe-with-sil"][corpus_key]["time-red-6"]["seq_filter_file"],
          "label_dep_mean_lens": bpe_sil_wo_sil_label_dep_mean_lens,
          "mean_non_sil_len": bpe_sil_wo_sil_mean_non_sil_len, "95_percentile": bpe_sil_wo_sil_95_percentile}}
      if corpus_key == "train":
        seq_filter_files_bpe_sil_devtrain = update_seq_list_file(seq_list_file=seq_filter_files_standard["devtrain"],
                                                                 seqs_to_skip=bpe_sil_skipped_seqs,
                                                                 alias="seq_filter_files_bpe_sil/time-red-%s/%s" % (
                                                                 6, "devtrain"))
        data_dict["bpe-sil-center-pos-wo-sil%s" % ("-in-middle" if remove_only_middle else "")]["devtrain"] = {
          "time-red-6": {
            "seq_filter_file": seq_filter_files_bpe_sil_devtrain}}

  # ----------------------- BPE COMBINED RNA/HMM WITH SILENCE REMOVED -----------------------------------------

  for remove_only_middle in [False]:
    for corpus_key in ["train", "cv"]:
      bpe_sil_wo_sil_align_job = RemoveLabelFromAlignmentJob(
        alignment=data_dict["bpe-with-sil"][corpus_key]["time-red-6"]["align"], blank_idx=1031,
        remove_idx=0, remove_only_middle=remove_only_middle)
      bpe_sil_wo_sil_align_job.add_alias("bpe-sil-wo-sil%s/time-red-6/%s" % ("-in-middle" if remove_only_middle else "", corpus_key))
      tk.register_output(bpe_sil_wo_sil_align_job.get_one_alias(), bpe_sil_wo_sil_align_job.out_alignment)

      bpe_sil_wo_sil_label_dep_mean_lens, bpe_sil_wo_sil_mean_non_sil_len, bpe_sil_wo_sil_95_percentile = calc_align_stats(
        alignment=bpe_sil_wo_sil_align_job.out_alignment, blank_idx=1031,
        seq_filter_file=data_dict["bpe-with-sil"][corpus_key]["time-red-6"]["seq_filter_file"],
        alias="bpe-sil-wo-sil%s_align_stats/time-red-6/stats_%s" % ("-in-middle" if remove_only_middle else "", corpus_key))

      data_dict["bpe-sil-wo-sil%s" % ("-in-middle" if remove_only_middle else "")].update({
        "json_vocab": bpe_sil_vocab_path, "state_tying": bpe_sil_state_tying, "allophones": bpe_sil_allophones,
        "rasr_label_file": bpe_sil_rasr_label_file})
      data_dict["bpe-sil-wo-sil%s" % ("-in-middle" if remove_only_middle else "")][corpus_key] = {
        "label_seqs": None,
        "time-red-6": {
          "align": bpe_sil_wo_sil_align_job.out_alignment,
          "seq_filter_file": data_dict["bpe-with-sil"][corpus_key]["time-red-6"]["seq_filter_file"],
          "label_dep_mean_lens": bpe_sil_wo_sil_label_dep_mean_lens,
          "mean_non_sil_len": bpe_sil_wo_sil_mean_non_sil_len, "95_percentile": bpe_sil_wo_sil_95_percentile}}
      if corpus_key == "train":
        seq_filter_files_bpe_sil_devtrain = update_seq_list_file(seq_list_file=seq_filter_files_standard["devtrain"],
          seqs_to_skip=bpe_sil_skipped_seqs, alias="seq_filter_files_bpe_sil/time-red-%s/%s" % (6, "devtrain"))
        data_dict["bpe-sil-wo-sil%s" % ("-in-middle" if remove_only_middle else "")]["devtrain"] = {
          "time-red-6": {
            "seq_filter_file": seq_filter_files_bpe_sil_devtrain}}

      # ----------------------- BPE Sil w/o Sil Center Position ALIGNMENTS -----------------------------------------
      bpe_center_pos_align = alignment_center_seg_boundaries(
        blank_idx=1031,
        alias="bpe_sil_wo_sil_center_pos_align/align_%s" % corpus_key, alignment=bpe_sil_wo_sil_align_job.out_alignment,
        seq_filter_file=data_dict["bpe-with-sil"][corpus_key]["time-red-6"]["seq_filter_file"])
      bpe_center_pos_label_dep_mean_lens, bpe_center_pos_mean_non_sil_len, bpe_center_pos_95_percentile = calc_align_stats(
        alignment=bpe_center_pos_align, blank_idx=1031,
        seq_filter_file=data_dict["bpe-with-sil"][corpus_key]["time-red-6"]["seq_filter_file"],
        alias="bpe_sil_wo_sil_center_pos_align_stats/time-red-6/stats_" + corpus_key)

      bpe_center_pos_labels_job = DumpNonBlanksFromAlignmentJob(alignment=bpe_center_pos_align, blank_idx=1031,
                                                                time_rqmt=time_rqmt)
      bpe_center_pos_labels_job.add_alias("bpe_sil_wo_sil_center_pos_labels/%s" % corpus_key)
      tk.register_output(bpe_center_pos_labels_job.get_one_alias(), bpe_center_pos_labels_job.out_labels)

      data_dict["bpe-sil-wo-sil-center-positions"].update({
        "json_vocab": bpe_sil_vocab_path, "state_tying": bpe_sil_state_tying, "allophones": bpe_sil_allophones,
        "rasr_label_file": bpe_sil_rasr_label_file})
      data_dict["bpe-sil-wo-sil-center-positions"][corpus_key] = {
        "label_seqs": None,
        "time-red-6": {
          "align": bpe_center_pos_align,
          "seq_filter_file": data_dict["bpe-with-sil"][corpus_key]["time-red-6"]["seq_filter_file"],
          "label_dep_mean_lens": bpe_center_pos_label_dep_mean_lens,
          "mean_non_sil_len": bpe_center_pos_mean_non_sil_len,
          "95_percentile": bpe_center_pos_95_percentile}}
      if corpus_key == "train":
        seq_filter_files_bpe_sil_devtrain = update_seq_list_file(
          seq_list_file=seq_filter_files_standard["devtrain"],
          seqs_to_skip=bpe_sil_skipped_seqs,
          alias="seq_filter_files_bpe_sil/time-red-%s/%s" % (6, "devtrain"))
        data_dict["bpe-sil-wo-sil-center-positions"]["devtrain"] = {
          "time-red-6": {
            "seq_filter_file": seq_filter_files_bpe_sil_devtrain}}

  compare_swb_alignments(align_dict=data_dict)


def compare_swb_alignments(align_dict):
  seq_tag = "switchboard-1/sw02102A/sw2102A-ms98-a-0002"

  align1 = "bpe"
  align2 = "bpe-with-sil"
  compare_aligns_job = CompareAlignmentsJob(
    hdf_align1=align_dict[align1]["cv"]["time-red-6"]["align"],
    hdf_align2=align_dict[align2]["cv"]["time-red-6"]["align"],
    seq_tag=seq_tag, blank_idx1=1030, blank_idx2=1031, vocab1=align_dict[align1]["json_vocab"],
    vocab2=align_dict[align2]["json_vocab"], name1=align1, name2=align2)
  compare_aligns_job.add_alias(
    "compare-aligns/" + align1 + "_vs_" + align2)
  tk.register_output(compare_aligns_job.get_one_alias(), compare_aligns_job.out_align)

  align1 = "bpe"
  align2 = "bpe-center-positions"
  compare_aligns_job = CompareAlignmentsJob(
    hdf_align1=align_dict[align1]["cv"]["time-red-6"]["align"],
    hdf_align2=align_dict[align2]["cv"]["time-red-6"]["align"],
    seq_tag=seq_tag, blank_idx1=1030, blank_idx2=1030, vocab1=align_dict[align1]["json_vocab"],
    vocab2=align_dict[align2]["json_vocab"], name1=align1, name2=align2)
  compare_aligns_job.add_alias(
    "compare-aligns/" + align1 + "_vs_" + align2)
  tk.register_output(compare_aligns_job.get_one_alias(), compare_aligns_job.out_align)

  align1 = "bpe-with-sil"
  align2 = "bpe-with-sil-split-sil"
  compare_aligns_job = CompareAlignmentsJob(
    hdf_align1=align_dict[align1]["cv"]["time-red-6"]["align"],
    hdf_align2=align_dict[align2]["cv"]["time-red-6"]["align"],
    seq_tag=seq_tag, blank_idx1=1031, blank_idx2=1031, vocab1=align_dict[align1]["json_vocab"],
    vocab2=align_dict[align2]["json_vocab"], name1=align1, name2=align2)
  compare_aligns_job.add_alias(
    "compare-aligns/" + align1 + "_vs_" + align2)
  tk.register_output(compare_aligns_job.get_one_alias(), compare_aligns_job.out_align)

  align1 = "bpe-with-sil"
  align2 = "bpe-sil-wo-sil"
  compare_aligns_job = CompareAlignmentsJob(
    hdf_align1=align_dict[align1]["cv"]["time-red-6"]["align"],
    hdf_align2=align_dict[align2]["cv"]["time-red-6"]["align"],
    seq_tag=seq_tag, blank_idx1=1031, blank_idx2=1031, vocab1=align_dict[align1]["json_vocab"],
    vocab2=align_dict[align2]["json_vocab"], name1=align1, name2=align2)
  compare_aligns_job.add_alias(
    "compare-aligns/" + align1 + "_vs_" + align2)
  tk.register_output(compare_aligns_job.get_one_alias(), compare_aligns_job.out_align)


def create_librispeech_alignments(
  data_dict, rasr_nn_trainer, phon_extraction_rasr_configs):
  for corpus_key in [
    "train"
  ]:
    if corpus_key == "cv":
      time_rqmt = 2
      mem_rqmt = 4
    else:
      time_rqmt = 10
      mem_rqmt = 6


    # ----------------------- PHONEME ALIGNMENTS -----------------------------------------
    # extract phoneme alignments
    phoneme_align, phoneme_vocab_path = dump_phoneme_align(
      time_rqmt=time_rqmt, rasr_exe=rasr_nn_trainer, rasr_config=phon_extraction_rasr_configs[corpus_key],
      mem_rqmt=mem_rqmt, time_red=1, alias="phon_align_ls/%s/%s" % ("time-red-1", corpus_key),
      state_tying_file=Path("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/librispeech/dependencies/luescher-phoneme-align/state-tying"))

    # phon_state_tying, phon_allophones, phon_rasr_label_file = convert_phon_json_vocab_to_rasr_formats(
    #   phoneme_vocab_path, blank_idx=89)
    #
    # # calculate alignment statistics for phoneme alignment without time reduction
    # phoneme_label_dep_mean_lens, phoneme_mean_non_sil_len, phoneme_95_percentile = calc_align_stats(
    #   alignment=phoneme_align,
    #   seq_filter_file=bpe_seq_filter_file, alias="phon_align_stats/stats_" + corpus_key)
    #
    # phoneme_labels_job = DumpNonBlanksFromAlignmentJob(
    #   alignment=phoneme_align, blank_idx=89, time_rqmt=time_rqmt
    # )
    # phoneme_labels_job.add_alias("phoneme_labels/%s" % corpus_key)
    # tk.register_output("phoneme_labels/%s" % corpus_key, phoneme_labels_job.out_labels)
    #
    #
    # data_dict["phonemes"].update({
    #   "json_vocab": phoneme_vocab_path,
    #   "state_tying": phon_state_tying, "allophones": phon_allophones, "rasr_label_file": phon_rasr_label_file})
    # data_dict["phonemes"][corpus_key] = {
    #   "label_seqs": phoneme_labels_job.out_labels,
    #   "time-red-1": {
    #     "align": phoneme_align,
    #     "seq_filter_file": bpe_seq_filter_file, "label_dep_mean_lens": phoneme_label_dep_mean_lens,
    #     "mean_non_sil_len": phoneme_mean_non_sil_len, "95_percentile": phoneme_95_percentile}}
    # if corpus_key == "train":
    #   data_dict["phonemes"]["devtrain"] = {
    #     "time-red-1": {
    #       "seq_filter_file": seq_filter_files_standard["devtrain"]}}
