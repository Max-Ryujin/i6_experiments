from i6_core.corpus.filter import FilterCorpusBySegmentsJob

from i6_experiments.common.datasets.librispeech import get_g2p_augmented_bliss_lexicon_dict
from i6_experiments.common.setups.rasr.util import RasrDataInput

from i6_experiments.users.rossenbach.datasets.librispeech import get_librispeech_tts_segments, get_ls_train_clean_100_tts_silencepreprocessed


def get_corpus_data_inputs():
    """
    :return: a 3-sized tuple containing lists of RasrDataInput for train, dev and test
    """

    # this is the FFmpeg silence preprocessed version of LibriSpeech train-clean-100
    sil_pp_train_clean_100 = get_ls_train_clean_100_tts_silencepreprocessed()

    # segments for train-clean-100-tts-train and train-clean-100-tts-dev
    # (1004 segments for dev, 4 segments for each of the 251 speakers)
    train_segments, dev_segments = get_librispeech_tts_segments()

    # remove the dev segments from the GMM training corpus
    sil_pp_train_clean_100_tts_train_xml = FilterCorpusBySegmentsJob(
        sil_pp_train_clean_100.corpus_file,
        train_segments,
        compressed=True,
    ).out_corpus
    sil_pp_train_clean_100.corpus_file = sil_pp_train_clean_100_tts_train_xml

    g2p_lexica = get_g2p_augmented_bliss_lexicon_dict(
        output_prefix="corpora",
        add_unknown_phoneme_and_mapping=False,
    )
    lm = None
    lexicon = {
        'filename': g2p_lexica['train-clean-100'],
        'normalize_pronunciation': False,
    }

    train_data_inputs = {}
    dev_data_inputs = {}
    test_data_inputs = {}

    train_data_inputs['train-clean-100'] = RasrDataInput(
        corpus_object=sil_pp_train_clean_100,
        concurrent=10,
        lexicon=lexicon,
        lm=lm,
    )

    return train_data_inputs, dev_data_inputs, test_data_inputs