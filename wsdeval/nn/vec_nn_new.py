import logging

from finntk.wordnet.utils import fi2en_post
from finntk.wordnet.reader import fiwn_encnt
from .vec_nn_common import lemmas_from_iden, write_most_freq
from .new import nearest_from_mats, value_from_mat


logger = logging.getLogger(__name__)


def train_word_experts(manager, training_examples):
    for iden, cnt, group in training_examples:
        group_it = (
            (vec, synset_id.encode("utf-8"))
            for _inst_id, vec, synset_id in group
            if vec is not None
        )
        manager.add_group(iden, group_it)


def train_synset_examples(manager, instances):
    for iden, cnt, it in instances:
        manager.add_group(iden, (vec for _inst_id, vec in it))


def write_one(keyout, inst_id, prediction):
    keyout.write("{} {}\n".format(inst_id, prediction.decode("utf-8")))


def pred_write(keyout, inst_id, allow_most_freq, iden, pred, mapper):
    if pred is None:
        if allow_most_freq:
            if write_most_freq(inst_id, iden, keyout):
                return
            prediction = "U"
    else:
        prediction = mapper(pred)
    write_one(keyout, inst_id, prediction)


def test_word_experts(grouped, instances, keyout, allow_most_freq=False):
    for iden, cnt, group in instances:
        mat, val_it = grouped.get_group(iden)
        synset_ids = list(val_it)
        for inst_id, vec in group:
            best_synset = value_from_mat(mat, synset_ids, vec)
            pred_write(
                keyout,
                inst_id,
                allow_most_freq,
                iden,
                best_synset,
                lambda best_synset: best_synset,
            )


def test_synset_experts(grouped, instances, keyout, allow_most_freq=False):
    for iden, cnt, group in instances:
        lemmas = lemmas_from_iden(iden)
        vec_mats = []
        synsets = []
        for lemma in lemmas:
            of = fi2en_post(fiwn_encnt.ss2of(lemma.synset()))
            vec_mats.append(grouped.get_group(of))
            synsets.append(of)
        for inst_id, query_vec in group:
            best_idx = nearest_from_mats(vec_mats, query_vec)
            pred_write(
                keyout,
                inst_id,
                allow_most_freq,
                iden,
                best_idx,
                lambda best_idx: synsets[best_idx],
            )
