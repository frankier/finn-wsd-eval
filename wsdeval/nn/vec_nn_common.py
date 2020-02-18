from itertools import starmap
import numpy as np

from stiff.sup_corpus import next_key
from finntk.wordnet.reader import fiwn_encnt
from wsdeval.formats.wordnet import write_lemma


class MismatchedGoldException(Exception):
    pass


def mk_training_examples(instances, keyin):
    def add_gold(inst_id, vec):
        key_id, synset_ids = next_key(keyin)

        if inst_id != key_id:
            raise MismatchedGoldException(
                f"Gold instance id {key_id} does not match expected instance id {inst_id}"
            )
        return inst_id, vec, synset_ids[0]

    for item, cnt, it in instances:
        yield item, cnt, starmap(add_gold, it)


def lemmas_from_iden(iden):
    lemma, pos = iden.split(".", 1)
    return fiwn_encnt.lemmas(lemma, pos=pos)


def write_most_freq(inst_id, iden, keyout):
    lemmas = lemmas_from_iden(iden)
    if not lemmas:
        return False
    write_lemma(keyout, inst_id, lemmas[0])
    return True


def normalize(vec):
    vec_norm = np.linalg.norm(vec)
    if not np.nonzero(vec_norm):
        return None
    return vec / vec_norm
