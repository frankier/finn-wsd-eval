from itertools import starmap
import logging

from wsdeval.formats.sup_corpus import next_key
from finntk.wsd.nn import FixedWordExpert


logger = logging.getLogger(__name__)


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


def train_many_vec_nn(managers, training_examples):
    for iden, cnt, group in training_examples:
        clfs = [FixedWordExpert(cnt) for _ in managers]
        for _, vecs, synset_id in group:
            if vecs is None:
                continue
            for clf, vec in zip(clfs, vecs):
                if clf is None:
                    continue
                clf.add_word(vec, synset_id)
        for manager, clf in zip(managers, clfs):
            if clf is None:
                continue
            clf.fit()
            manager.dump_expert(iden, clf)


def train_vec_nn(manager, training_examples):
    for iden, cnt, group in training_examples:
        clf = FixedWordExpert(cnt)
        for _, vec, synset_id in group:
            if vec is None:
                continue
            clf.add_word(vec, synset_id)
        clf.fit()
        manager.dump_expert(iden, clf)


def pred_write(inst_id, clf, vec, keyout):
    prediction = None
    if clf is not None and vec is not None:
        prediction = clf.predict(vec)
    if prediction is None:
        prediction = "U"
    keyout.write("{} {}\n".format(inst_id, prediction))


def test_many_vec_nn(managers, instances, keyouts):
    logger.debug(f"test_many_vec_nn: keyouts: {keyouts}")
    wrote_anything = False
    for iden, cnt, group in instances:
        clfs = [manager.load_expert(iden) if manager else None for manager in managers]
        for inst_id, vecs in group:
            if vecs is None:
                vecs = [None] * len(clfs)
            for clf, vec, keyout in zip(clfs, vecs, keyouts):
                pred_write(inst_id, clf, vec, keyout)
                wrote_anything = True
    if wrote_anything:
        logger.debug("Wrote at least something...")


def test_vec_nn(manager, instances, keyout):
    for iden, cnt, group in instances:
        clf = manager.load_expert(iden)
        for inst_id, vec in group:
            pred_write(inst_id, clf, vec, keyout)
