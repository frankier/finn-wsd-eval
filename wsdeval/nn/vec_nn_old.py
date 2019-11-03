import logging

from finntk.wsd.nn import FixedWordExpert
from .vec_nn_common import write_most_freq


logger = logging.getLogger(__name__)


def train_many_vec_nn(managers, training_examples):
    for iden, cnt, group in training_examples:
        clfs = [FixedWordExpert(cnt, algorithm="brute") for _ in managers]
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
        clf = FixedWordExpert(cnt, algorithm="brute")
        empty = True
        for _, vec, synset_id in group:
            if vec is None:
                continue
            empty = False
            clf.add_word(vec, synset_id)
        if empty:
            continue
        clf.fit()
        manager.dump_expert(iden, clf)


def pred_write(inst_id, clf, vec, keyout, allow_most_freq=False, iden=None):
    prediction = None
    if clf is not None and vec is not None:
        prediction = clf.predict(vec)
    if prediction is None:
        if allow_most_freq:
            if write_most_freq(inst_id, iden, keyout):
                return
        prediction = "U"
    keyout.write("{} {}\n".format(inst_id, prediction))


def test_many_vec_nn(managers, instances, keyouts, allow_most_freq=False):
    logger.debug(f"test_many_vec_nn: keyouts: {keyouts}")
    wrote_anything = False
    for iden, cnt, group in instances:
        clfs = [manager.load_expert(iden) if manager else None for manager in managers]
        for inst_id, vecs in group:
            if vecs is None:
                vecs = [None] * len(clfs)
            for clf, vec, keyout in zip(clfs, vecs, keyouts):
                pred_write(
                    inst_id,
                    clf,
                    vec,
                    keyout,
                    allow_most_freq=allow_most_freq,
                    iden=iden,
                )
                wrote_anything = True
    if wrote_anything:
        logger.debug("Wrote at least something...")


def test_vec_nn(manager, instances, keyout, allow_most_freq=False):
    for iden, cnt, group in instances:
        clf = manager.load_expert(iden)
        for inst_id, vec in group:
            pred_write(
                inst_id, clf, vec, keyout, allow_most_freq=allow_most_freq, iden=iden
            )
