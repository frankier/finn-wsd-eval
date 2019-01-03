from wsdeval.formats.sup_corpus import next_key
from itertools import groupby
from finntk.wsd.nn import WordExpert


def mk_training_examples(instances, keyin):
    for inst_id, item, vec in instances:
        key_id, synset_ids = next_key(keyin)
        assert inst_id == key_id
        yield inst_id, item, vec, synset_ids[0]


def lemma_group(instances):
    for item, group in groupby(instances, lambda x: x[1]):
        yield ".".join(item), group


def train_many_vec_nn(managers, training_examples):
    for iden, group in lemma_group(training_examples):
        clfs = [WordExpert() for _ in managers]
        for _, _, vecs, synset_id in group:
            if vecs is None:
                continue
            for clf, vec in zip(clfs, vecs):
                clf.add_word(vec, synset_id)
        for manager, clf in zip(managers, clfs):
            clf.fit()
            manager.dump_expert(iden, clf)


def train_vec_nn(manager, training_examples):
    for iden, group in lemma_group(training_examples):
        clf = WordExpert()
        for _, _, vec, synset_id in group:
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
    for iden, group in lemma_group(instances):
        clfs = [manager.load_expert(iden) for manager in managers]
        for inst_id, _, vecs in group:
            for clf, vec, keyout in zip(clfs, vecs, keyouts):
                pred_write(inst_id, clf, vec, keyout)


def test_vec_nn(manager, instances, keyout):
    for iden, group in lemma_group(instances):
        clf = manager.load_expert(iden)
        for inst_id, _, vec in group:
            pred_write(inst_id, clf, vec, keyout)
