from sup_corpus import next_key
from itertools import groupby


def mk_training_examples(instances, keyin):
    for inst_id, item, vec in instances:
        key_id, synset_ids = next_key(keyin)
        assert inst_id == key_id
        yield inst_id, item, vec, synset_ids[0]


def train_vec_nn(manager, training_examples):
    from finntk.wsd.nn import WordExpert

    for item, group in groupby(training_examples, lambda x: x[1]):
        clf = WordExpert()
        for inst_id, item, vec, synset_id in group:
            if vec is None:
                continue
            clf.add_word(vec, synset_id)
        clf.fit()
        manager.dump_expert(".".join(item), clf)


def test_vec_nn(manager, instances, keyout):
    for item, group in groupby(instances, lambda x: x[1]):
        clf = manager.load_expert(".".join(item))
        for inst_id, item, vec in group:
            prediction = None
            if clf is not None and vec is not None:
                prediction = clf.predict(vec)
            if prediction is None:
                prediction = "U"
            keyout.write("{} {}\n".format(inst_id, prediction))
