from sup_corpus import next_key


def mk_training_examples(instances, keyin):
    for inst_id, item, vec in instances:
        key_id, synset_ids = next_key(keyin)
        assert inst_id == key_id
        yield inst_id, item, vec, synset_ids[0]


def train_vec_nn(training_examples):
    from finntk.wsd.nn import WsdNn

    classifier = WsdNn()

    prev_item = None
    for inst_id, item, vec, synset_id in training_examples:
        if vec is not None:
            classifier.add_word(item, vec, synset_id)

        if prev_item is not None and item != prev_item:
            classifier.fit_word(prev_item)
        prev_item = item
    classifier.fit_word(prev_item)
    return classifier


def test_vec_nn(classifier, instances, keyout):
    for inst_id, item, vec in instances:
        prediction = None
        if vec is not None:
            try:
                prediction = classifier.predict(item, vec)
            except KeyError:
                pass
        if prediction is None:
            prediction = "U"
        keyout.write("{} {}\n".format(inst_id, prediction))
