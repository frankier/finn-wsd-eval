import click
import pickle


@click.group()
def elmo():
    pass


def iter_inst_vecs(inf, output_layer):
    from sup_corpus import iter_instances
    from finntk.emb.elmo import vecs
    from finntk.vendor.elmo import embed_sentence
    model = vecs.get()
    for inst_id, item_pos, (be, he, af) in iter_instances(inf):
        sent = be + he + af
        vecs = embed_sentence(model, sent, output_layer)
        vec = vecs[len(be): len(be) + len(he)].mean(axis=0)
        yield inst_id, item_pos, vec


@elmo.command("train")
@click.argument("inf", type=click.File("rb"))
@click.argument("keyin", type=click.File("r"))
@click.argument("model", type=click.File("wb"))
@click.argument("output_layer", type=int)
def train(inf, keyin, model, output_layer):
    """
    Train nearest neighbour classifier with ELMo.
    """
    from finntk.wsd.nn import WsdNn
    classifier = WsdNn()

    prev_item = None
    for inst_id, item, vec in iter_inst_vecs(inf, output_layer):
        key_id, synset_id = next_key(keyin)
        assert inst_id == key_id
        if vec is not None:
            classifier.add_word(item, vec, synset_id)

        if prev_item is not None and item != prev_item:
            classifier.fit_word(prev_item)
        prev_item = item
    classifier.fit_word(prev_item)

    pickle.dump(classifier, model)


@elmo.command("test")
@click.argument("model", type=click.File("rb"))
@click.argument("inf", type=click.File("rb"))
@click.argument("keyout", type=click.File("w"))
@click.argument("output_layer", type=int)
def test(model, inf, keyout, output_layer):
    """
    Test nearest neighbour classifier.
    """
    classifier = pickle.load(model)

    for inst_id, item, vec in iter_inst_vecs(inf, output_layer):
        prediction = None
        if vec is not None:
            try:
                prediction = classifier.predict(item, vec)
            except KeyError:
                pass
        if prediction is None:
            prediction = "U"
        keyout.write("{} {}\n".format(inst_id, prediction))
