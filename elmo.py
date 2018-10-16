import click
import pickle
from vec_nn_utils import mk_training_examples, train_vec_nn, test_vec_nn


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
        vec = vecs[len(be) : len(be) + len(he)].mean(axis=0)
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
    classifier = train_vec_nn(
        mk_training_examples(iter_inst_vecs(inf, output_layer), keyin)
    )

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

    test_vec_nn(classifier, iter_inst_vecs(inf, output_layer), keyout)
