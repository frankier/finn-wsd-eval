import click
import pickle

from finntk.wsd.lesk_emb import mk_context_vec_fasttext_fi
from stiff.utils.xml import iter_blocks
from means import ALL_MEANS


@click.group()
def nn():
    pass


def iter_inst_ctxs(inf, aggf, repr_instance_ctx, repr_ctx):
    from sup_corpus import iter_instances
    for inst_id, item_pos, (be, he, af) in iter_instances(inf):
        ctx = map(repr_instance_ctx, be + af)
        ctx_vec = repr_ctx(aggf, ctx) if ctx else None
        yield inst_id, item_pos, ctx_vec


@nn.command("train")
@click.argument("mean")
@click.argument("inf", type=click.File("rb"))
@click.argument("keyin", type=click.File("r"))
@click.argument("model", type=click.File("wb"))
def train(mean, inf, keyin, model):
    """
    Train nearest neighbour classifier.
    """
    return train_nn(
        ALL_MEANS[mean],
        lambda inst: inst.lower(),
        mk_context_vec_fasttext_fi,
        inf,
        keyin,
        model,
    )


def train_nn(aggf, repr_instance_ctx, repr_ctx, inf, keyin, model):
    from finntk.wsd.nn import WsdNn
    classifier = WsdNn()

    prev_item = None
    for inst_id, item, ctx_vec in iter_inst_ctxs(inf, aggf, repr_instance_ctx, repr_ctx):
        key_id, synset_id = next(keyin).strip().split()
        assert inst_id == key_id
        if ctx_vec is not None:
            classifier.add_word(item, ctx_vec, synset_id)

        if prev_item is not None and item != prev_item:
            classifier.fit_word(prev_item)
        prev_item = item
    classifier.fit_word(prev_item)

    pickle.dump(classifier, model)


@nn.command("test")
@click.argument("mean")
@click.argument("model", type=click.File("rb"))
@click.argument("inf", type=click.File("rb"))
@click.argument("keyout", type=click.File("w"))
def test(mean, model, inf, keyout):
    """
    Test nearest neighbour classifier.
    """
    return test_nn(
        ALL_MEANS[mean],
        lambda inst: inst.lower(),
        mk_context_vec_fasttext_fi,
        inf,
        keyout,
        model,
    )


def test_nn(aggf, repr_instance_ctx, repr_ctx, inf, keyout, model):
    classifier = pickle.load(model)

    for inst_id, item, ctx_vec in iter_inst_ctxs(inf, aggf, repr_instance_ctx, repr_ctx):
        prediction = None
        if ctx_vec is not None:
            try:
                prediction = classifier.predict(item, ctx_vec)
            except KeyError:
                pass
        if prediction is None:
            prediction = "U"
        keyout.write("{} {}\n".format(inst_id, prediction))


if __name__ == '__main__':
    nn()
