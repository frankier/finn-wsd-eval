import click
import pickle

from finntk.wsd.lesk_emb import mk_context_vec_fasttext_fi
from stiff.utils.xml import iter_blocks
from means import ALL_MEANS


@click.group()
def nn():
    pass


def iter_instances(inf, aggf, repr_instance_ctx, repr_ctx):
    for lexelt in iter_blocks("lexelt")(inf):
        item = lexelt.get("item")
        pos = lexelt.get("pos")
        item_pos = "{}.{}".format(item, pos)
        for instance in lexelt.xpath("instance"):
            context_tag = instance.xpath("context")[0]
            tokens = "".join(context_tag.xpath('//text()')).split(" ")
            context = set()
            for token in tokens:
                context.add(repr_instance_ctx(token))
            inst_id = instance.attrib["id"]
            head_word = context_tag.xpath("head")[0].text
            wf = repr_instance_ctx(head_word)
            sub_ctx = context - {wf}
            ctx_vec = repr_ctx(aggf, sub_ctx) if sub_ctx else None
            yield inst_id, item_pos, ctx_vec


@nn.command("train")
@click.argument("mean")
@click.argument("inf", type=click.File("rb"))
@click.argument("keyin", type=click.File("r"))
@click.argument("model", type=click.File("wb"))
@click.option("--wn-filter/--no-wn-filter")
def train(mean, inf, keyin, model, wn_filter):
    """
    Train nearest neighbour classifier.
    """
    return train_nn(
        ALL_MEANS[mean],
        lambda inst: inst.lower(),
        mk_context_vec_fasttext_fi,
        wn_filter,
        inf,
        keyin,
        model,
    )


def train_nn(aggf, repr_instance_ctx, repr_ctx, wn_filter, inf, keyin, model):
    from finntk.wsd.nn import WsdNn
    classifier = WsdNn()

    prev_item = None
    for inst_id, item, ctx_vec in iter_instances(inf, aggf, repr_instance_ctx, repr_ctx):
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
@click.option("--wn-filter/--no-wn-filter")
def test(mean, model, inf, keyout, wn_filter):
    """
    Test nearest neighbour classifier.
    """
    return test_nn(
        ALL_MEANS[mean],
        lambda inst: inst.lower(),
        mk_context_vec_fasttext_fi,
        wn_filter,
        inf,
        keyout,
        model,
    )


def test_nn(aggf, repr_instance_ctx, repr_ctx, wn_filter, inf, keyout, model):
    classifier = pickle.load(model)

    for inst_id, item, ctx_vec in iter_instances(inf, aggf, repr_instance_ctx, repr_ctx):
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
