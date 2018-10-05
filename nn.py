import click
import pickle

from finntk.wordnet.reader import fiwn
from finntk.wsd.lesk_emb import mk_context_vec_fasttext_fi
from stiff.utils.xml import iter_blocks
from means import ALL_MEANS
from utils import lemmas_from_instance


@click.group()
def nn():
    pass


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

    for instance in iter_blocks("instance")(inf):
        context_tag = instance.xpath("context")[0]
        tokens = "".join(context_tag.xpath('//text()')).split(" ")
        context = set()
        for token in tokens:
            context.add(repr_instance_ctx(token))
        inst_id = instance.attrib["id"]
        key_id, synset_id = next(keyin).strip().split()
        assert inst_id == key_id
        head_word = context_tag.xpath("head")[0].text
        wf = repr_instance_ctx(head_word)
        sub_ctx = context - {wf}
        ctx_vec = repr_ctx(aggf, sub_ctx)
        print("add_word(wf, ctx_vec, synset_id)", wf, ctx_vec, synset_id)
        classifier.add_word(wf, ctx_vec, synset_id)

    classifier.fit()
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

    for sent in iter_blocks("context")(inf):
        context = set()
        word_tags = sent.xpath("/wf|instance")
        for tag in word_tags:
            context.add(repr_instance_ctx(tag))
        for instance in sent.xpath("instance"):
            wf = repr_instance_ctx(instance)
            sub_ctx = context - {wf}
            ctx_vec = repr_ctx(sub_ctx)
            _lemma_str, _pos, lemmas = lemmas_from_instance(fiwn, instance)
            keyout.write("{}\n".format(classifier.predict(wf, ctx_vec)))


if __name__ == '__main__':
    nn()
