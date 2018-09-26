import click

from finntk.wordnet.reader import fiwn
from finntk.wsd.lesk_emb import disambg_conceptnet, disambg_fasttext
from stiff.utils.xml import iter_sentences
from means import MEANS
from utils import lemmas_from_instance, write_lemma


@click.group()
def lesk():
    pass


def wordvec_lesk(aggf, repr_instance_ctx, disambg, wn_filter, inf, keyout):
    for sent in iter_sentences(inf):
        context = set()
        word_tags = sent.xpath("/wf|instance")
        for tag in word_tags:
            context.add(repr_instance_ctx(tag))
        for instance in sent.xpath("instance"):
            inst_id = instance.attrib["id"]
            wf = repr_instance_ctx(instance)
            sub_ctx = context - {wf}
            _lemma_str, _pos, lemmas = lemmas_from_instance(fiwn, instance)
            lemma, dist = disambg(aggf, lemmas, sub_ctx, wn_filter=wn_filter)
            write_lemma(keyout, inst_id, lemma)


def lesk_fasttext_inner(aggf, wn_filter, inf, keyout):
    return wordvec_lesk(
        aggf,
        lambda inst: inst.text.lower(),
        disambg_fasttext,
        wn_filter,
        inf,
        keyout
    )


@lesk.command()
@click.argument("mean")
@click.argument("inf", type=click.File("rb"))
@click.argument("keyout", type=click.File("w"))
@click.option("--wn-filter/--no-wn-filter")
def lesk_fasttext(mean, inf, keyout, wn_filter):
    """
    Performs word vector averaging simplified Lesk
    """
    return lesk_fasttext_inner(
        MEANS[mean],
        wn_filter,
        inf,
        keyout,
    )


def lesk_conceptnet_inner(aggf, wn_filter, inf, keyout):
    return wordvec_lesk(
        aggf,
        lambda inst: inst.attrib["lemma"].lower(),
        disambg_conceptnet,
        wn_filter,
        inf,
        keyout,
    )


@lesk.command()
@click.argument("mean")
@click.argument("inf", type=click.File("rb"))
@click.argument("keyout", type=click.File("w"))
@click.option("--wn-filter/--no-wn-filter")
def lesk_conceptnet(mean, inf, keyout, wn_filter):
    """
    Performs word vector averaging simplified Lesk
    """
    wordvec_lesk(
        MEANS[mean],
        wn_filter,
        inf,
        keyout,
    )


def lesk_double_inner(aggf, wn_filter, inf, keyout):
    return wordvec_lesk(
        aggf,
        lambda inst: (inst.text.lower(), inst.attrib["lemma"].lower()),
        disambg_conceptnet,
        wn_filter,
        inf,
        keyout,
    )


@lesk.command()
@click.argument("mean")
@click.argument("inf", type=click.File("rb"))
@click.argument("keyout", type=click.File("w"))
@click.option("--wn-filter/--no-wn-filter")
def lesk_double(mean, inf, keyout, wn_filter):
    """
    Performs word vector averaging simplified Lesk
    """
    wordvec_lesk(
        MEANS[mean],
        wn_filter,
        inf,
        keyout,
    )


if __name__ == '__main__':
    lesk()
