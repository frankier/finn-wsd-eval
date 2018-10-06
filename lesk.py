import click

from finntk.wordnet.reader import fiwn
from finntk.wsd.lesk_emb import disambg_conceptnet, disambg_fasttext, disambg_double
from stiff.utils.xml import iter_sentences
from means import ALL_MEANS
from utils import lemmas_from_instance, write_lemma


@click.command()
@click.argument("vec")
@click.argument("mean")
@click.argument("inf", type=click.File("rb"))
@click.argument("keyout", type=click.File("w"))
@click.option("--wn-filter/--no-wn-filter")
def lesk(vec, mean, inf, keyout, wn_filter):
    if vec == "fasttext":
        disambg = disambg_fasttext
        repr_ctx = lambda inst: inst.text.lower()
    elif vec == "numberbatch":
        disambg = disambg_conceptnet
        repr_ctx = lambda inst: inst.attrib["lemma"].lower()
    else:
        disambg = disambg_double
        repr_ctx = lambda inst: (inst.text.lower(), inst.attrib["lemma"].lower())
    return wordvec_lesk(
        ALL_MEANS[mean],
        repr_ctx,
        disambg,
        wn_filter,
        inf,
        keyout
    )


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


if __name__ == '__main__':
    lesk()
