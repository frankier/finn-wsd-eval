import sys
import click
from stiff.filter_utils import iter_sentences
from stiff.data import UNI_POS_WN_MAP
from finntk.wordnet.reader import fiwn, fiwn_encnt, get_en_fi_maps
from finntk.wordnet.utils import pre_id_to_post, ss2pre
from finntk.wsd.lesk_emb import disambg_fasttext, disambg_conceptnet
from finntk.emb.utils import CATP_3, CATP_4, catp_mean, sif_mean, unnormalized_mean, normalized_mean
from functools import partial


MEANS = {
    "catp3_mean": partial(catp_mean, ps=CATP_3),
    "catp4_mean": partial(catp_mean, ps=CATP_4),
    "sif_mean": sif_mean,
    "unnormalized_mean": unnormalized_mean,
    "normalized_mean": normalized_mean,
}


@click.group()
def baselines():
    pass


def lemmas_from_instance(wn, instance):
    word = instance.attrib["lemma"]
    pos = UNI_POS_WN_MAP[instance.attrib["pos"]]
    lemmas = wn.lemmas(word, pos=pos)
    return word, pos, lemmas


def write_lemma(keyout, inst_id, lemma):
    fi2en, en2fi = get_en_fi_maps()
    chosen_synset_fi_id = ss2pre(lemma.synset())
    if chosen_synset_fi_id not in fi2en:
        sys.stderr.write(
            "No fi2en mapping found for {} ({})\n".format(chosen_synset_fi_id, lemma)
        )
        return
    keyout.write("{} {}\n".format(inst_id, pre_id_to_post(fi2en[chosen_synset_fi_id])))


def unigram(inf, keyout, wn):
    for sent in iter_sentences(inf):
        for instance in sent.xpath("instance"):
            inst_id = instance.attrib["id"]
            word, pos, lemmas = lemmas_from_instance(wn, instance)
            if not len(lemmas):
                sys.stderr.write("No lemma found for {} {}\n".format(word, pos))
                continue
            lemma = lemmas[0]
            write_lemma(keyout, inst_id, lemma)


@baselines.command()
@click.argument("inf", type=click.File("rb"))
@click.argument("keyout", type=click.File("w"))
def first(inf, keyout):
    """
    Just picks the first sense according to FinnWordNet (essentially random)
    """
    unigram(inf, keyout, fiwn)


@baselines.command()
@click.argument("inf", type=click.File("rb"))
@click.argument("keyout", type=click.File("w"))
def mfe(inf, keyout):
    """
    Picks the synset with the most usages *in English* (by summing along all
    its lemmas).
    """
    unigram(inf, keyout, fiwn_encnt)


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


@baselines.command()
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


@baselines.command()
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


@baselines.command()
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


if __name__ == "__main__":
    baselines()
