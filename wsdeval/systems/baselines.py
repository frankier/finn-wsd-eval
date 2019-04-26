import sys
import click
from finntk.wordnet.reader import fiwn, fiwn_encnt
from stiff.utils.xml import iter_sentences
from wsdeval.formats.wordnet import lemmas_from_instance, write_lemma


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


@click.group()
def baselines():
    pass


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


if __name__ == "__main__":
    baselines()
