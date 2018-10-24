import click
from finntk.wordnet.reader import fiwn, fiwn_encnt
from wsdeval.utils import unigram


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
