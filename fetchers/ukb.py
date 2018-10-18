from plumbum import local
from plumbum.cmd import python, make, bash, git
import click
import os
from os.path import abspath
from finntk.wordnet.reader import fiwn_encnt
from finntk.wordnet.utils import fi2en_post


@click.group()
def ukb():
    pass


@ukb.command()
def fetch():
    os.makedirs("systems", exist_ok=True)
    with local.cwd("systems"):
        git("clone", "https://github.com/asoroa/ukb.git")
        with local.cwd("ukb/src"):
            local["./configure"]()
            make()
    # Prepare
    with local.env(UKB_PATH=abspath("systems/ukb/src")):
        with local.cwd("support/ukb"):
            bash("./prepare_wn30graph.sh")
    (python[__file__, "mkwndict", "--en-synset-ids"] > "support/ukb/wndict.fi.txt")()


@ukb.command()
@click.option("--en-synset-ids/--fi-synset-ids")
def mkwndict(en_synset_ids):
    lemma_names = fiwn_encnt.all_lemma_names()

    for lemma_name in lemma_names:
        lemmas = fiwn_encnt.lemmas(lemma_name)
        synsets = []
        for lemma in lemmas:
            synset = lemma.synset()
            post_synset_id = fiwn_encnt.ss2of(synset)
            if en_synset_ids:
                post_synset_id = fi2en_post(post_synset_id)
            synsets.append("{}:{}".format(post_synset_id, lemma.count()))
        if not lemma_name:
            continue
        print("{}\t{}".format(lemma_name, " ".join(synsets)))


if __name__ == "__main__":
    ukb()
