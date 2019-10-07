from plumbum import local
from plumbum.cmd import git
import click
import os
from finntk.emb.word2vec import vecs as word2vec_res
from finntk.emb.fasttext import vecs as fasttext_res
from os.path import join as pjoin


@click.command()
def supwsd():
    fetch_program()
    fetch_emb()


def fetch_emb():
    emb = "support/emb"
    os.makedirs(emb, exist_ok=True)
    word2vec_res.get_vecs().save_word2vec_format(pjoin(emb, "word2vec.txt"))
    fasttext_res.get_fi().save_word2vec_format(pjoin(emb, "fasttext.txt"))


def fetch_program():
    from plumbum.cmd import mvn

    os.makedirs("systems", exist_ok=True)
    with local.cwd("systems"):
        git("clone", "https://github.com/frankier/supWSD.git")
        with local.cwd("supWSD"):
            git("checkout", "fix-null-exception")
            mvn("package")


if __name__ == "__main__":
    supwsd()
