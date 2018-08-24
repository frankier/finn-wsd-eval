import tempfile
import os
import click
from plumbum import local
from plumbum.cmd import git, python, pipenv, ln
from shutil import copyfile
from os.path import join as pjoin


@click.group()
def ctx2vec():
    pass


@ctx2vec.command()
def fetch():
    os.makedirs("systems", exist_ok=True)
    with local.cwd("systems"):
        git("clone", "https://github.com/orenmel/context2vec.git")

    copyfile("support/context2vec/Pipfile", "systems/context2vec/Pipfile")
    copyfile("support/context2vec/Pipfile.lock", "systems/context2vec/Pipfile.lock")

    with local.cwd("systems/context2vec"):
        pipenv("install")


@ctx2vec.command()
@click.argument("modelin")
@click.argument("trainin")
@click.argument("keyin")
@click.argument("testin")
@click.argument("resultout")
def test(modelin, trainin, keyin, testin, resultout):
    tempdir = tempfile.mkdtemp(prefix="ctx2vec")
    train_path = pjoin(tempdir, "train")
    ln("-s", trainin, train_path)
    ln("-s", keyin, pjoin(tempdir, "train.key"))
    result_path = pjoin(tempdir, "results")
    with local.cwd("systems/context2vec"):
        pipenv("run", "python", "eval/wsd/wsd_main.py", train_path, testin, result_path, modelin, "1")
    python(__file__, "context2vec-key-to-unified", result_path, resultout)


@ctx2vec.command("context2vec-key-to-unified")
@click.argument("keyin", type=click.File("r"))
@click.argument("keyout", type=click.File("w"))
def context2vec_key_to_unified(keyin, keyout):
    for line in keyin:
        bits = line.split(" ")
        lemma_pos = bits[0]
        iden = bits[1]
        guesses = bits[2:]
        if guesses:
            guessed = guesses[0].split("/")[0]
            keyout.write("{} {} {}\n".format(lemma_pos, iden, guessed))


if __name__ == "__main__":
    ctx2vec()
