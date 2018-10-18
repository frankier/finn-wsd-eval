import tempfile
import click
from plumbum import local
from plumbum.cmd import python, pipenv, ln
from os.path import join as pjoin


@click.group()
def ctx2vec():
    pass


def get_xml_key_pair(xml_path, key_path):
    tempdir = tempfile.mkdtemp(prefix="ctx2vec")
    pair_path = pjoin(tempdir, "corpus")
    ln("-s", xml_path, pair_path)
    ln("-s", key_path, pjoin(tempdir, "corpus.key"))
    return pair_path


@ctx2vec.command()
@click.argument("modelin")
@click.argument("trainin")
@click.argument("keyin")
@click.argument("testin")
@click.argument("resultout")
def test(modelin, trainin, keyin, testin, testkeyin, resultout):
    train_path = get_xml_key_pair(trainin, keyin)
    test_path = get_xml_key_pair(testin, testkeyin)
    tempdir = tempfile.mkdtemp(prefix="ctx2vec")
    result_path = pjoin(tempdir, "results")
    with local.cwd("systems/context2vec"):
        pipenv(
            "run",
            "python",
            "context2vec/eval/wsd/wsd_main.py",
            train_path,
            test_path,
            result_path,
            modelin,
            "1",
        )
    python(__file__, "context2vec-key-to-unified", result_path, resultout)


@ctx2vec.command("context2vec-key-to-unified")
@click.argument("keyin", type=click.File("r"))
@click.argument("keyout", type=click.File("w"))
def context2vec_key_to_unified(keyin, keyout):
    for line in keyin:
        bits = line.split(" ")
        iden = bits[1]
        guesses = bits[2:]
        if guesses:
            guessed = guesses[0].split("/")[0]
            keyout.write("{} {}\n".format(iden, guessed))


if __name__ == "__main__":
    ctx2vec()
