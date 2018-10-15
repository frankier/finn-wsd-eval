import os
import click
from plumbum import local
from plumbum.cmd import git, java
from string import Template
from os.path import join as pjoin
from finntk.emb.word2vec import vecs as word2vec_res
from finntk.emb.fasttext import vecs as fasttext_res


SUPWSD_JAR = "target/supwsd-toolkit-1.0.0.jar"


@click.group()
def supwsd():
    pass


@supwsd.command()
def fetch():
    fetch_program.callback()
    fetch_emb.callback()


@supwsd.command()
def fetch_emb():
    emb = "support/emb"
    os.makedirs(emb, exist_ok=True)
    word2vec_res.get_vecs().save_word2vec_format(pjoin(emb, "word2vec.txt"))
    fasttext_res.get_fi().save_word2vec_format(pjoin(emb, "fasttext.txt"))


@supwsd.command()
def fetch_program():
    from plumbum.cmd import mvn

    os.makedirs("systems", exist_ok=True)
    with local.cwd("systems"):
        git("clone", "https://github.com/frankier/supWSD.git")
        with local.cwd("supWSD"):
            git("checkout", "fixes-sep-24-1")
            mvn("package")


@supwsd.command()
def conf(work_dir, vec_path="", use_vec=False, use_surrounding_words=True):
    from finntk.wordnet.reader import fiwn_resman

    fiwn_path = fiwn_resman.get_res("")
    for src_fn, dst_fn in [
        ("jwnl-properties.xml", "resources/wndictionary/prop.xml"),
        ("supconfig.xml", "supconfig.xml"),
    ]:
        content = Template(
            open("support/supWSD/{}.tmpl".format(src_fn)).read()
        ).substitute(
            {
                "FIWN_PATH": fiwn_path,
                "WORK_DIR": work_dir,
                "VEC_PATH": vec_path,
                "USE_VEC": "true" if use_vec else "false",
                "USE_SURROUNDING_WORDS": "true" if use_surrounding_words else "false",
            }
        )

        with open("systems/supWSD/{}".format(dst_fn), "w") as dst_f:
            dst_f.write(content)


@supwsd.command()
@click.argument("inf")
@click.argument("keyin")
def train(inf, keyin):
    with local.cwd("systems/supWSD"):
        inf_path = pjoin("../../", inf)
        keyin_path = pjoin("../../", keyin)
        print(
            "java", "-jar", SUPWSD_JAR, "train", "supconfig.xml", inf_path, keyin_path
        )
        java("-jar", SUPWSD_JAR, "train", "supconfig.xml", inf_path, keyin_path)


@supwsd.command()
@click.argument("inf")
@click.argument("goldkey")
def test(inf, goldkey):
    with local.cwd("systems/supWSD"):
        inf_path = pjoin("../../", inf)
        goldkey_path = pjoin("../../", goldkey)
        java("-jar", SUPWSD_JAR, "test", "supconfig.xml", inf_path, goldkey_path)


if __name__ == "__main__":
    supwsd()
