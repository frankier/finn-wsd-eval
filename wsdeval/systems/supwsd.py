import click
from plumbum import local
from plumbum.cmd import java, ln
from string import Template
from os import makedirs
from os.path import abspath, exists, join as pjoin
from glob import glob


SUPWSD_JAR = "target/supwsd-toolkit-1.0.0.jar"


@click.group()
def supwsd():
    pass


@supwsd.command()
@click.argument("work_dir")
@click.argument("vec_path", required=False)
@click.argument("dest", required=False)
@click.option("--use-vec/--no-use-vec")
@click.option("--use-surrounding-words/--no-use-surrounding-words")
def conf(work_dir, vec_path="", dest=None, use_vec=False, use_surrounding_words=True):
    from finntk.wordnet.reader import fiwn_resman

    if dest is not None and not exists(dest):
        makedirs(dest, exist_ok=True)
        ln("-s", *glob(abspath("systems/supWSD") + "/*"), dest)
    else:
        dest = "systems/supWSD"

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

        with open("{}/{}".format(dest, dst_fn), "w") as dst_f:
            dst_f.write(content)


@supwsd.command()
@click.argument("inf")
@click.argument("keyin")
@click.argument("supwsd-dir", required=False)
def train(inf, keyin, supwsd_dir="systems/supWSD"):
    with local.cwd(supwsd_dir):
        inf_path = pjoin("../../", inf)
        keyin_path = pjoin("../../", keyin)
        print(
            "java",
            "-jar",
            "-Xmx512g",
            SUPWSD_JAR,
            "train",
            "supconfig.xml",
            inf_path,
            keyin_path,
        )
        java(
            "-jar",
            "-Xmx512g",
            SUPWSD_JAR,
            "train",
            "supconfig.xml",
            inf_path,
            keyin_path,
        )


@supwsd.command()
@click.argument("inf")
@click.argument("goldkey")
@click.argument("supwsd-dir", required=False)
def test(inf, goldkey, supwsd_dir="systems/supWSD"):
    with local.cwd(supwsd_dir):
        inf_path = pjoin("../../", inf)
        goldkey_path = pjoin("../../", goldkey)
        java(
            "-jar",
            "-Xmx512g",
            SUPWSD_JAR,
            "test",
            "supconfig.xml",
            inf_path,
            goldkey_path,
        )


if __name__ == "__main__":
    supwsd()
