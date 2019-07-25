import click
from plumbum import local
from plumbum.cmd import java
from string import Template
from os.path import join as pjoin


SUPWSD_JAR = "target/supwsd-toolkit-1.0.0.jar"


@click.group()
def supwsd():
    pass


@supwsd.command()
@click.argument("work_dir")
@click.argument("vec_path", required=False)
@click.option("--use-vec/--no-use-vec")
@click.option("--use-surrounding-words/--no-use-surrounding-words")
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
def test(inf, goldkey):
    with local.cwd("systems/supWSD"):
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
