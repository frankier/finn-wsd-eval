import os
import sys
import click
from plumbum import local
from plumbum.cmd import java, python, make, bash
from stiff.filter_utils import iter_sentences
from stiff.data import UNI_POS_WN_MAP

def get_ukb():
    ukb_path = local.env.get("UKB_PATH", "systems/ukb/src")
    return local[ukb_path + "/ukb_wsd"]


@click.group()
def ukb():
    pass


@ukb.command()
@click.argument("input_fn")
@click.argument("output_fn")
@click.argument("variant")
@click.argument("graph_fn")
@click.argument("dict_fn")
def run(input_fn, output_fn, variant, graph_fn, dict_fn):
    run_inner(input_fn, output_fn, variant.split(" "), graph_fn, dict_fn)


def run_inner(input_fn, output_fn, variant, graph_fn, dict_fn):
    ukb_wsd = get_ukb()
    os.makedirs("guess", exist_ok=True)
    args = variant + ("-D", dict_fn, "-K", graph_fn, "-")
    pred_pipeline = (
        python[__file__, "unified_to_ukb", input_fn, "-"]
        | ukb_wsd[args]
        | python[__file__, "clean_keyfile", "-", "-"]
        > output_fn
    )
    pred_pipeline(stderr=sys.stderr)


@ukb.command()
@click.argument("inf", type=click.File("rb"))
@click.argument("outf", type=click.File("w"))
def unified_to_ukb(inf, outf):
    for sent_elem in iter_sentences(inf):
        bits = []
        outf.write(sent_elem.attrib["id"])
        outf.write("\n")
        for instance in sent_elem.xpath("instance"):
            id = instance.attrib["id"]
            lemma = instance.attrib["lemma"]
            pos = UNI_POS_WN_MAP[instance.attrib["pos"]]
            bits.append(f"{lemma}#{pos}#{id}#1")
        outf.write(" ".join(bits))
        outf.write("\n")


@ukb.command()
@click.argument("keyin", type=click.File("r"))
@click.argument("keyout", type=click.File("w"))
def clean_keyfile(keyin, keyout):
    for line in keyin:
        bits = line.split()
        inst_id = bits[1]
        ids = bits[2:-2]
        keyout.write(inst_id)
        keyout.write(" ")
        keyout.write(" ".join(ids))
        keyout.write("\n")


@ukb.command()
def fetch():
    os.makedirs("systems", exist_ok=True)
    with local.cd("systems"):
        git("clone", "https://github.com/asoroa/ukb.git")
        with local.cd("ukb/src"):
            make()
    # Prepare
    with local.cd("ukb-eval"):
        bash("./prepare_wn30graph.sh")
    (python["mkwndict.py", "--en-synset-ids"] > "wndict.fi.txt")()


if __name__ == "__main__":
    ukb()
