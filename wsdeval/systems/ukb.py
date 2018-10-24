import os
import sys
import click
from plumbum import local
from plumbum.cmd import python
from stiff.utils.xml import iter_sentences
from stiff.data.constants import UNI_POS_WN_MAP


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
@click.option("--extract-extra/--no-extract_extra")
def run(input_fn, output_fn, variant, graph_fn, dict_fn, extract_extra):
    run_inner(input_fn, output_fn, variant.split(" "), graph_fn, dict_fn, extract_extra)


def run_inner(input_fn, output_fn, variant, graph_fn, dict_fn, extract_extra):
    ukb_wsd = get_ukb()
    os.makedirs("guess", exist_ok=True)
    args = variant + (
        "-D",
        dict_fn,
        "-K",
        graph_fn,
        "-",
        "--smooth_dict_weight",
        "1000",
    )
    preproc_args = (__file__, "unified-to-ukb", input_fn, "-")
    if extract_extra:
        preproc_args += ("--extract-extra",)
    pred_pipeline = (
        python[preproc_args]
        | ukb_wsd[args]
        | python[__file__, "clean-keyfile", "-", "-"]
        > output_fn
    )
    pred_pipeline(stderr=sys.stderr)


def fake_starts(toks):
    idx = 0
    for tok in toks:
        yield idx
        idx += len(tok) + 1


@ukb.command()
@click.argument("inf", type=click.File("rb"))
@click.argument("outf", type=click.File("w"))
@click.option("--extract-extra/--no-extract_extra")
def unified_to_ukb(inf, outf, extract_extra):
    from stiff.extract.fin import FinExtractor

    if extract_extra:
        extractor = FinExtractor()
    for sent_elem in iter_sentences(inf):
        bits = []
        outf.write(sent_elem.attrib["id"])
        outf.write("\n")
        for instance in sent_elem.xpath("instance"):
            id = instance.attrib["id"]
            lemma = instance.attrib["lemma"].lower()
            pos = UNI_POS_WN_MAP[instance.attrib["pos"]]
            bits.append(f"{lemma}#{pos}#{id}#1")
        if extract_extra:
            elems = sent_elem.xpath("wf|instance")
            toks = [node.text for node in elems]
            known_idxs = {
                idx for idx, elem in enumerate(elems) if elem.tag == "instance"
            }
            tagging = extractor.extract_toks(toks, list(fake_starts(toks)))
            for tok_idx, tok in enumerate(tagging.tokens):
                if tok_idx in known_idxs:
                    continue
                extra_id = 0
                lemma_poses = set()
                for tag in tok.tags:
                    for wn, lemma_obj in tag.lemma_objs:
                        lemma_name = lemma_obj.name().lower().strip()
                        if lemma_name == "":
                            continue
                        lemma_pos = lemma_obj.synset().pos()
                        if lemma_pos == "s":
                            lemma_pos = "a"
                        lemma_poses.add((lemma_name, lemma_pos))
                for lemma, pos in lemma_poses:
                    bits.append(f"{lemma}#{pos}#xT{tok_idx}N{extra_id}#0")
                    extra_id += 1
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


if __name__ == "__main__":
    ukb()
