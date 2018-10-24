import re
import sys
from stiff.utils.xml import iter_sentences
from stiff.data.constants import UNI_POS_WN_MAP
from finntk.wordnet.reader import get_en_fi_maps
from finntk.wordnet.utils import pre_id_to_post, ss2pre


def lemmas_from_instance(wn, instance):
    word = instance.attrib["lemma"]
    pos = UNI_POS_WN_MAP[instance.attrib["pos"]]
    lemmas = wn.lemmas(word, pos=pos)
    return word, pos, lemmas


def write_lemma(keyout, inst_id, lemma):
    fi2en, en2fi = get_en_fi_maps()
    if lemma is None:
        guess = "U"
    else:
        chosen_synset_fi_id = ss2pre(lemma.synset())
        if chosen_synset_fi_id not in fi2en:
            sys.stderr.write(
                "No fi2en mapping found for {} ({})\n".format(
                    chosen_synset_fi_id, lemma
                )
            )
            guess = "U"
        else:
            guess = pre_id_to_post(fi2en[chosen_synset_fi_id])
    keyout.write("{} {}\n".format(inst_id, guess))


def unigram(inf, keyout, wn):
    for sent in iter_sentences(inf):
        for instance in sent.xpath("instance"):
            inst_id = instance.attrib["id"]
            word, pos, lemmas = lemmas_from_instance(wn, instance)
            if not len(lemmas):
                sys.stderr.write("No lemma found for {} {}\n".format(word, pos))
                continue
            lemma = lemmas[0]
            write_lemma(keyout, inst_id, lemma)


CHUNK_SIZE = 4096
TAG_REGEX = re.compile(
    br" ?<(?P<START_TAG>.)(?P<SYNSETS>[^>]*)>(?P<WF>.)</(?P<END_TAG>.)>"
)


def proc_match(match):
    groups = match.groupdict()
    assert groups["START_TAG"] == groups["END_TAG"]
    synsets = []
    synsets_str = groups["SYNSETS"].strip(b" ")
    if synsets_str:
        synsets_bits = synsets_str.split(b" ")
        for synset in synsets_bits:
            synset_id, weight_str = synset.split(b"|")
            weight = float(weight_str)
            synsets.append((synset_id, weight))
    return groups["START_TAG"], synsets, groups["WF"]


def iter_supwsd_result(fp):
    buffer = b""
    done = False
    while not done:
        chunk = fp.read(CHUNK_SIZE)
        if not chunk:
            done = True
        else:
            buffer += chunk
        while len(buffer) > 0:
            match = TAG_REGEX.match(buffer)
            if not match:
                break
            yield proc_match(match)
            end_pos = match.end(0)
            if end_pos != len(buffer):
                assert buffer[end_pos] in b" \n"
            buffer = buffer[end_pos + 1 :]
        if done:
            assert len(buffer) == 0


def proc_supwsd(goldkey, supwsd_result_fp, guess_fp):
    for gold_line, supwsd_result in zip(goldkey, iter_supwsd_result(supwsd_result_fp)):
        key = gold_line.split()[0]
        synsets = supwsd_result[1]
        if len(synsets):
            synset = synsets[0][0].decode("utf-8")
        else:
            synset = "U"
        guess_fp.write("{} {}\n".format(key, synset))
