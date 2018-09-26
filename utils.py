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
    chosen_synset_fi_id = ss2pre(lemma.synset())
    if chosen_synset_fi_id not in fi2en:
        sys.stderr.write(
            "No fi2en mapping found for {} ({})\n".format(chosen_synset_fi_id, lemma)
        )
        return
    keyout.write("{} {}\n".format(inst_id, pre_id_to_post(fi2en[chosen_synset_fi_id])))


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
