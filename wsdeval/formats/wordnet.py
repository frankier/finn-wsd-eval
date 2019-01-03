import sys
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
