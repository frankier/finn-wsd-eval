from finntk.finnpos import sent_finnpos
from finntk.wordnet.reader import fiwn
from finntk.wsd.lesk_emb import mk_defn_vec_conceptnet_en
from finntk.wsd.lesk_pp import mk_lemma_vec, mk_context_vec
from finntk.emb.utils import cosine_sim
from stiff.utils.xml import iter_sentences
import click
from means import ALL_MEANS
from utils import lemmas_from_instance, write_lemma


@click.command('lesk-pp')
@click.argument("mean")
@click.argument("inf", type=click.File("rb"))
@click.argument("keyout", type=click.File("w"))
@click.option("--include-wfs/--no-include-wfs")
def lesk_pp(mean, inf, keyout, include_wfs):
    mean_func = ALL_MEANS[mean]
    for sent in iter_sentences(inf):
        if include_wfs:
            instances = sent.xpath("instance|wf")
            sent = [inst.text for inst in instances]
            tagged_sent = sent_finnpos(sent)
        else:
            instances = sent.xpath("instance")
            tagged_sent = None
        sent_lemmas = []
        instance_ids = []
        # XXX: SHOULD add wfs too! (equiv to wn_filter)
        for idx, instance in enumerate(instances):
            if instance.tag == 'wf':
                lemma_str = tagged_sent[idx][1]
                lemmas = []
            else:
                lemma_str, _pos, lemmas = lemmas_from_instance(fiwn, instance)
            sent_lemmas.append((lemma_str, lemmas))
            if instance.tag == 'instance':
                instance_ids.append(instance.attrib["id"])
        disambg_order = sorted((len(lemmas), idx) for idx, (lemma_str, lemmas) in enumerate(sent_lemmas) if len(lemmas) > 0)
        for ambiguity, lemma_idx in disambg_order:
            if ambiguity <= 1:
                continue
            lemma_str, lemmas = sent_lemmas[lemma_idx]

            # XXX: Should context_vec exclude the word being disambiguated
            context_vec = mk_context_vec(mean_func, sent_lemmas, "fi")
            if context_vec is None:
                # Back off to MFS
                sent_lemmas[lemma_idx] = (lemma_str, [lemmas[0]])
            else:
                best_lemma = None
                best_score = -2
                for lemma in lemmas:
                    defn_vec = mk_defn_vec_conceptnet_en(mean_func, lemma)
                    score = cosine_sim(defn_vec, context_vec)
                    try:
                        lemma_vec = mk_lemma_vec(lemma)
                    except KeyError:
                        # XXX: Is this reasonable, or should there be a penalty?
                        score = 2 * score
                    else:
                        score = score + cosine_sim(lemma_vec, context_vec)
                    if score > best_score:
                        best_lemma = lemma
                        best_score = score
                sent_lemmas[lemma_idx] = (lemma_str, [best_lemma])
        for (lemma_str, lemmas), inst_id in zip((x for x in sent_lemmas if len(x[1]) > 0), instance_ids):
            if lemmas[0] is None:
                continue
            write_lemma(keyout, inst_id, lemmas[0])


if __name__ == '__main__':
    lesk_pp()
