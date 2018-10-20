from finntk.finnpos import sent_finnpos
from finntk.wordnet.reader import fiwn
from finntk.emb.numberbatch import multispace as numberbatch_multispace
from finntk.emb.autoextend import mk_lemma_vec
from finntk.wsd.lesk_pp import LeskPP
from finntk.emb.utils import cosine_sim
from stiff.utils.xml import iter_sentences
import click
from means import ALL_MEANS
from utils import lemmas_from_instance, write_lemma


@click.command("lesk-pp")
@click.argument("mean")
@click.argument("inf", type=click.File("rb"))
@click.argument("keyout", type=click.File("w"))
@click.option("--include-wfs/--no-include-wfs")
@click.option("--expand/--no-expand")
@click.option("--exclude-cand/--no-exclude-cand")
@click.option("--score-by")
def lesk_pp(mean, inf, keyout, include_wfs, expand, exclude_cand, score_by):
    aggf = ALL_MEANS[mean]
    lesk_pp = LeskPP(numberbatch_multispace, aggf, False, expand)
    for sent_idx, sent in enumerate(iter_sentences(inf)):
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
            if instance.tag == "wf":
                lemma_str = tagged_sent[idx][1]
                lemmas = []
            else:
                lemma_str, _pos, lemmas = lemmas_from_instance(fiwn, instance)
            sent_lemmas.append((lemma_str, lemmas))
            if instance.tag == "instance":
                instance_ids.append(instance.attrib["id"])
        disambg_order = sorted(
            (len(lemmas), idx)
            for idx, (lemma_str, lemmas) in enumerate(sent_lemmas)
            if len(lemmas) > 0
        )
        for ambiguity, lemma_idx in disambg_order:
            if ambiguity <= 1:
                continue
            lemma_str, lemmas = sent_lemmas[lemma_idx]

            # XXX: Should context_vec exclude the word being disambiguated
            context_vec = lesk_pp.mk_ctx_vec(
                sent_lemmas, *([lemma_idx] if exclude_cand else [])
            )
            if context_vec is None:
                # Back off to MFS
                sent_lemmas[lemma_idx] = (lemma_str, [lemmas[0]])
            else:
                best_lemma = None
                best_score = -2
                for lemma in lemmas:
                    defn_vec = lesk_pp.mk_defn_vec(lemma)
                    defn_ctx_score = cosine_sim(defn_vec, context_vec)
                    try:
                        lemma_vec = mk_lemma_vec(lemma)
                    except KeyError:
                        # XXX: Is this reasonable, or should there be a penalty?
                        lemma_ctx_score = defn_ctx_score
                    else:
                        lemma_ctx_score = cosine_sim(lemma_vec, context_vec)
                    if score_by == "both":
                        score = defn_ctx_score + lemma_ctx_score
                    elif score_by == "defn":
                        score = defn_ctx_score
                    elif score_by == "lemma":
                        score = lemma_ctx_score
                    else:
                        assert False
                    if score > best_score:
                        best_lemma = lemma
                        best_score = score
                sent_lemmas[lemma_idx] = (lemma_str, [best_lemma])
        for (lemma_str, lemmas), inst_id in zip(
            (x for x in sent_lemmas if len(x[1]) > 0), instance_ids
        ):
            if lemmas[0] is None:
                continue
            write_lemma(keyout, inst_id, lemmas[0])


if __name__ == "__main__":
    lesk_pp()
