import os
from functools import partial
from itertools import groupby


def get_batch_size():
    return int(os.environ.get("BATCH_SIZE", "32"))


class CtxEmbedder:
    def iter_inst_vecs(self, inf, batch_size=None, **kwargs):
        from wsdeval.formats.sup_corpus import iter_instances

        if batch_size is None:
            batch_size = get_batch_size()

        model = self.vecs.get()
        iter = iter_instances(inf)
        while 1:
            infos = []
            sents = []
            is_end = True
            for idx, (inst_id, item_pos, (be, he, af)) in enumerate(iter):
                sents.append(be + he + af)
                start_idx = len(be)
                end_idx = len(be) + len(he)
                infos.append((inst_id, item_pos, start_idx, end_idx))
                if (idx + 1) == batch_size:
                    is_end = False
                    break
            embs = self.embed_sentences(model, sents, **kwargs)
            for (inst_id, item_pos, start_idx, end_idx), sent, emb in zip(
                infos, sents, embs
            ):
                yield inst_id, item_pos, self.proc_vec(
                    emb, sent, start_idx, end_idx, **kwargs
                )
            if is_end:
                break

    def iter_inst_vecs_grouped(self, inf, batch_size=None, **kwargs):
        ungrouped = self.iter_inst_vecs(inf, batch_size, **kwargs)
        for item_pos, group_iter in groupby(ungrouped, lambda tpl: tpl[1]):
            group_list = list(group_iter)
            yield (
                ".".join(item_pos),
                len(group_list),
                ((inst_id, vec) for inst_id, item_pos, vec in group_list),
            )


class ElmoEmbedder(CtxEmbedder):
    from finntk.emb.elmo import vecs
    from finntk.vendor.elmo import embed_sentences

    vecs = staticmethod(vecs)
    embed_sentences = staticmethod(embed_sentences)

    def proc_vec(self, emb, sent, start_idx, end_idx, **kwargs):
        output_layer = kwargs.get("output_layer", -1)
        if output_layer == -2:
            vec = emb[:, start_idx:end_idx]
        else:
            vec = emb[start_idx:end_idx]
        vec.shape = vec.shape[:-2] + (vec.shape[-2] * vec.shape[-1],)
        return vec


class BertEmbedder(CtxEmbedder):
    from finntk.emb.bert import vecs
    from finntk.vendor.bert import embed_sentences

    vecs = staticmethod(vecs)
    embed_sentences = staticmethod(partial(embed_sentences, tokenize=False))

    def proc_vec(self, emb_tokens, sent, start_idx, end_idx):
        emb, tokens = emb_tokens
        tok_start_idxs = []
        for idx, tok in enumerate(tokens):
            # Skip when start, end sentinal or when BPE split continuation
            if idx in (0, len(tokens) - 1) or tok.startswith("##"):
                continue
            tok_start_idxs.append(idx)
        if len(tok_start_idxs) < end_idx:
            return None
        tok_start_idxs.append(len(tokens) - 1)
        vec = emb[:, tok_start_idxs[start_idx] : tok_start_idxs[end_idx], :]
        vec = vec.reshape((vec.shape[0] * vec.shape[1], vec.shape[2]))
        return vec


elmo_embedder = ElmoEmbedder()
bert_embedder = BertEmbedder()
