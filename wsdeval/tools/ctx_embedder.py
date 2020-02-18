import torch
import numpy
import os
from functools import partial
from itertools import groupby
from stiff.sup_corpus import iter_instances
from wsdeval.nn.vec_nn_common import normalize
from itertools import islice


def get_batch_size():
    return int(os.environ.get("BATCH_SIZE", "32"))


def memory_debug(func):
    try:
        return func()
    except MemoryError:
        # Try once more
        import gc

        gc.collect()
        try:
            return func()
        except MemoryError:
            # Output some debugging info
            print("Whoops! Out of memory!")
            from pympler import muppy, summary

            all_objects = muppy.get_objects()
            summary = summary.summarize(all_objects)
            summary.print_(summary)
            for ao in all_objects:
                if not isinstance(ao, numpy.ndarry):
                    continue
                print(ao.dtype, ao.shape)
            raise


def make_batch(iter, batch_size):
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

    return infos, sents, is_end


class CtxEmbedder:
    def iter_inst_vecs(self, inf, batch_size=None, synsets=False, **kwargs):
        if batch_size is None:
            batch_size = get_batch_size()

        model = self.vecs.get()
        iter = iter_instances(inf, synsets=synsets)
        while 1:
            infos, sents, is_end = make_batch(iter, batch_size)
            embs = memory_debug(lambda: self.embed_sentences(model, sents, **kwargs))
            for (inst_id, item_pos, start_idx, end_idx), sent, emb in zip(
                infos, sents, embs
            ):
                yield inst_id, item_pos, self.proc_vec(
                    emb, sent, start_idx, end_idx, **kwargs
                )
            if is_end:
                break

    def iter_inst_vecs_grouped(self, inf, synsets=False, **kwargs):
        ungrouped = self.iter_inst_vecs(inf, synsets=synsets, **kwargs)
        for item_pos, group_iter in groupby(ungrouped, lambda tpl: tpl[1]):
            group_list = list(group_iter)
            yield (
                ".".join(item_pos) if not synsets else item_pos,
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


def tok_start_idxs(tokens, startend=True):
    for idx, tok in enumerate(tokens):
        # Skip when start, end sentinal or when BPE split continuation
        if startend and idx in (0, len(tokens) - 1):
            continue
        if tok.startswith("##"):
            continue
        yield idx + (0 if startend else 1)
    yield len(tokens) - 1


class BertEmbedder(CtxEmbedder):
    from finntk.emb.bert import vecs
    from finntk.vendor.bert import embed_sentences

    vecs = staticmethod(vecs)
    embed_sentences = staticmethod(partial(embed_sentences, tokenize=False))

    def proc_vec(self, emb_tokens, sent, start_idx, end_idx):
        emb, tokens = emb_tokens
        start_idxs = list(tok_start_idxs(tokens))
        if len(start_idxs) - 1 < end_idx:
            return None
        vec = emb[:, start_idxs[start_idx] : start_idxs[end_idx], :]
        vec = vec.reshape((vec.shape[0] * vec.shape[1], vec.shape[2]))
        return vec


class Bert2Embedder(CtxEmbedder):
    TENSOR_LENGTH = 256
    _tokenizer = None
    _model = None

    def get_bert2_models(cls):
        from transformers import BertTokenizer, BertModel

        if cls._tokenizer is None:
            cls._tokenizer = BertTokenizer.from_pretrained(
                "bert-base-multilingual-cased"
            )
            cls._tokenizer.do_basic_tokenize = False
            cls._model = BertModel.from_pretrained("bert-base-multilingual-cased")
            device = torch.device(
                "cuda"
                if torch.cuda.is_available() and not os.environ.get("NO_GPU")
                else "cpu"
            )
            cls._model.to(device)
            cls._model.device = device
            cls._model.eval()

        return cls._tokenizer, cls._model

    def iter_inst_vecs(self, inf, batch_size=None, synsets=False, **kwargs):

        if batch_size is None:
            batch_size = get_batch_size()

        tokenizer, model = self.get_bert2_models()
        iter = iter_instances(inf, synsets=synsets)
        while 1:
            infos, sents, is_end = make_batch(iter, batch_size)

            all_indexed_tokens = []
            all_segment_ids = []
            vec_idxs = []

            for sent, (_inst_id, _item_pos, start_idx, _end_idx) in zip(sents, infos):
                tokenized_text = tokenizer._tokenize(" ".join(sent))
                try:
                    vec_idx = next(
                        islice(
                            tok_start_idxs(tokenized_text, startend=False),
                            start_idx,
                            start_idx + 1,
                        )
                    )
                except StopIteration:
                    print(tokenized_text)
                    print(list(tok_start_idxs(tokenized_text, startend=False)))
                    print(start_idx)
                    raise
                else:
                    vec_idxs.append(vec_idx if vec_idx < self.TENSOR_LENGTH else None)
                indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
                indexed_tokens = tokenizer.prepare_for_model(
                    indexed_tokens,
                    max_length=self.TENSOR_LENGTH,
                    add_special_tokens=True,
                )["input_ids"]

                indexed_tokens = indexed_tokens + (
                    [0] * (self.TENSOR_LENGTH - len(indexed_tokens))
                )
                segment_ids = [0] * self.TENSOR_LENGTH
                all_indexed_tokens.append(indexed_tokens)
                all_segment_ids.append(segment_ids)

            def embed():

                tokens_tensor = torch.tensor(all_indexed_tokens, device=model.device)
                segments_tensors = torch.tensor(all_segment_ids, device=model.device)

                vecs = []
                with torch.no_grad():
                    outputs = model(tokens_tensor, token_type_ids=segments_tensors)
                    for sent_idx, vec_idx in enumerate(vec_idxs):
                        if vec_idx is None:
                            vecs.append(None)
                        else:
                            vecs.append(
                                normalize(
                                    outputs[0][sent_idx, vec_idx, :].cpu().numpy()
                                )
                            )

                return vecs

            embs = memory_debug(embed)
            for (inst_id, item_pos, _, _), emb in zip(infos, embs):
                yield inst_id, item_pos, emb
            if is_end:
                break


class Ctx2Vec2Embedder(CtxEmbedder):
    def get_ctx2vec_models(cls):
        from os.path import dirname, join as pjoin
        from context2vec.common.model_reader import ModelReader

        return ModelReader(
            pjoin(
                dirname(__file__), "..", "..", "systems", "context2vec", "model.params"
            )
        ).model

    def iter_inst_vecs(self, inf, synsets=False, **kwargs):
        model = self.get_ctx2vec_models()
        iter = iter_instances(inf, synsets=synsets)
        for inst_id, item_pos, (be, he, af) in iter:
            vec = model.context2vec(be + he + af, len(be))
            yield inst_id, item_pos, vec


elmo_embedder = ElmoEmbedder()
bert_embedder = BertEmbedder()
bert2_embedder = Bert2Embedder()
ctx2vec2_embedder = Ctx2Vec2Embedder()
