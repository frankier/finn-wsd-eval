from expcomb.filter import AndFilter, SimpleFilter


def corpus(corpusseg):
    return corpusseg.split("-", 1)[0]


def seg(corpusseg):
    return corpusseg.split("-", 1)[1]


class TrainSegFilter:
    def doc_included(self, d_path, d_opts):
        return seg(d_opts["train-corpus"]) in ("train", "trainf")


class InCorpusFilter:
    def doc_included(self, d_path, d_opts):
        res = corpus(d_opts["train-corpus"]) == corpus(d_opts["test-corpus"])
        return res


class OutCorpusFilter:
    def doc_included(self, d_path, d_opts):
        res = corpus(d_opts["train-corpus"]) != corpus(d_opts["test-corpus"])
        return res


UNSUP_SELECTED = [
    ("1st", SimpleFilter("Baseline", "1st")),
    ("Rand", SimpleFilter("Baseline", "Rand")),
    (
        "UKB",
        SimpleFilter("Knowledge", "UKB", **{"extract_extra": False, "use_freq": False}),
    ),
    (
        "UKB freq",
        SimpleFilter("Knowledge", "UKB", **{"extract_extra": False, "use_freq": True}),
    ),
    (
        "Lesk",
        SimpleFilter(
            "Knowledge",
            "Cross-lingual Lesk",
            **{
                "use_freq": False,
                "vec": "double",
                "expand": True,
                "wn_filter": False,
                "mean": "catp3_mean",
            }
        ),
    ),
    (
        "Lesk freq",
        SimpleFilter(
            "Knowledge",
            "Cross-lingual Lesk",
            **{
                "use_freq": True,
                "vec": "double",
                "expand": False,
                "wn_filter": False,
                "mean": "normalized_mean",
            }
        ),
    ),
    (
        "Lesk++",
        SimpleFilter(
            "Knowledge",
            "Lesk++",
            **{
                "expand": True,
                "mean": "pre_sif_mean",
                "exclude_cand": False,
                "score_by": "defn",
            }
        ),
    ),
]

XLING_SELECTED = [
    (
        "xAWE-NN",
        SimpleFilter("Supervised", "XAWE-NN", mean="normalized_mean", vec="double"),
    ),
    ("xBERT-NN", SimpleFilter("Supervised", "XBERT2-NN")),
]

SUP_SELECTED = [
    (
        "SupWSD",
        SimpleFilter(
            "Supervised",
            "SupWSD",
            vec="fasttext",
            sur_words=True,
            **{"1stsensecomb": "x1st"}
        ),
    ),
    (
        "AWE-NN",
        SimpleFilter("Supervised", "AWE-NN", mean="normalized_mean", vec="double"),
    ),
    ("BERT-NN", SimpleFilter("Supervised", "BERT2-NN")),
    ("Ctx2Vec", SimpleFilter("Supervised", "Context2Vec2")),
]

INOUT = [("in", InCorpusFilter()), ("out", OutCorpusFilter())]

SUP_INOUT_SELECTED = [
    (label + " " + inout, AndFilter(filter, TrainSegFilter(), inout_filter))
    for label, filter in SUP_SELECTED
    for inout, inout_filter in INOUT
]

LIMITS_SELECTED = [
    ("Unambg", SimpleFilter("Limits", "Floor")),
    ("1st", SimpleFilter("Baseline", "1st")),
    ("Rand", SimpleFilter("Baseline", "Rand")),
]

for inout, inout_filter in INOUT:
    # LIMITS_SELECTED += [
    # (
    # "InstKnown " + inout,
    # AndFilter(SimpleFilter("Limits", "CeilInst"), TrainSegFilter(), inout_filter),
    # ),
    # (
    # "TokKnown " + inout,
    # AndFilter(SimpleFilter("Limits", "CeilTok"), TrainSegFilter(), inout_filter),
    # ),
    # ]
    LIMITS_SELECTED += [
        (
            "Known " + inout,
            AndFilter(
                SimpleFilter("Limits", "CeilTok"), TrainSegFilter(), inout_filter
            ),
        )
    ]
