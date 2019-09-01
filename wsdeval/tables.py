from expcomb.table.spec import (
    SqTableSpec,
    SumTableSpec,
    CatValGroup,
    LookupGroupDisplay,
    UnlabelledMeasure,
)
from expcomb.filter import SimpleFilter


def tf_group(name):
    return CatValGroup(name, [False, True])


VECS = LookupGroupDisplay(
    CatValGroup(
        "opts,vec", ["fasttext", "numberbatch", "word2vec", "double", "triple"]
    ),
    {
        "fasttext": "fastText",
        "numberbatch": "Numberbatch",
        "word2vec": "Word2Vec",
        "double": "Concat 2",
        "triple": "Concat 3",
    },
)
MEANS = LookupGroupDisplay(
    CatValGroup(
        "opts,mean",
        [
            "pre_sif_mean",
            # "sif_mean",
            "normalized_mean",
            # "unnormalized_mean",
            "catp3_mean",
            "catp4_mean",
        ],
    ),
    {
        "normalized_mean": "AWE",
        "catp3_mean": "CATP3",
        "catp4_mean": "CATP4",
        "pre_sif_mean": "pre-SIF",
    },
)

TRAIN_CORPORA = LookupGroupDisplay(
    CatValGroup("train-corpus", ["eurosense-train", "stiff-train"]),
    {"eurosense-train": "Eurosense", "stiff-train": "STIFF"},
)

TEST_CORPORA = LookupGroupDisplay(
    CatValGroup("test-corpus", ["eurosense-test", "stiff-test"]),
    {"eurosense-test": "Eurosense", "stiff-test": "STIFF"},
)

DEV_CORPORA = LookupGroupDisplay(
    CatValGroup("test-corpus", ["eurosense-dev", "stiff-dev"]),
    {"eurosense-dev": "Eurosense", "stiff-dev": "STIFF"},
)


TABLES = [
    (
        "full_sum_table",
        SumTableSpec([TEST_CORPORA, TRAIN_CORPORA], UnlabelledMeasure("F1")),
    ),
    (
        "lesk_square",
        SqTableSpec(
            [
                LookupGroupDisplay(
                    tf_group("opts,use_freq"), {False: "No freq", True: "Freq"}
                ),
                VECS,
                MEANS,
            ],
            [
                LookupGroupDisplay(
                    tf_group("opts,expand"), {False: "No expand", True: "Expand"}
                ),
                LookupGroupDisplay(
                    tf_group("opts,wn_filter"), {False: "No filter", "True": "Filter"}
                ),
            ],
            UnlabelledMeasure("F1"),
        ),
        SimpleFilter("Knowledge", "Cross-lingual Lesk", **{"test-corpus": "stiff-dev"}),
    ),
    (
        "nn_awe_square",
        SqTableSpec(
            [TRAIN_CORPORA, MEANS], [DEV_CORPORA, VECS], UnlabelledMeasure("F1")
        ),
        SimpleFilter("Supervised", "AWE-NN"),
    ),
]
