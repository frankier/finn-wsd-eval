from expcomb.table.spec import (
    SqTableSpec,
    SumTableSpec,
    CatValGroup,
    MeasuresSplit,
    LookupGroupDisplay,
    UnlabelledMeasure,
)
from expcomb.filter import SimpleFilter


def tf_group(name):
    return CatValGroup(name, [False, True])


VECS = LookupGroupDisplay(
    CatValGroup("opts,vec", ["fasttext", "numberbatch", "word2vec", "double", "triple"])
)
MEANS = LookupGroupDisplay(
    CatValGroup(
        "opts,mean",
        [
            "pre_sif_mean",
            "sif_mean",
            "normalized_mean",
            "unnormalized_mean",
            "catp3_mean",
            "catp4_mean",
        ],
    )
)


TABLES = [
    (
        "full_sum_table",
        SumTableSpec(
            [
                LookupGroupDisplay(
                    CatValGroup("test-corpus", ["eurosense-test", "stiff-test"]),
                    {"eurosense-test": "Eurosense", "stiff-test": "STIFF"},
                ),
                LookupGroupDisplay(
                    CatValGroup("train-corpus", ["eurosense-train", "stiff-train"]),
                    {"eurosense-train": "Eurosense", "stiff-train": "STIFF"},
                ),
            ],
            MeasuresSplit(["P", "R", "F1"]),
        ),
    ),
    (
        "lesk_square",
        SqTableSpec(
            [
                LookupGroupDisplay(tf_group("opts,expand")),
                LookupGroupDisplay(tf_group("opts,wn_filter")),
            ],
            [LookupGroupDisplay(tf_group("opts,use_freq")), VECS, MEANS],
            UnlabelledMeasure("F1"),
        ),
        SimpleFilter("Knowledge", "Cross-lingual Lesk", **{"test-corpus": "stiff-dev"}),
    ),
    (
        "nn_awe_square",
        SqTableSpec([MEANS], [VECS], UnlabelledMeasure("F1")),
        SimpleFilter(
            "Supervised",
            "AWE-NN",
            **{"train-corpus": "eurosense-train", "test-corpus": "eurosense-dev"}
        ),
    ),
]
