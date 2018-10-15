from finntk.emb.utils import (
    CATP_3,
    CATP_4,
    catp_mean,
    unnormalized_mean,
    normalized_mean,
)

from functools import partial

EXPANDING_MEANS = {
    "catp3_mean": partial(catp_mean, ps=CATP_3),
    "catp4_mean": partial(catp_mean, ps=CATP_4),
}

NON_EXPANDING_MEANS = {
    # "sif_mean": sif_mean,
    "unnormalized_mean": unnormalized_mean,
    "normalized_mean": normalized_mean,
}

ALL_MEANS = {**EXPANDING_MEANS, **NON_EXPANDING_MEANS}

MEAN_DISPS = {
    "catp3_mean": "CATP3-WE",
    "catp4_mean": "CATP4-WE",
    # "sif_mean": "SIF-WE",
    "unnormalized_mean": "AWE",
    "normalized_mean": "AWE-norm",
}
