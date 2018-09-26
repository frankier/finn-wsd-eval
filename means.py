from finntk.emb.utils import (
    CATP_3, CATP_4, catp_mean, sif_mean, unnormalized_mean, normalized_mean
)

from functools import partial


MEANS = {
    "catp3_mean": partial(catp_mean, ps=CATP_3),
    "catp4_mean": partial(catp_mean, ps=CATP_4),
    "sif_mean": sif_mean,
    "unnormalized_mean": unnormalized_mean,
    "normalized_mean": normalized_mean,
}
