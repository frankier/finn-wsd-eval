from expcomb.table import SumTableSpec, CatValGroup, UnlabelledMeasure

TABLES = [
    (
        "full_sum_table",
        SumTableSpec(
            [
                CatValGroup("train-corpus", ["eurosense-train", "stiff-train"]),
                CatValGroup("test-corpus", ["eurosense-test", "stiff-test"]),
            ],
            UnlabelledMeasure("F1"),
        ),
    )
]
