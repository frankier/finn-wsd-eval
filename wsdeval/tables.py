from expcomb.table import SumTableSpec, CatValGroup, MeasuresSplit, LookupGroupDisplay

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
    )
]
