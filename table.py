import click
from tinydb import TinyDB


def pk(doc):
    pk_doc = doc.copy()
    for attr in ["nick", "disp", "time", "F1", "P", "R"]:
        if attr in pk_doc:
            del pk_doc[attr]
    if "opts" in pk_doc:
        pk_doc.update(pk_doc["opts"])
        del pk_doc["opts"]
    return tuple(sorted(pk_doc.items()))


def all(dbs):
    for db in dbs:
        for doc in db.all():
            yield doc


def all_recent(dbs):
    recents = {}
    for doc in all(dbs):
        if "time" not in doc:
            continue
        key = pk(doc)
        if key not in recents or doc["time"] > recents[key]["time"]:
            recents[key] = doc
    return recents.values()


def get_values(docs, attr):
    vals = set()
    for doc in docs:
        vals.add(doc[attr])
    return sorted(vals)


def get_attr_combs(docs, attrs):
    if len(attrs) == 0:
        return [[]]
    (head_attr, head_vals), tail = attrs[0], attrs[1:]
    return [
        [(head_attr, val)] + comb
        for val in head_vals
        for comb in get_attr_combs(docs, tail)
    ]


def str_of_comb(comb):
    return ", ".join("{}={}".format(*pair) for pair in comb)


def get_doc(docs, opts):
    found = []
    for doc in docs:
        equal = True
        for k, v in opts.items():
            if doc[k] != v:
                equal = False
                break
        if equal:
            found.append(doc)
    if len(found):
        assert len(found) == 1
        return found[0]


def get_attr_value_pairs(spec, docs):
    pairs = []
    bits = spec.split(";")
    for bit in bits:
        av_bits = bit.split(":")
        if len(av_bits) == 1:
            attr = av_bits[0]
            vals = get_values(docs, attr)
        elif len(av_bits) == 2:
            attr = av_bits[0]
            vals = av_bits[1].split(",")
        else:
            assert False
        pairs.append((attr, vals))
    return pairs


@click.command()
@click.argument("db_paths", type=click.Path(), nargs=-1)
@click.option("--filter", nargs=1, required=False)
@click.option("--table", nargs=1, required=False)
@click.option("--header/--no-header", default=True)
def main(db_paths, filter, table, header):
    from eval_table import parse_opts

    dbs = []
    for db_path in db_paths:
        dbs.append(TinyDB(db_path).table("results"))
    docs = all_recent(dbs)
    if filter:
        bits = filter.split(";")
        filter_l1 = bits[0] if len(bits) >= 1 else None
        filter_l2 = bits[1] if len(bits) >= 2 else None
        opt_dict = parse_opts(bits[2:])
        docs = [
            doc
            for doc in docs
            if (filter_l1 is None or doc["category"] == filter_l1)
            and (filter_l2 is None or doc["subcat"] == filter_l2)
            and (all((doc[opt] == opt_dict[opt] for opt in opt_dict)))
        ]
    if table:
        x_groups, y_groups = table.split()
        x_bits = get_attr_value_pairs(x_groups, docs)
        x_combs = get_attr_combs(docs, x_bits)
        y_bits = get_attr_value_pairs(y_groups, docs)
        y_combs = get_attr_combs(docs, y_bits)
        if header:
            print(" & ", end="")
            print(
                " & ".join((str_of_comb(y_comb) for y_comb in y_combs)), end=" \\\\\n"
            )
        for x_comb in x_combs:
            if header:
                print(str_of_comb(x_comb) + " & ", end="")
            f1s = []
            for y_comb in y_combs:
                opts = dict(x_comb + y_comb)
                picked_doc = get_doc(docs, opts)
                if picked_doc:
                    f1s.append(picked_doc["F1"].replace("%", "\\%"))
                else:
                    f1s.append("---")
            print(" & ".join(f1s), end=" \\\\\n")
    else:
        for doc in docs:
            print(doc)


if __name__ == "__main__":
    main()
