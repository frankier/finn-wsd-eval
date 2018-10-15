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


@click.command()
@click.argument("db_paths", type=click.Path(), nargs=-1)
def main(db_paths):
    dbs = []
    for db_path in db_paths:
        dbs.append(TinyDB(db_path).table("results"))
    for doc in all_recent(dbs):
        print(doc)


if __name__ == "__main__":
    main()
