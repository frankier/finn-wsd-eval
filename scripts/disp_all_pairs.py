import click
import pickle
from networkx import DiGraph, Graph
from networkx.algorithms.clique import find_cliques


@click.group()
def disp_all_pairs():
    pass


def num2letter(num):
    res = ""
    num += 1
    while num:
        res = chr(97 + (num % 26)) + res
        num = num // 26
    return res


def iter_all_pairs_cmp(pvalmat):
    """
    Unpack the triangular matrix structure of an p values from an all pairs
    comparison.
    """
    for idx_a, row in enumerate(pvalmat):
        for idx_b_adj, (b_bigger, p_val) in enumerate(row):
            idx_b = idx_a + idx_b_adj + 1
            yield idx_a, idx_b, b_bigger, p_val


def mk_sd_graph(pvalmat, thresh=0.05):
    """
    Make a graph with edges as signifcant differences between treatments.
    """
    digraph = DiGraph()
    for idx_a, idx_b, b_bigger, p_val in iter_all_pairs_cmp(pvalmat):
        if p_val > thresh:
            continue
        if b_bigger:
            digraph.add_edge(idx_a, idx_b)
        else:
            digraph.add_edge(idx_b, idx_a)
    return digraph


def mk_nsd_graph(pvalmat, thresh=0.05):
    """
    Make a graph with edges as non signifcant differences between treatments.
    """
    graph = Graph()
    for idx_a, idx_b, b_bigger, p_val in iter_all_pairs_cmp(pvalmat):
        if p_val <= thresh:
            continue
        graph.add_edge(idx_a, idx_b)
    return graph


@disp_all_pairs.command("hasse")
@click.argument("inf", type=click.File("rb"))
@click.option("--thresh", type=float, default=0.05)
def hasse(inf, thresh):
    """
    Draw a hasse diagram showing which treatments/expcombs have significantly
    differences from each other.
    """
    from networkx.algorithms.dag import transitive_reduction
    from networkx.drawing.nx_pylab import draw
    from networkx.drawing.nx_agraph import graphviz_layout
    import matplotlib.pyplot as plt

    pvalmat, guesses = pickle.load(inf)
    digraph = mk_sd_graph(pvalmat, thresh)
    digraph = transitive_reduction(digraph)
    layout = graphviz_layout(digraph, prog="dot")
    draw(digraph, pos=layout)
    plt.show()


@disp_all_pairs.command("cld")
@click.argument("inf", type=click.File("rb"))
@click.argument("outf", type=click.File("wb"))
@click.option("--thresh", type=float, default=0.05)
def cld(inf, outf, thresh):
    """
    Create a Compact Letter Display (CLD) grouping together treatments/expcombs
    which have no significant difference. See:

    Hans-Peter Piepho (2004) An Algorithm for a Letter-Based Representation of
    All-Pairwise Comparisons, Journal of Computational and Graphical
    Statistics, 13:2, 456-466, DOI: 10.1198/1061860043515

    https://www.tandfonline.com/doi/abs/10.1198/1061860043515

    Gramm et al. (2006) Algorithms for Compact Letter Displays: Comparison and
    Evaluation Jens Gramm

    http://www.akt.tu-berlin.de/fileadmin/fg34/publications-akt/letter-displays-csda06.pdf
    """
    pvalmat, guesses = pickle.load(inf)

    graph = mk_nsd_graph(pvalmat, thresh)
    res = {}

    for clique_idx, clique in enumerate(find_cliques(graph)):
        for elem in clique:
            res.setdefault(elem, []).append(num2letter(clique_idx))

    print("\n".join(f"{elem}: {letters}" for elem, letters in sorted(res.items())))
    pickle.dump(res, outf)


if __name__ == "__main__":
    disp_all_pairs()
