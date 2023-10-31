from os import path

from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from networkx import draw_networkx, shell_layout,Graph, add_path

from exp import categorize
from exp.utility import ensure_dir


def plot_graph(c, a):
    """Plot a constraint-dependency graph."""
    immutable, mutable = categorize(c.constraints)
    g, dep_nodes = Graph(), [s for (s, _) in mutable.values()]
    nodes = immutable + [c for dn in dep_nodes for c in dn]
    g.add_nodes_from(list(set(nodes)))
    for ngr in dep_nodes:
        add_path(g, ngr)
    gn = sorted(g.nodes)
    if len(gn) > 0:
        fn = path.join(c.out, f'__graph_{c.name}.pdf')
        lbl, clr = ['immutable', 'mutable'], ['#CFD8DC', '#FDD835']
        nc = [clr[0] if n in immutable else clr[1] for n in gn]
        ax = plt.figure(1).add_subplot(1, 1, 1)
        draw_networkx(
            g, ax=ax, pos=shell_layout(g),
            with_labels=True, node_color=nc, linewidths=.75,
            width=.75, font_size=8, font_weight='bold')
        legend1 = plt.legend(
            labels=[f'{k:>2}: {a[k]}' for k in gn], loc='upper left',
            handles={Patch(fill=False, alpha=0) for _ in gn},
            ncol=(len(gn) // 20) + 1 if len(gn) > 25 else 1,
            bbox_to_anchor=(.93, 1.02), frameon=False)
        pairs = [(Patch(fill=True, color=clr[i]), lbl[i])
                 for i, _ in enumerate(clr)]
        legend2 = plt.legend(
            *zip(*pairs), frameon=False, borderpad=0,
            handlelength=1, handleheight=1, ncol=1,
            loc='lower left', prop={"size": 8})
        ax.add_artist(legend1)
        ax.add_artist(legend2)
        ensure_dir(fn)
        plt.savefig(fn, bbox_inches="tight")

