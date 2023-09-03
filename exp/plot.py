import os

from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from networkx import draw_networkx, shell_layout

from exp.utility import ensure_dir


def plot_graph(v, c, a):
    """Plot a constraint-dependency graph."""
    gn = sorted(v.graph.nodes)
    if len(gn) > 0:
        fn = os.path.join(c.out, f'__graph_{c.name}.pdf')
        lbl, clr = ['immutable', 'mutable'], ['#CFD8DC', '#FDD835']
        nc = [clr[0] if n in v.immutable else clr[1] for n in gn]
        ax = plt.figure(1).add_subplot(1, 1, 1)
        draw_networkx(
            v.graph, ax=ax, pos=shell_layout(v.graph),
            with_labels=True, node_color=nc, linewidths=.75,
            width=.75, font_size=8, font_weight='bold')
        legend1 = plt.legend(
            labels=[f'{k}: {a[k]}' for k in gn], loc='upper left',
            handles={Patch(fill=False, alpha=0) for _ in gn},
            bbox_to_anchor=(.92, 1.02), frameon=False)
        pairs = [(Patch(fill=True, color=clr[i]), lbl[i])
                 for i, _ in enumerate(clr)]
        legend2 = plt.legend(
            *zip(*pairs), frameon=False, borderpad=0,
            handlelength=1, handleheight=1, ncol=2,
            loc='lower left', bbox_to_anchor=(0, -0.09))
        ax.add_artist(legend1)
        ax.add_artist(legend2)
        ensure_dir(fn)
        plt.savefig(fn, bbox_inches="tight")
