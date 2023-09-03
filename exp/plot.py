import os

from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from networkx import draw_networkx, shell_layout

from exp.utility import ensure_dir


def plot_graph(v, c, a):
    """Plot a constraint-dependency graph."""
    gn = sorted(v.dep_graph.nodes)
    if len(gn) > 0:
        fn = os.path.join(c.out, f'__graph_{c.name}.pdf')
        lbl = ['immutable', 'mutable']
        colors = ['#CFD8DC', '#FFC107']
        color_map = [colors[0] if n in v.immutable
                     else colors[1] for n in gn]
        ensure_dir(fn)
        ax = plt.figure(1).add_subplot(1, 1, 1)
        draw_networkx(
            v.dep_graph.to_undirected(),
            pos=shell_layout(v.dep_graph),
            with_labels=True, node_color=color_map,
            font_size=8, font_weight='bold', ax=ax)
        legend1 = plt.legend(
            labels=[f'{k}: {a[k]}' for k in gn], loc='upper left',
            handles={Patch(fill=False, alpha=0) for _ in gn},
            bbox_to_anchor=(.92, 1.02), frameon=False)
        pairs = [(Patch(fill=True, color=colors[i]), lbl[i])
                 for i, _ in enumerate(colors)]
        legend2 = plt.legend(
            *zip(*pairs), frameon=False, borderpad=0,
            handlelength=1, handleheight=1,
            ncol=len(colors), loc='lower left',
            bbox_to_anchor=(0, -0.09))
        ax.add_artist(legend1)
        ax.add_artist(legend2)
        plt.savefig(fn, bbox_inches="tight")
