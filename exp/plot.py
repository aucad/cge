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
        lbl = ['any-value', 'immutable', 'single-feat', 'multi-feat']
        colors = ['#00E676', '#CFD8DC', '#00BCD4', '#FFC107']
        color_map = [
            colors[1] if n in v.immutable else
            colors[0] if n not in v.constraints.keys() else
            colors[2] if n in v.single_feat.keys() else
            colors[3] for n in gn]
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
        leg_colors = [i for i in sorted(list(set(
            [colors.index(c) for c in color_map])))]
        pairs = [(Patch(fill=True, color=colors[i]), lbl[i])
                 for i, _ in enumerate(colors) if i in leg_colors]
        legend2 = plt.legend(
            *zip(*pairs), frameon=False, borderpad=0,
            handlelength=1, handleheight=1,
            ncol=len(colors), loc='lower left',
            bbox_to_anchor=(0, -0.09))
        ax.add_artist(legend1)
        ax.add_artist(legend2)
        plt.savefig(fn, bbox_inches="tight")
