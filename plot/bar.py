import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import datetime as dt

col0 = [240 / 255, 249 / 255, 232 / 255, 1]
col1 = [186 / 255, 228 / 255, 188 / 255, 1]
col2 = [123 / 255, 204 / 255, 196 / 255, 1]
col3 = [43 / 255, 140 / 255, 190 / 255, 1]
col4 = [8 / 255, 104 / 255, 172 / 255, 1]
light_blue = [166 / 255, 206 / 255, 227 / 255, 1]
dark_blue = [15 / 255, 90 / 255, 160 / 255, 1]
black = [0.2, 0.2, 0.2, 1]

mid_blue = \
    [0.6 * light_blue[i] + 0.4 * dark_blue[i] for i in range(4)]
mid_light_blue = \
    [0.8 * light_blue[i] + 0.2 * dark_blue[i] for i in range(4)]
mid_dark_blue = \
    [0.3 * light_blue[i] + 0.7 * dark_blue[i] for i in range(4)]


def get_color_scheme(n):
    if n == 5:
        return [dark_blue, mid_dark_blue, mid_blue, mid_light_blue,
                light_blue]
    if n < 5:
        return [col4, col3, col2, col1][-n:]

    assert False


def multibar_graph(
        results, category_names, colors=None,
        size=(5.8, 3.1), legend=True, log=False
):
    if isinstance(results, dict):
        results = [r for r in results.items()]

    labels = [r[0] for r in results]
    data = np.array([r[1] for r in results])
    data_cum = data.cumsum(axis=1)
    if colors is None:
        colors = plt.get_cmap('RdYlGn')(
            np.linspace(1, 0.5, data.shape[1]))

    fig, ax = plt.subplots(figsize=size)
    ax.invert_yaxis()
    # ax.xaxis.set_visible(True)

    for i, (colname, color) in enumerate(zip(category_names, colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        heights = [0.8 if label == "overall" else 0.5 for label in
                   labels]
        rects = ax.barh(labels, widths, left=starts, height=heights,
                        label=colname, color=color)

        r, g, b, _ = color
        # text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        # ax.bar_label(rects, label_type='center', color=text_color)

    if "overall" in labels:
        for idx in range(len(labels)):
            if labels[idx] == "overall":
                ax.get_yticklabels()[idx].set_fontweight('bold')
            else:
                ax.get_yticklabels()[idx].set_fontweight('light')
                ax.get_yticklabels()[idx].set_size(8)

    if legend:
        ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
                  loc='lower left', fontsize='small', frameon=False,
                  handlelength=1)
        fig.tight_layout()

    if log:
        ax.set_xscale('log')
    else:
        ax.set_xlim(0, np.sum(data, axis=1).max())

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.yaxis.set_tick_params(length=0)
    fig.tight_layout()
    return fig, ax


def plot(dialect_data, mean_data, plot_name, limit=None,
         sort_key=None, colors=None, groups=None):
    if groups is not None:
        pass
    elif limit is None:
        groups = [str(i) for i in range(len(mean_data))]
    else:
        groups = [str(i) for i in range(limit)] + [f"{limit}+"]
        dialect_data = {name: r[:limit] + [sum(r[limit:])] for
                        name, r in dialect_data.items()}
        mean_data = mean_data[:limit] + [sum(mean_data[limit:])]
    dialect_results = [(name, r) for name, r in
                       dialect_data.items()]
    dialect_results = [(name, [v / sum(val) * 100 for v in val]) for
                       name, val in dialect_results]
    if sort_key is not None:
        dialect_results.sort(key=sort_key)
    dialect_results = dialect_results + [
        ("", [0] * len(mean_data))] + [("overall",
                                        [v / sum(mean_data) * 100
                                         for v in mean_data])]
    if colors is None:
        color_count = len(mean_data)
        colors = get_color_scheme(color_count)
        colors.reverse()

    fig, ax = multibar_graph(
        dialect_results, groups, size=(3, 4.5),
        colors=colors, legend=False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.yaxis.set_tick_params(length=0)
    ax.legend(ncol=4, bbox_to_anchor=(-0.1, 0.97),
              loc='lower left', fontsize='small',
              frameon=False, handlelength=1)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    fig.tight_layout()
    plt.savefig(plot_name, metadata={'CreationDate': None})
    plt.close()


if __name__ == "__main__":
    plot_dir_path = 'result'
    num_operands_per_dialect = {
        'builtin': [2, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        'pdl_interp': [6, 24, 3, 1, 0, 0, 0, 0, 0, 0],
        'pdl': [3, 8, 1, 2, 0, 0, 0, 0, 0, 0],
        'llvm': [14, 53, 49, 15, 4, 0, 0, 0, 0, 0],
        'rocdl': [13, 20, 0, 0, 0, 1, 1, 0, 0, 0],
        'nvvm': [15, 9, 1, 0, 1, 0, 0, 0, 0, 0],
        'tensor': [0, 4, 3, 1, 1, 1, 0, 0, 0, 0],
        'complex': [0, 8, 7, 0, 0, 0, 0, 0, 0, 0],
        'arm_sve': [2, 0, 12, 27, 0, 0, 0, 0, 0, 0],
        'quant': [0, 10, 0, 0, 0, 0, 0, 0, 0, 0],
        'x86vector': [0, 2, 4, 3, 0, 6, 0, 0, 0, 0],
        'async': [2, 17, 6, 0, 0, 0, 0, 0, 0, 0],
        'emitc': [2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
        'sparse_tensor': [0, 5, 2, 0, 0, 0, 0, 0, 0, 0],
        'memref': [3, 11, 8, 3, 2, 0, 0, 0, 0, 0],
        'tosa': [1, 36, 25, 8, 0, 0, 0, 0, 0, 0],
        'gpu': [12, 6, 2, 6, 0, 0, 0, 1, 0, 1],
        'linalg': [1, 4, 0, 2, 0, 1, 0, 0, 0, 0],
        'vector': [1, 11, 8, 6, 7, 2, 0, 0, 0, 0],
        'std': [1, 22, 31, 5, 0, 0, 0, 0, 0, 0],
        'affine': [0, 7, 3, 2, 0, 0, 0, 0, 0, 0],
        'arm_neon': [0, 0, 1, 2, 0, 0, 0, 0, 0, 0],
        'math': [0, 16, 3, 1, 0, 0, 0, 0, 0, 0],
        'spv': [21, 56, 86, 15, 3, 0, 0, 0, 0, 0],
        'shape': [4, 20, 11, 0, 0, 0, 0, 0, 0, 0],
        'scf': [1, 5, 1, 0, 2, 0, 0, 0, 0, 0],
        'arith': [1, 12, 22, 0, 0, 0, 0, 0, 0, 0],
        'amx': [1, 0, 2, 3, 1, 1, 5, 0, 0, 0]}
    num_operands_mean = [106, 369, 291, 102, 21, 12, 6, 1, 0, 1]

    # Number of operands per operation
    plot(num_operands_per_dialect, num_operands_mean,
         plot_dir_path + "/numbers.pdf", limit=3,
         sort_key=lambda x: (-x[1][3], -x[1][2], -x[1][1]))
