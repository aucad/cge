from itertools import groupby

import matplotlib.pyplot as plt
import numpy as np

from exp import ensure_dir
from plot import ResultData

col0 = [240 / 255, 249 / 255, 232 / 255, 1]
col1 = [186 / 255, 228 / 255, 188 / 255, 1]
col2 = [123 / 255, 204 / 255, 196 / 255, 1]
col3 = [43 / 255, 140 / 255, 190 / 255, 1]
col4 = [220 / 255, 220 / 255, 220 / 255, 1]
col5 = [8 / 255, 104 / 255, 172 / 255, 1]
light_blue = [166 / 255, 206 / 255, 227 / 255, 1]
dark_blue = [15 / 255, 90 / 255, 160 / 255, 1]


def gradient(light, dark):
    def col(n, m):
        return [n * light[i] + m * dark[i] for i in range(4)]

    return [dark, col(.3, .7), col(.6, .4), col(.8, .2), light]


def get_color_scheme(n):
    if 1 <= n < 5:
        return [col4, col1, col2, col3][-n:]
    if n == 5:
        return gradient(light_blue, dark_blue)
    assert False


def multi_bar(ax, results, cat_names, colors):
    llbl, rlbl = list(zip(*[(r, l) for (l, r), _ in results]))
    uniq_rl = [x for k, v in groupby(rlbl)
               for x in [k] + [' '] * (sum(1 for __ in v) - 1)]
    labels = [i for i, _ in enumerate(results)]
    data = np.array([v for _, v in results])
    data_cum = data.cumsum(axis=1)
    ay = ax.secondary_yaxis('right')
    ax.invert_yaxis()

    for i, (name, color) in enumerate(zip(cat_names, colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        heights = [0.8 if label == "overall" else 0.6
                   for label in llbl]
        ax.barh(labels, widths, left=starts, height=heights,
                label=name, color=color)
        r, g, b, _ = color

    ax.set_yticklabels(llbl, weight='light')
    ax.set_yticks(np.arange(0, len(llbl), 1))
    ax.set_xlim(0, np.sum(data, axis=1).max())
    ax.yaxis.set_tick_params(length=0)
    ay.set_yticklabels(uniq_rl, weight='light')
    ay.set_ticks(np.arange(0, len(rlbl), 1))
    ay.yaxis.set_tick_params(length=0)
    for idx, lbl in enumerate(llbl):
        if idx < len(ax.get_yticklabels()) and lbl == "overall":
            ax.get_yticklabels()[idx].set_fontweight('bold')
    ax.get_yticklabels()[0].set_ha("left")
    ax.get_yticklabels()[0].set_position((.035, 0))
    ay.spines["right"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.set_tick_params(bottom=True, top=False)
    ax.set_xticks([])
    ax.set_xticklabels([])


def plot_acc(data, mean_data, subplots, plot_name, data_labels,
             sort_key=None, colors=None, dlen=4):
    if colors is None:
        color_count = len(mean_data)
        colors = get_color_scheme(color_count)
        colors.reverse()

    # determine plot size
    sp_n = len(subplots)
    plot_height = 1 + 4.5 * (len(data) / 24.)
    h_ratios = [1] if sp_n == 1 else \
        [len([x for x in data.values() if c == x[0][0]]) +
         (3 if i == sp_n - 1 else 0)
         for i, c in enumerate(subplots)]
    min_hr = max(.01, min(h_ratios))
    h_ratios = [round(h / min_hr, 2) for h in h_ratios]

    # setup figure
    fig, axes = plt.subplots(
        sp_n, 1, figsize=(3, plot_height),
        gridspec_kw={'height_ratios': h_ratios})
    ax = axes if sp_n == 1 else axes[-1]
    plt.subplots_adjust(wspace=0, hspace=0)

    # draw sub plots
    for i, ckey in enumerate(subplots):
        cdata = [(x[0][1:], x[1]) for x in data.values()
                 if ckey in x[0][0]]
        if sort_key is not None:
            cdata.sort(key=sort_key)
        cdata.insert(0, (('', ckey), [0] * dlen))
        if ckey == subplots[-1]:
            empty = [((' ', ' '), [0] * dlen)]
            overall = [v / sum(mean_data) * 100 for v in mean_data]
            cdata = cdata + empty + [(('', 'overall'), overall)]
        multi_bar(axes if len(subplots) == 1 else axes[i],
                  cdata, data_labels, colors=colors)

    # full figure formatting
    fig.legend(data_labels, ncol=2, bbox_to_anchor=(0.22, 1.05),
               loc='upper left', frameon=False,
               handlelength=.9, handletextpad=0.2,
               columnspacing=.8, borderpad=0)
    ax.yaxis.set_tick_params(length=0)
    # ax.xaxis.set_tick_params(labelsize='small')
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.spines['bottom'].set_visible(True)
    ax.xaxis.set_tick_params(bottom=True)
    ax.xaxis.set_visible(True)
    for i, lbl in enumerate(subplots):
        ax_ = axes if sp_n == 1 else axes[i]
        shift = 0 if sp_n > 1 and lbl == subplots[-1] else .03
        box = ax_.get_position()
        box.y0 += shift
        box.y1 += shift
        ax_.set_position(box)
    ensure_dir(plot_name)
    fig.tight_layout()
    plt.savefig(plot_name, bbox_inches='tight',
                metadata={'CreationDate': None})
    plt.close()


class BarData(ResultData):

    def get_acc_data(self):
        nums = [BarData.fmt(r) for r in self.raw_rata]
        means = np.rint(np.mean(np.array(
            [v for _, v in nums]), axis=0)).tolist()
        cats = sorted(list(set([x for ((x, _, _), _) in nums])))
        return dict(enumerate(nums)), means, cats

    @staticmethod
    def numeric_groups(data, mean_data, limit=3):
        if limit is None:
            groups = [str(i) for i in range(len(mean_data))]
        else:
            groups = [str(i) for i in range(limit)] + [f"{limit}+"]
            data = {name: r[:limit] + [sum(r[limit:])]
                    for name, r in data.items()}
            mean_data = mean_data[:limit] + [sum(mean_data[limit:])]
        return data, mean_data, groups

    @staticmethod
    def name(r):
        name = ResultData.name(r)
        return "UNSW" if name == "UNSW-NB15" else name

    @staticmethod
    def fmt(r):
        keys = (BarData.cls(r), BarData.name(r).lower(),
                BarData.attack(r).lower())
        valid = round(BarData.valid(r))
        evades = round(BarData.evades(r)) - valid
        accurate = round(BarData.acc(r)) - evades - valid
        total = 100 - accurate - evades - valid
        return keys, [valid, evades, accurate, total]

    def plot_name(self, pattern, out_dir):
        return self.fn_pattern('pdf', pattern, out_dir)


def plot_bars(data_dir, out_dir=None):
    bdata = BarData(data_dir)
    if bdata.n_results:
        nums, means, cats = bdata.get_acc_data()
        plot_acc(nums, means, subplots=cats,
                 data_labels=['valid', 'evasive', 'accurate', 'FP/FN'],
                 plot_name=bdata.plot_name('bar_acc', out_dir),
                 sort_key=
                 (lambda x: (x[1][-1], x[0][1]))
                 if 'perf' not in data_dir else
                 (lambda x: (x[0][0], x[0][1])))
