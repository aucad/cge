from itertools import groupby

import matplotlib.pyplot as plt
import numpy as np

from exp import ensure_dir
from plot import ResultData

col0 = [240 / 255, 249 / 255, 232 / 255, 1]
col1 = [255 / 255, 201 / 255, 71 / 255, 1]
col2 = [100 / 255, 204 / 255, 197 / 255, 1]
col3 = [55 / 255, 71 / 255, 79 / 255,
        1]  # [34 / 255, 40 / 255, 49 / 255, 1]
col4 = [7 / 255, 102 / 255, 173 / 255, 1]
col5 = [8 / 255, 104 / 255, 172 / 255, 1]
light_blue = [166 / 255, 206 / 255, 227 / 255, 1]
dark_blue = [15 / 255, 90 / 255, 160 / 255, 1]
white = [1] * 4

# global options
overall = 'overall'
patterns = [None, None, None, None, None]
plt.rcParams['font.family'] = ['Arial']
plt.rcParams['hatch.linewidth'] = .5


def gradient(light, dark):
    def col(n, m):
        return [n * light[i] + m * dark[i] for i in range(4)]

    return [dark, col(.3, .7), col(.6, .4), col(.8, .2), light]


def get_color_scheme(n):
    if 1 <= n < 5:
        return [white, col1, col2, col3][-n:]
    if n == 5:
        return gradient(light_blue, dark_blue)
    assert False


def multi_bar(ax, results, cat_names, colors):
    llbl, rlbl = list(zip(*[
        (r, l.replace("0", "").replace("PT-1-", "")
         .replace("PT-2-", "")) for (l, r), _ in results]))
    uniq_rl = [x for k, v in groupby(rlbl)
               for x in [k] + [' '] * (sum(1 for __ in v) - 1)]
    labels = [i for i, _ in enumerate(results)]
    has_ov = llbl[-1] == overall
    data = np.array([v for _, v in results])
    data_cum = data.cumsum(axis=1)
    ay = ax.secondary_yaxis('right')
    ax.invert_yaxis()

    barh = ([0.6] * (len(llbl) - 2)) + [.8 if has_ov else 0.6]
    totals = [sum(d) - .3 for d in data[1:]]
    for i, (name, color) in enumerate(zip(cat_names, colors)):
        widths = data[1:, i]
        starts = data_cum[1:, i] - widths
        ax.barh(labels[1:], widths, left=starts, height=barh,
                label=name, color=color, zorder=0, lw=0,
                hatch=patterns[i],
                edgecolor=[.95, .95, .95, 1])
    bars = ax.barh(
        labels[1:], totals, left=.1, height=barh, color='none',
        lw=0.3, edgecolor=[0, 0, 0, 1], zorder=2, hatch=None)
    for idx in [i for i, tot in enumerate(totals) if tot < 1]:
        bars[idx].set_linewidth(0)

    ax.set_yticklabels(llbl)
    ax.set_yticks(np.arange(0, len(llbl), 1))
    ax.set_xlim(0, np.sum(data, axis=1).max())
    ax.yaxis.set_tick_params(length=0)
    ay.set_yticklabels(uniq_rl)
    ay.set_ticks(np.arange(0, len(rlbl), 1))
    ay.yaxis.set_tick_params(length=0)
    for idx, lbl in enumerate(llbl):
        if idx < len(ax.get_yticklabels()) and lbl == overall:
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


def plot_acc(input_data, plot_name, data_labels,
             sort_key=None, colors=None, dlen=4, overall_bar=True):
    data, mean_data, subplots = input_data[0]
    if colors is None:
        color_count = len(mean_data)
        colors = get_color_scheme(color_count)
        colors.reverse()

    # determine plot size
    sp_n, pl_n = len(subplots), len(input_data)
    plot_height = (1 if overall_bar else .7) + 4.5 * (len(data) / 24.)
    h_ratios = [1] if sp_n == 1 else \
        [len([x for x in data.values() if c == x[0][0]]) +
         (3 if i == sp_n - 1 else 0)
         for i, c in enumerate(subplots)]
    min_hr = max(.01, min(h_ratios))
    h_ratios = [round(h / min_hr, 2) for h in h_ratios]

    # setup figure
    fig, axes = plt.subplots(
        sp_n, pl_n, figsize=(3 * pl_n, plot_height),
        gridspec_kw={'height_ratios': h_ratios})
    ax = axes if sp_n == 1 else axes[-1]
    plt.subplots_adjust(wspace=0, hspace=0)

    # draw sub plots
    for i, ckey in enumerate(subplots):
        cdata = [(x[0][1:], x[1])
                 for x in data.values()
                 if ckey == x[0][0]]
        if sort_key is not None:
            cdata.sort(key=sort_key)
        cdata.insert(0, (('', ckey), [0] * dlen))
        if ckey == subplots[-1] and overall_bar:
            empty = [((' ', ' '), [0] * dlen)]
            ov = [v / sum(mean_data) * 100 for v in mean_data]
            cdata = cdata + empty + [(('', overall), ov)]
        multi_bar(axes if len(subplots) == 1 else axes[i],
                  cdata, data_labels, colors=colors)

    # full figure formatting
    leg = fig.legend(data_labels, ncol=2, bbox_to_anchor=(0.22, 1.05),
                     loc='upper left', frameon=False,
                     handlelength=.9, handletextpad=0.4,
                     columnspacing=.8, borderpad=0)
    for p in leg.get_patches():
        p.set_edgecolor([0, 0, 0, .85])
        p.set_linewidth(.75)

    ax.yaxis.set_tick_params(length=0)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.spines['bottom'].set_visible(True)
    ax.xaxis.set_tick_params(bottom=True)
    ax.xaxis.set_visible(True)
    for i, lbl in enumerate(subplots):
        ax_ = axes if sp_n == 1 else axes[i]
        shift = 0 if sp_n > 1 and lbl == subplots[-1] else \
            (.1 if sp_n == 1 else .03)
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

    def get_acc_data(self, key_test):
        is_perf = 'perf' in self.directory
        nums = [BarData.fmt(r) for r in self.raw_rata if key_test(r)]
        means = np.rint(np.mean(np.array(
            [v for _, v in nums]), axis=0)).tolist()
        if not is_perf:
            cnums = [BarData.fmt(
                r, key="Comparison", att="CPGD")
                for r in self.raw_rata if
                ResultData.attack(r) == 'CPGD']
            if len(cnums):
                nums += cnums
        cats = sorted(list(set([x for ((x, _, _), _) in nums])))
        ndict = dict(enumerate(nums))
        return ndict, means, cats

    @staticmethod
    def name(r):
        name = ResultData.name(r)
        return "UNSW" if name == "UNSW-NB15" else name

    @staticmethod
    def attack(r):
        return ResultData.attack(r).upper()

    @staticmethod
    def fmt(r, key=None, att=None):
        keys = (key or BarData.cls(r), BarData.name(r),
                att or BarData.attack(r))
        valid = round(BarData.valid(r))
        evades = round(BarData.evades(r)) - valid
        accurate = round(BarData.acc(r)) - evades - valid
        total = 100 - accurate - evades - valid
        return keys, [valid, evades, accurate, total]

    def plot_name(self, pattern, out_dir):
        return self.fn_pattern('pdf', pattern, out_dir)


def make_plot(bdata, out_dir, is_perf, plot_name, key_test,
              overall_bar=True):
    bar_inputs = [[d.get_acc_data(key_test) for d in bdata][0]]
    # assert (bars have same shape)
    legend = ['valid', 'evasive', 'accurate', 'inaccurate']
    plot_acc(bar_inputs,
             data_labels=legend, overall_bar=overall_bar,
             plot_name=bdata[0].plot_name(plot_name, out_dir),
             sort_key=(lambda x: (x[0][0], len(x[0][1]), x[0][1]))
             if is_perf else
             (lambda x: (x[0][0], x[0][1])))


def plot_bars(data_dir, out_dir=None):
    dirs = data_dir.split(',')
    is_perf = 'perf/' in data_dir
    bdata = [BarData(d) for d in dirs]
    if is_perf:
        for cats in ['PT-1', 'PT-2']:
            key_test = lambda r: cats in BarData.name(r)
            if len(bdata) and bdata[0].n_results:
                make_plot(
                    bdata, out_dir, is_perf, 'bar_acc_' + cats,
                    key_test, overall_bar=False)
    else:
        key_test = lambda r: ResultData.attack(r) != 'CPGD'
        if len(bdata) and bdata[0].n_results:
            make_plot(bdata, out_dir, is_perf, 'bar_acc', key_test)
