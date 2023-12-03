from itertools import groupby
from os import path

import matplotlib.pyplot as plt
import numpy as np

from exp import ensure_dir
from plot import ResultData

# colors
col1 = [x / 255 for x in [237, 248, 177, 255]]
col2 = [x / 255 for x in [127, 205, 187, 255]]
col3 = [x / 255 for x in [44, 127, 184, 255]]
white = [1] * 4

# global options
overall = 'overall'
plt.rcParams['font.family'] = ['Arial']


def get_color_scheme(n):
    if 1 <= n < 5:
        return [white, col1, col2, col3][-n:]
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
                label=name, color=color, zorder=0, lw=0)
    bars = ax.barh(
        labels[1:], totals, left=.1, height=barh, color='none',
        lw=0.3, edgecolor=[0, 0, 0, 1], zorder=2)
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
        nrows=sp_n, ncols=pl_n,
        figsize=(3 * pl_n, plot_height),
        gridspec_kw={'height_ratios': h_ratios})
    plt.subplots_adjust(wspace=0, hspace=0)

    for ix, (data, mean_data, subplots) in enumerate(input_data):
        if sp_n == 1 and pl_n == 1:
            ax = sp_ax = axes
        elif sp_n == 1 and pl_n > 1:
            sp_ax = [[x] for x in axes]
            ax = sp_ax[ix][-1]
        else:
            axes_ = axes if pl_n > 1 else [[x] for x in axes]
            sp_ax = [x[ix] for x in axes_]
            ax = sp_ax[-1]

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
            multi_bar(ax if len(subplots) == 1 else sp_ax[i],
                      cdata, data_labels, colors=colors)

        ax.yaxis.set_tick_params(length=0)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.spines['bottom'].set_visible(True)
        ax.xaxis.set_tick_params(bottom=True)
        ax.xaxis.set_visible(True)
        if pl_n == 1:
            for i, lbl in enumerate(subplots):
                ax_ = sp_ax if sp_n == 1 else sp_ax[i]
                shift = 0 if sp_n > 1 and lbl == subplots[-1] else \
                    (.1 if sp_n == 1 else .03)
                box = ax_.get_position()
                box.y0 += shift
                box.y1 += shift
                ax_.set_position(box)

    # full figure formatting
    leg = fig.legend(
        data_labels,
        ncol=len(data_labels) if pl_n > 1 else 2,
        bbox_to_anchor=(
            (0.22, 1.1) if pl_n > 1 and sp_n == 1
            else (0.22, 1.05)),
        loc='upper left', frameon=False,
        handlelength=.9, handletextpad=0.4,
        columnspacing=.8 if pl_n == 1 else 1.5,
        borderpad=0,
        **{'prop': {'size': 11} if pl_n > 1 else None})
    for p in leg.get_patches():
        p.set_edgecolor([0, 0, 0, .85])
        p.set_linewidth(.75)
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

    def fn_pattern(self, file_ext, pattern, out_dir=None, in_dirs=None):
        flat_name = (in_dirs or self.directory).replace('/', '_')
        file_name = f'__{pattern}_{flat_name}'
        return path.join(out_dir, f'{file_name}.{file_ext}')

    @staticmethod
    def fmt(r, key=None, att=None):
        keys = (key or BarData.cls(r), BarData.name(r),
                att or BarData.attack(r))
        valid = round(BarData.valid(r))
        evades = round(BarData.evades(r)) - valid
        accurate = round(BarData.acc(r)) - evades - valid
        total = 100 - accurate - evades - valid
        return keys, [valid, evades, accurate, total]

    def plot_name(self, pattern, out_dir, dirs=None):
        return self.fn_pattern('pdf', pattern, out_dir, in_dirs=dirs)


def match_bdata(x, y):
    for xk in [x[0] for x in x[0].values()]:
        pair = None
        for yk in [y[0] for y in y[0].values()]:
            attm = xk[2] == yk[2] or \
                   xk[2][1:] == yk[2] or xk[2] == yk[2][1:]
            if attm and xk[0] == yk[0] and xk[1] == yk[1]:
                pair = yk
        assert pair is not None


def attack_plot(bdata, out_dir, plot_name, dirs=None):
    key_test = lambda r: ResultData.attack(r) != 'CPGD'
    bar_inputs = [d.get_acc_data(key_test) for d in bdata]
    for b in bar_inputs[1:]:
        match_bdata(bar_inputs[0], b)
    plot_acc(
        bar_inputs, overall_bar=True,
        data_labels=['valid', 'evasive', 'accurate', 'inaccurate'],
        plot_name=bdata[0].plot_name(plot_name, out_dir, dirs=dirs),
        sort_key=(lambda x: (x[0][0], x[0][1])))


def perf_plot(bdata, out_dir, plot_name):
    bar_inputs = [
        tuple(bdata[0].get_acc_data(
            lambda r: cats in BarData.name(r)))
        for cats in ['PT-1', 'PT-2']]
    plot_acc(
        bar_inputs, overall_bar=False,
        data_labels=['valid', 'evasive', 'accurate', 'inaccurate'],
        plot_name=bdata[0].plot_name(plot_name, out_dir),
        sort_key=(lambda x: (x[0][0], len(x[0][1]), x[0][1])))


def plot_bars(data_dir, out_dir=None):
    dirs = data_dir.split(',')
    bdata = [BarData(d) for d in dirs]
    if not (len(bdata) and bdata[0].n_results):
        return
    if 'perf/' in data_dir:
        return perf_plot(bdata, out_dir, 'bar_acc')
    else:
        attack_plot(
            bdata, out_dir, 'bar_acc',
            dirs='_'.join(dirs) if len(dirs) > 1 else None)
