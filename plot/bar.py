import matplotlib.pyplot as plt
import numpy as np

from plot import ResultData
from exp import ensure_dir

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
        return [col4, col3, col2, col1][-n:]
    if n == 5:
        return gradient(light_blue, dark_blue)
    assert False


def multi_bar(results, category_names, colors=None,
              size=(5.8, 3.1), legend=True, log=False):
    labels = [r[0] for r in results]
    data = np.array([r[1] for r in results])
    data_cum = data.cumsum(axis=1)
    if colors is None:
        colors = plt.get_cmap('RdYlGn')(
            np.linspace(1, 0.5, data.shape[1]))

    fig, ax = plt.subplots(figsize=size)
    ax.invert_yaxis()

    for i, (colname, color) in enumerate(
            zip(category_names, colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        heights = [0.8 if label == "overall" else 0.5
                   for label in labels]
        ax.barh(labels, widths, left=starts, height=heights,
                label=colname, color=color)
        r, g, b, _ = color

    if "overall" in labels:
        for idx, lbl in enumerate(labels):
            if lbl == "overall":
                ax.get_yticklabels()[idx].set_fontweight('bold')
            elif lbl:
                ax.get_yticklabels()[idx].set_fontweight('light')
                ax.get_yticklabels()[idx].set_size('small')

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


def plot(data, mean_data, plot_name, groups,
         sort_key=None, colors=None):
    clss = sorted(list(set([x[1] for x in data.keys()])))
    plot_data = []
    for i, cls in enumerate(clss):
        cls_data = [x for x in data.items() if x[0][1] == cls]
        if sort_key is not None:
            cls_data.sort(key=sort_key)
        cls_data = [(f'{name[0]} {name[2]} {name[3]}', r)
                    for name, r in cls_data]
        plot_data = plot_data + cls_data

    plot_data = plot_data + [(' ', [0] * len(mean_data))] + [
        ("overall", [v / sum(mean_data) * 100
                     for v in mean_data])]
    if colors is None:
        color_count = len(mean_data)
        colors = get_color_scheme(color_count)
        colors.reverse()

    fig, ax = multi_bar(
        plot_data, groups, size=(3, 4.5),
        colors=colors, legend=False)
    ax.legend(ncol=4, bbox_to_anchor=(-0.55, 0.97),
              loc='lower left', fontsize='small', frameon=False,
              handlelength=.9, handletextpad=0.2,
              columnspacing=1.1, borderpad=0)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.yaxis.set_tick_params(length=0)

    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.tick_params(axis='x', labelsize='small')
    fig.tight_layout()
    ensure_dir(plot_name)
    plt.savefig(plot_name, metadata={'CreationDate': None})
    plt.close()


def get_groups(data, mean_data, limit=3):
    if limit is None:
        groups = [str(i) for i in range(len(mean_data))]
    else:
        groups = [str(i) for i in range(limit)] + [f"{limit}+"]
        data = {name: r[:limit] + [sum(r[limit:])]
                for name, r in data.items()}
        mean_data = mean_data[:limit] + [sum(mean_data[limit:])]
    return data, mean_data, groups


class BarData(ResultData):

    @staticmethod
    def r_cls(r):
        cls = ResultData.r_cls(r)
        if cls.lower() == "neural network":
            return "NN"
        if cls.lower() == "xgboost":
            return "XGB"
        return cls

    @staticmethod
    def r_name(r):
        name = ResultData.r_name(r)
        if name == "UNSW-NB15":
            return "UNSW"
        return name

    @staticmethod
    def fmt_attack_name(r):
        tmp = ResultData.fmt_attack_name(r)
        return 'CPGD' if tmp == 'CPGD[R]' else tmp

    def get_acc_data(self):
        nums = [BarData.fmt(i, r) for i, r in
                enumerate([x for x in self.raw_rata if
                           BarData.fmt_attack_name(x) != "CPGD"])]
        m_valid = int(round(sum(
            [x[1][0] for x in nums]) / len(nums), 0))
        m_evades = int(round(sum(
            [x[1][1] for x in nums]) / len(nums), 0))
        m_acc = int(round(sum(
            [x[1][2] for x in nums]) / len(nums), 0))
        m_tot = int(round(sum(
            [x[1][3] for x in nums]) / len(nums), 0))
        means = [m_valid, m_evades, m_acc, m_tot]
        return dict(nums), means

    @staticmethod
    def fmt(i, r):
        ac = r['folds']['accuracy']
        ev = r['folds']['n_evasions']
        vd = r['folds']['n_valid_evades']
        cls = BarData.r_cls(r)
        name = BarData.r_name(r)
        attack = BarData.fmt_attack_name(r)
        valid = int(round(BarData.fold_mean(vd, r), 0))
        evades = int(round(BarData.fold_mean(ev, r), 0)) - valid
        acc = int(round(BarData.arr_mean(ac), 0)) - evades - valid
        tot = 100 - acc - evades - valid
        key = (str(i), cls, name, attack)
        return key, [valid, evades, acc, tot]

    def plot_name(self, pattern, out_dir):
        return self.fn_pattern('pdf', pattern, out_dir)


def plot_bars(data_dir, out_dir=None):
    bdata = BarData(data_dir)
    nums, means = bdata.get_acc_data()
    plot(nums, means,
         plot_name=bdata.plot_name('bar_acc', out_dir),
         groups=['valid', 'evasive', 'accurate', 'FP/N'],
         sort_key=lambda x: (x[0][2], x[0][3]))
