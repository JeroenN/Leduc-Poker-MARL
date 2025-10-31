import numpy as np
import matplotlib.pyplot as plt


def save_comparison_with_ci(
    x1, y1, yerr1, label1,
    x2, y2, yerr2, label2,
    xlabel, ylabel, filename
):

    y1 = np.asarray(y1); yerr1 = np.asarray(yerr1)
    y2 = np.asarray(y2); yerr2 = np.asarray(yerr2)

    plt.figure()
    
    plt.plot(x1, y1, label=label1, linewidth=2, color='tab:blue')
    plt.fill_between(x1, y1 - yerr1, y1 + yerr1, alpha=0.20, color='tab:blue')

    plt.plot(x2, y2, label=label2, linewidth=2, linestyle='--', color='tab:orange')
    plt.fill_between(x2, y2 - yerr2, y2 + yerr2, alpha=0.20, color='tab:orange')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle=':', linewidth=0.8)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def save_gap_mean_comparison(
    steps_cfr, gap_mean_cfr,
    steps_ocfr, gap_mean_ocfr,
    filename
):
    """Plot comparison of average gap to best response (mean over both seats)."""
    plt.figure()
    plt.plot(steps_cfr, gap_mean_cfr, label="CFR (self-play)", linewidth=2, color="tab:blue")
    plt.plot(steps_ocfr, gap_mean_ocfr, label="OCFR (vs fixed)", linewidth=2, linestyle="--", color="tab:orange")
    plt.xlabel("CFR Iterations")
    plt.ylabel("Gap to Best Response (mean)")
    plt.ylim(bottom=0)
    plt.legend()
    plt.grid(True, linestyle=":", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def save_exploitability_comparison(
    steps1, exp1, label1,
    steps2, exp2, label2,
    filename
):
    """Plot a single figure comparing exploitability for CFR vs OCFR."""
    # In case step grids differ, align on common steps
    d1 = dict(zip(steps1, exp1))
    d2 = dict(zip(steps2, exp2))
    common = sorted(set(d1).intersection(d2))
    y1 = [d1[s] for s in common]
    y2 = [d2[s] for s in common]

    plt.figure()
    plt.plot(common, y1, label=label1, linewidth=2, color="tab:blue")
    plt.plot(common, y2, label=label2, linewidth=2, linestyle="--", color="tab:orange")
    plt.xlabel("CFR Iterations")
    plt.ylabel("Exploitability")
    plt.ylim(bottom=0)
    plt.legend()
    plt.grid(True, linestyle=":", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()