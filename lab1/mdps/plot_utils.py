import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

__all__ = ["plot_policy"]


def plot_policy(mdp, π, V):
    fig, ax = mdp.display(values=V)
    # action_names = [r"$\blacktriangle$", r"$\blacktriangleright$",
    # r"$\blacktriangledown$", r"$\blacktriangleleft$"]
    action_names = [r"$\uparrow$", r"$\rightarrow$", r"$\downarrow$", r"$\leftarrow$"]
    offsets = [(-0.05, -0.2), (0.15, -0.05), (-0.05, 0.15), (-0.2, -0.05)]
    for state_idx in mdp.S:
        if mdp.is_goal(state_idx):
            continue

        y, x = mdp._i2S[state_idx]
        # plot argmax for all optimal actions
        a_stars = np.flatnonzero(π[state_idx] == π[state_idx].max())
        for aidx in a_stars:
            action_glyph = action_names[aidx]
            dx, dy = offsets[aidx]
            ax.text(
                x + dx,
                y + dy,
                action_glyph,
                ha="center",
                va="center",
                fontsize="medium",
            )


def view_cmap(*cmap_list):
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))

    # Create figure and adjust figure height to number of colormaps
    nrows = len(cmap_list)
    figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
    fig, axs = plt.subplots(nrows=nrows + 1, figsize=(6.4, figh))
    fig.subplots_adjust(top=1 - 0.35 / figh, bottom=0.15 / figh, left=0.2, right=0.99)

    for ax, name in zip(axs, cmap_list):
        ax.imshow(gradient, aspect="auto", cmap=mpl.colormaps[name])
        ax.text(
            -0.01,
            0.5,
            name,
            va="center",
            ha="right",
            fontsize=10,
            transform=ax.transAxes,
        )

    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axs:
        ax.set_axis_off()
