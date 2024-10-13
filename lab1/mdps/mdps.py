import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .blueprints import Blueprint


class MDP:
    def __init__(self, blueprint: str):
        bprint = Blueprint.from_str(blueprint)
        self.world = bprint.map_
        self.X0_yx = bprint.s0_yx
        self.goals_yx = bprint.goal_yx
        self.rspec = bprint.rspec

        # set of reachable states in coords (y,x)
        self._yx = list(zip(*np.where(self.world != "x"), strict=True))
        # mapping from states (as indices) 2 states as coords
        self._x2yx = {i: s for i, s in enumerate(self._yx)}
        self._yx2x = {s: i for i, s in enumerate(self._yx)}

        # actions (4 actions by default)
        self._U = [0, 1] if self.world.shape[0] == 3 else [0, 1, 2, 3]

        # compute the set of states S, the Psas and the Rsas
        self._Psas, self._Rsas = self._get_mdp()

    @property
    def nX(self):
        return len(self._yx)

    @property
    def nU(self):
        return len(self._U)

    @property
    def nXU(self):
        return (self.nX, self.nU)

    @property
    def X(self):
        return list(self._x2yx.keys())

    @property
    def U(self):
        return self._U

    @property
    def Psas(self):
        return self._Psas

    def f(self, x, u):
        return np.flatnonzero(self.Psas[x, u]).item()

    @property
    def Rsas(self):
        return self._Rsas

    def rho(self, x, u):
        return (self._Rsas[x, u] * self._Psas[x, u]).sum()

    def ρ(self, x, u):
        return self.rho(x, u)

    def get_Pss(self, π=None):
        if π is None:
            return self._Psas.sum(1) / self.nU
        else:
            return (self._Psas * π[:, :, None]).sum(1)

    def get_Rss(self, π=None):
        if π is None:
            return self._Rsas.sum(1) / self.nU
        else:
            return (self._Rsas * π[:, :, None]).sum(1)

    def _get_mdp(self):
        a2m = {
            0: np.array([-1, 0]),
            1: np.array([0, 1]),
            2: np.array([1, 0]),
            3: np.array([0, -1]),
        }
        if self.nU == 2:
            a2m = {0: np.array([0, 1]), 1: np.array([0, -1])}

        w = self.world.copy()
        w[w == "s"] = " "  # we treat S0 as a usual state

        P = np.zeros((self.nX, self.nU, self.nX))
        R = np.zeros((self.nX, self.nU, self.nX))
        for yx in self._yx:
            x = self._yx2x[yx]

            # absorbing state
            if np.char.isupper(w[yx]):
                P[x, :, x] = 1.0
                R[x, :, x] = 0.0
                continue

            for u, move in a2m.items():
                yx_ = tuple(np.array(yx) + move)

                if w[yx_] == "x":
                    yx_ = yx
                    r = self.rspec["x"]
                else:
                    r = self.rspec[w[yx_]]

                x_ = self._yx2x[yx_]
                P[x, u, x_] = 1
                R[x, u, x_] = r
        P.flags.writeable = False
        R.flags.writeable = False
        return P, R

    # what follows are methods
    # meant to be used by a game implementation

    def is_goal(self, x):
        if isinstance(x, tuple):
            return np.char.isupper(self.world[x])
        else:
            return np.char.isupper(self.world[self._x2yx[x]])

    def get_init_states(self):
        return [self._yx2x[x] for x in self.X0_yx]

    def get_goal_states(self):
        return [self._yx2x[x] for x in self.goals_yx]

    # some eye candy
    @staticmethod
    def plot_values(mdp, values=None, cmap="viridis", ax=None, vmin=None, vmax=None):
        palette = np.array(
            [
                [255, 255, 255],  # white
                [0, 0, 0],  # black
                [233, 233, 233],  # grey
                [0, 255, 0],  # green
                [255, 0, 0],  # red
            ]
        )
        M = np.zeros_like(mdp.world, dtype=np.uint8)
        M[mdp.world == "x"] = 1.0
        # M[self.world=="G"] = 3
        # M[self.world=="S"] = 2

        mask = M == 0
        M_ = M.copy().astype(np.float32)
        M_[M_ != 1] = values
        M_ma = np.ma.array(M_, mask=~mask)

        fig = None
        if ax is None:
            fig, ax = plt.subplots()

        im = ax.imshow(palette[M], cmap="binary")
        im = ax.imshow(M_ma, cmap=cmap, vmin=vmin, vmax=vmax)

        # plot the goal
        for y, x in mdp.goals_yx:
            goal = mdp.world[y, x]
            ax.text(x, y, goal, ha="center", va="center", color="w")
        for y, x in mdp.X0_yx:
            ax.text(x, y, "s", ha="center", va="center")

        if fig is not None:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)

            plt.colorbar(im, cax=cax)
            return fig, ax
        else:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)

            plt.colorbar(im, cax=cax)
            return None

    @staticmethod
    def qsa_lineplot(Q):
        fig, ax = plt.subplots(figsize=(5, 3.75), layout="constrained")
        nX, nU = Q.shape
        # xs = np.arange(-0.5, nX - 0.5, 1)
        xs = np.arange(nX)
        us = ("top", "right", "down", "left") if nU == 4 else ("right", "left")
        for qu, u in zip(Q.t(), us, strict=True):
            print(qu, xs)
            ax.step(xs, qu, where="pre", label=f"Q(x, {u})")
        ax.legend()

    @staticmethod
    def plot_policy(mdp, π, V):
        fig, ax = MDP.plot_values(mdp, values=V)
        # action_names = [r"$\blacktriangle$", r"$\blacktriangleright$",
        # r"$\blacktriangledown$", r"$\blacktriangleleft$"]
        action_names = [
            r"$\uparrow$",
            r"$\rightarrow$",
            r"$\downarrow$",
            r"$\leftarrow$",
        ]
        offsets = [(-0.05, -0.2), (0.15, -0.05), (-0.05, 0.15), (-0.2, -0.05)]
        for state_idx in mdp.X:
            if mdp.is_goal(state_idx):
                continue

            y, x = mdp._x2yx[state_idx]
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
                    color="black",
                )

    def __repr__(self):
        s = "\n".join(["".join(line) for line in self.world])
        if self.nU == 2:
            s += "\nActions: 0: right, 1: left\n"
        else:
            s += "\nActions: 0: up, 1: right, 2: down, 3: left\n"
        return s


class Identity:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def __call__(self, state):
        return state


class OneHot:
    def __init__(self, mdp: MDP, *args, **kwargs) -> None:
        self._eye = np.eye(len(mdp.S), dtype=np.float32)

    def __call__(self, state):
        return self._eye[state]


class GridCoordinates:
    def __init__(self, mdp: MDP, *args, **kwargs) -> None:
        self.mdp = mdp

    def __call__(self, state):
        return self.mdp._i2S[state]


class ScaledGridCoordinates:
    def __init__(self, mdp: MDP, *args, **kwargs) -> None:
        self.mdp = mdp

    def __call__(self, state):
        rows, cols = self.mdp.world.shape
        r, c = self.mdp._i2S[state]
        # _i2s is one-based indexing because of the walls so we offset
        r, c = (r - 1) / (rows - 3), (c - 1) / (cols - 3)  # (0, 1) normalization
        yx = ((r - 0.5) * 2, (c - 0.5) * 2)  # (-1, 1) normalization
        return yx, state


PHI = {
    "identity": Identity,
    "onehot": OneHot,
    "coords": GridCoordinates,
    "scaled_coords": ScaledGridCoordinates,
}


class Sim:
    def __init__(self, mdp, phi="identity"):
        self.mdp = mdp
        self.phi = PHI[phi](mdp)

    def reset(self):
        self._crt_state = np.random.choice(self.mdp.get_init_states())
        self._goal_states = self.mdp.get_goal_states()
        return self.phi(self._crt_state), False

    def step(self, action):
        # use transition and reward matrices to compute next state and reward
        state = self._crt_state
        next_state = np.flatnonzero(self.mdp.Psas[state, action]).item()
        reward = self.mdp.Rsas[state, action, next_state].item()

        self._crt_state = next_state
        done = next_state in self._goal_states
        return self.phi(next_state), reward, done

    def states(self):
        return self.mdp.S

    def observations(self):
        return [self.phi(s) for s in self.mdp.S]
