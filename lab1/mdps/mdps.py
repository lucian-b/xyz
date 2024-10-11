import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .blueprints import Blueprint


class MDP:
    def __init__(self, blueprint: str):
        bprint = Blueprint.from_str(blueprint)
        self.world = bprint.map_
        self.S0_yx = bprint.s0_yx
        self.goals_yx = bprint.goal_yx
        self.rspec = bprint.rspec
        self.a2m = {
            0: np.array([-1, 0]),
            1: np.array([0, 1]),
            2: np.array([1, 0]),
            3: np.array([0, -1]),
        }

        # set of reachable states and a mapping
        self._S = list(zip(*np.where(self.world != "x"), strict=True))
        self._i2S = {i: s for i, s in enumerate(self._S)}
        self._S2i = {s: i for i, s in enumerate(self._S)}

        self.SA_space = (len(self._S), len(self.a2m))

        # compute the set of states S, the Psas and the Rsas
        self._Psas, self._Rsas = self._get_mdp()

    @property
    def num_states(self):
        return len(self.S)

    @property
    def num_actions(self):
        return len(self.a2m)

    @property
    def S(self):
        return list(self._i2S.keys())

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
            return self._Psas.sum(1) / len(self.a2m)
        else:
            return (self._Psas * π[:, :, None]).sum(1)

    def get_Rss(self, π=None):
        if π is None:
            return self._Rsas.sum(1) / len(self.a2m)
        else:
            return (self._Rsas * π[:, :, None]).sum(1)

    def _get_mdp(self):
        w = self.world.copy()
        w[w == "s"] = " "  # we treat S0 as a usual state

        N = len(self._S)
        P = np.zeros((N, len(self.a2m), N))
        R = np.zeros((N, len(self.a2m), N))
        for yx in self._S:
            s = self._S2i[yx]

            # absorbing state
            if np.char.isupper(w[yx]):
                P[s, :, s] = 1.0
                R[s, :, s] = 0.0
                continue

            for a, move in self.a2m.items():
                yx_ = tuple(np.array(yx) + move)

                if w[yx_] == "x":
                    yx_ = yx
                    r = self.rspec["x"]
                else:
                    r = self.rspec[w[yx_]]

                s_ = self._S2i[yx_]
                P[s, a, s_] = 1
                R[s, a, s_] = r
        P.flags.writeable = False
        R.flags.writeable = False
        return P, R

    # what follows are methods
    # meant to be used by a game implementation

    def is_goal(self, s):
        if isinstance(s, tuple):
            return np.char.isupper(self.world[s])
        else:
            return np.char.isupper(self.world[self._i2S[s]])

    def get_init_states(self):
        return [self._S2i[x] for x in self.S0_yx]

    def get_goal_states(self):
        return [self._S2i[x] for x in self.goals_yx]

    # some eye candy

    def display(self, values=None, cmap="Blues", ax=None, vmin=None, vmax=None):
        palette = np.array(
            [
                [255, 255, 255],  # white
                [0, 0, 0],  # black
                [233, 233, 233],  # grey
                [0, 255, 0],  # green
                [255, 0, 0],  # red
            ]
        )
        M = np.zeros_like(self.world, dtype=np.uint8)
        M[self.world == "x"] = 1.0
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
        for y, x in self.goals_yx:
            ax.text(x, y, "G", ha="center", va="center", color="w")
        for y, x in self.S0_yx:
            ax.text(x, y, "S", ha="center", va="center")

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

    def __repr__(self):
        s = "\n".join(["".join(line) for line in self.world])
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


class Env:
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
