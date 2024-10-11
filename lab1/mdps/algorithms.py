from dataclasses import dataclass

import numpy as np


def policy_evaluation(mdp, V: np.array, π: np.array, δ: float, γ: float):
    """Alternatives:
    One iteration is the equivalent of: V = Rss.sum(1) + γ * Pss @ V
    Or just do: V = np.linalg.inv(np.eye(Ns) - Pss * γ) @ Rss.sum(1)
    """

    Psas = mdp.Psas
    Rsas = mdp.Rsas

    Vnew = np.zeros_like(V)
    for s in mdp.S:
        if mdp.is_goal(s):
            continue

        # get the actions and the next states they lead to
        for a, s_ in zip(*np.where(Psas[s] == 1)):
            π_a = π[s, a]
            r = Rsas[s, a, s_]
            Vnew[s] += π_a * (r + γ * V[s_])

        # threshold
        δ = max(δ, abs(V[s] - Vnew[s]))
    return Vnew, δ


def policy_improvement(mdp, V: np.array, γ: float):
    Ns, Na = mdp.SA_space
    Psas = mdp.Psas
    Rsas = mdp.Rsas

    π = np.zeros((Ns, Na))
    for s in mdp.S:
        if mdp.is_goal(s):
            continue

        values = np.zeros(Na)
        for a, s_ in zip(*np.where(Psas[s] == 1)):
            r = Rsas[s, a, s_]
            values[a] = r + γ * V[s_]

        # maybe there are ties
        argmax_actions = np.flatnonzero(values == values.max())
        π[s] = [
            1 / len(argmax_actions) if i in argmax_actions else 0 for i in range(Na)
        ]
    return π


@dataclass
class PIstep:
    π: np.array
    V: np.array
    t: int
    npe: int
    δv: float
    δπ: float

    def __str__(self):
        return "PolicyIteration(t:{:3d}, npe:{:3d}, δv:{:5f}, δπ:{:.2f})".format(
            self.t, self.npe, self.δv, self.δπ
        )


def policy_iteration(
    mdp,
    γ: float = 0.99,
    δV: float = 0.0001,
    T: int = 1000,
) -> list[PIstep]:
    Ns, Na = mdp.SA_space

    # initial pi and V
    V = np.random.randn(Ns) / 100
    π = np.array([1 / Na] * Ns * Na).reshape((Ns, Na))

    pivs = []
    for t in range(T):
        # policy evaluation
        pe_cnt = 0
        while True:
            V, δ = policy_evaluation(mdp, V, π, 0, γ=γ)
            pe_cnt += 1
            if δ < δV:
                break

        # policy improvement
        π_ = policy_improvement(mdp, V, γ=γ)
        δπ = (π_ == π).all(axis=1).sum() / Ns

        # append resulting policies and their associated values
        pivs.append(PIstep(π, V, t, pe_cnt, δ, δπ))

        # early exit if policy does not change
        if (π_ == π).all():
            return pivs

        π = π_

    return pivs
