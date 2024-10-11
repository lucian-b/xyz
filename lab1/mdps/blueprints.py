from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass
class Blueprint:
    map_: ArrayLike
    s0_yx: tuple
    goal_yx: tuple
    rspec: dict[str, float]

    @classmethod
    def from_str(cls, s: str) -> Blueprint:
        header, blueprint = s.split("---\n")
        hdr_lines = header.strip().strip("\t").split("\n")
        map_lines = blueprint.strip().strip("\t").split("\n")
        map_ = np.array([list(line) for line in map_lines])
        # rewards
        rspec = (rspec.split(":") for rspec in hdr_lines)
        rspec = {k: float(v) for k, v in rspec}
        # m(ove) is a special one
        rspec[" "] = rspec.pop(".") if "." in rspec else 0.0
        # we treat S as usual
        rspec["s"] = rspec[" "]
        # start and goal positions
        s0_yx = tuple(zip(*np.nonzero(map_ == "s"), strict=True))
        goal_yx = tuple(zip(*np.nonzero(np.char.isupper(map_)), strict=True))
        return Blueprint(map_, s0_yx, goal_yx, rspec)


ROOMBA = """
B: 1.0
T: 5.0
x: 0.0
.: 0.0
---
xxxxxxxx
xB  s Tx
xxxxxxxx
"""


FOUR_ROOMS_X = """
G=20.0
r=1.0
g=-1.0
x=0.0
m=0.0
---
xxxxxxxxxxxxx
x     x    Gx
x     x     x
x  r        x
x     x     x
x     x     x
xxx xxx     x
xgrgrgxxx xxx
x    rx     x
x    g   r  x
x    rx     x
xs   gx     x
xxxxxxxxxxxxx
"""


SLALOM = """
G=1.0
r=0.1
g=-0.1
x=0.0
move=0.0
---
xxxxxxxxxxxx
xS         x
x          x
xg g g g g x
x          x
x r r r r rx
x          x
xg g g g g x
x          x
x         Gx
xxxxxxxxxxxx
"""


TWO_ROOMS = """
G=1.0
r=1.0
g=-2.0
x=0.0
move=0.0
---
xxxxxxxxxx
xS       x
x        x
x        x
xxxxx xxxx
x        x
x        x
x        x
x       Gx
xxxxxxxxxx
"""


TWO_ROOMS_SMALL = """
G=5.0
x=-1
m=0.0
xxxxxxxx
x     Gx
x      x
xxxx xxx
x      x
xS     x
xxxxxxxx
"""


THREE_ROOMS = """
G=10.0
x=0.0
m=0.0
xxxxxxxxxx
xS       x
x        x
x        x
x        x
xxxxxx xxx
x        x
x        x
xx xxxxxxx
x        x
x        x
x        x
x       Gx
xxxxxxxxxx
"""

FOUR_ROOMS = """
G=1.0
r=1.0
g=-2.0
x=0.0
move=0.0
---
xxxxxxxxxxxxx
x     x    Gx
x     x     x
x           x
x     x     x
x     x     x
xxx xxx     x
x     xxx xxx
x     x     x
x           x
x     x     x
xS    x     x
xxxxxxxxxxxxx
"""

SMALL = """
G=10.0
x=-0.1
m=0.0
xxxx
x Gx
x xx
x Sx
xxxx
"""


if __name__ == "__main__":
    print(Blueprint.from_str(ROOMBA))
