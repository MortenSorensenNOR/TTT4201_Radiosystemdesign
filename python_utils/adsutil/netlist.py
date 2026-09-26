"""Edit ADS netlists as text.

VAR blocks become top-level ``name=value`` lines in the netlist, so changing a
variable is a line rewrite. Only top-level statements are touched; anything
inside ``define ... end`` subcircuits is left alone.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

_VAR_RE = re.compile(r"^([A-Za-z_]\w*(?:\([^)]*\))?)=(.*)$")


def fmt(value) -> str:
    """Format a Python value as an ADS expression."""
    if isinstance(value, str):
        return value
    if isinstance(value, complex):
        return f"({value.real:.12g}{value.imag:+.12g}*j)"
    return f"{value:.12g}"


class Netlist:
    def __init__(self, text: str):
        # one entry per statement, continuation lines ("\" at end) kept together
        self._stmts: list[str] = []
        buf = ""
        for line in text.splitlines():
            buf = f"{buf}\n{line}" if buf else line
            if not line.rstrip().endswith("\\"):
                self._stmts.append(buf)
                buf = ""
        if buf:
            self._stmts.append(buf)

    @classmethod
    def from_file(cls, path: str | Path) -> "Netlist":
        return cls(Path(path).read_text())

    @property
    def text(self) -> str:
        return "\n".join(self._stmts) + "\n"

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.text)

    def copy(self) -> "Netlist":
        return Netlist(self.text)

    # -- top-level statements -------------------------------------------------

    def _top_level(self) -> Iterable[int]:
        return self._in_scope(None)

    def _in_scope(self, subcircuit: str | None) -> Iterable[int]:
        """Statement indices at top level (None) or inside ``define <subcircuit>``."""
        scope = []
        for i, s in enumerate(self._stmts):
            head = s.lstrip()
            if head.startswith("define "):
                scope.append(head.split()[1])
            elif head.startswith("end ") and scope:
                scope.pop()
            elif (scope[-1] if scope else None) == subcircuit:
                yield i

    def _find_var(self, name: str) -> int | None:
        for i in self._top_level():
            m = _VAR_RE.match(self._stmts[i])
            if m and m.group(1) == name:
                return i
        return None

    # -- variables ------------------------------------------------------------

    @property
    def variables(self) -> dict[str, str]:
        out = {}
        for i in self._top_level():
            m = _VAR_RE.match(self._stmts[i])
            if m:
                out[m.group(1)] = m.group(2)
        return out

    def get(self, name: str) -> str:
        i = self._find_var(name)
        if i is None:
            raise KeyError(f"variable {name!r} not in netlist")
        return _VAR_RE.match(self._stmts[i]).group(2)

    def set(self, name: str, value, create: bool = False) -> None:
        i = self._find_var(name)
        line = f"{name}={fmt(value)}"
        if i is not None:
            self._stmts[i] = line
        elif create:
            self._stmts.append(line)
        else:
            raise KeyError(f"variable {name!r} not in netlist (pass create=True to add it)")

    def update(self, values: dict, create: bool = False) -> None:
        for k, v in values.items():
            self.set(k, v, create=create)

    # -- instance parameters ------------------------------------------------------

    def set_instance_param(self, instance: str, param: str, value, subcircuit: str | None = None) -> None:
        """Change one parameter on a component line, e.g. ("PORT1", "P[1]", ...).

        ``value`` replaces the whole value including units, e.g. "11 mm".
        Use ``subcircuit`` for components inside a ``define`` block.
        """
        head = re.compile(rf"^\S+:{re.escape(instance)}\s")
        # a value is one token, optionally followed by a unit ("11 mm")
        value_re = r"\S+(?:\s+(?:[a-zA-Z]+)(?=\s|$))?"
        for i in self._in_scope(subcircuit):
            if head.match(self._stmts[i]):
                new, n = re.subn(rf"(\s{re.escape(param)}=){value_re}",
                                 lambda m: m.group(1) + fmt(value), self._stmts[i], count=1)
                if not n:
                    raise KeyError(f"{instance} has no parameter {param!r}")
                self._stmts[i] = new
                return
        raise KeyError(f"no instance {instance!r} in {subcircuit or 'top level'}")

    # -- sweeps -----------------------------------------------------------------

    def remove_sweeps(self) -> None:
        """Drop every ParamSweep and its SweepPlan so analyses run at one point."""
        plans = set()
        keep = []
        for s in self._stmts:
            if s.startswith("ParamSweep:"):
                m = re.search(r'SweepPlan="([^"]+)"', s)
                if m:
                    plans.add(m.group(1))
                continue
            keep.append(s)
        self._stmts = [
            s for s in keep
            if not (m := re.match(r"SweepPlan:\s*(\S+)", s)) or m.group(1) not in plans
        ]

    def sweep_lists(self, analysis: str, lists: dict[str, Sequence], index: str = "py_i") -> int:
        """Sweep several variables together, one value per point.

        Each variable becomes ``name=name__list[index]`` and one ParamSweep over
        ``index`` (1..N) drives ``analysis``. Returns N. ADS lists are 1-based.
        """
        lengths = {len(v) for v in lists.values()}
        if len(lengths) != 1:
            raise ValueError(f"all lists must be the same length, got {lengths}")
        n = lengths.pop()
        self.set(index, 1, create=True)
        for name, values in lists.items():
            self.set(f"{name}__list", "list(" + ",".join(fmt(v) for v in values) + ")", create=True)
            self.set(name, f"{name}__list[{index}]", create=True)
        self._stmts.append(
            f'ParamSweep:py_sweep SimInstanceName[1]="{analysis}" SweepVar="{index}" SweepPlan="py_sweep_stim"'
        )
        self._stmts.append(f"SweepPlan: py_sweep_stim Start=1 Stop={n} Step=1")
        return n
