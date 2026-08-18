"""The live-config census: which fingerprinted fields reach something that acts.

**Reference-counting would have passed `rehearsal_frac`.** It is read — inside
`rehearsal_split`, which `train_on_ring` never calls. So the question is not
*is this field mentioned* but *is the site that mentions it reachable from a
production entry point*, and that is what this computes.

Method
------
1. Enumerate every field of `Config`, recursively, at runtime — the fingerprint
   is `yaml.safe_dump` of exactly this tree, so this IS the fingerprinted set.
2. AST-parse `src/reckoner` and `scripts`. For each function, record the config
   fields it reads and the functions it calls.
3. Compute the reachable set from production entry points by fixpoint. **Tests
   are not entry points** — that is the whole point: four passing assertions on a
   function nobody calls are not evidence that a field is consumed.
4. A field is LIVE if some reading site sits in a reachable function.

Two honest limits, stated rather than hidden:

* **Name-based call resolution.** A call to `foo()` binds to every function named
  `foo`. This over-connects the graph, so the error runs toward calling a dead
  field live — the census UNDER-reports. A field it calls dead is dead.
* **Attribute matching** prefers the dotted form (`cfg.train.rehearsal_frac`) and
  falls back to the bare attribute name, which can collide (`.seed`). Same
  direction: collisions mark fields live, never dead.

Both limits are conservative toward the answer that requires no action, which is
the direction a census of this kind must err in to be worth running.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import json
from collections import defaultdict
from pathlib import Path

from reckoner.config import Config

REPO = Path(__file__).resolve().parents[1]
SOURCES = (REPO / "src" / "reckoner", REPO / "scripts")
#: SPECIFICATION documents only. `FINDINGS.md` is deliberately excluded: it
#: records what was found, and a finding that *names* a dead key would otherwise
#: mark that key spec-backed — so writing about the problem would reclassify it
#: as a defect fix. The distinction the column exists to draw is whether a PAGE
#: ASKED FOR the field, and only these pages ask for anything.
DOCS = tuple(
    d for d in REPO.glob("*.md") if d.name.startswith(("PREREG", "BRIEF", "plan", "SWEEP"))
)

#: Readers whose reads are NOT consumption. A range check, a fingerprint dump or
#: a yaml round-trip touches every field by construction — counting them would
#: make every field live and the census vacuous. This is the two-tier rule at its
#: root: the minimum bar is that a value reaches something that ACTS on it, and
#: validating a value is not acting on it.
#:
#: Acceptance test for this census: it must report `train.rehearsal_frac` dead.
#: Before this exclusion it did not, because `validate()` range-checks it — which
#: is precisely the near-miss that makes the distinction worth encoding.
GUARD_READERS = frozenset(
    {"validate", "config_fingerprint", "_dataclass_to_dict", "save_config", "load_config"}
)

#: Production entry points. `main` covers every script; the driver's two doors
#: and the training/eval surface cover the library paths a script reaches.
ENTRY_POINTS = frozenset(
    {"main", "run", "run_campaign", "preflight", "run_instruments", "run_iteration"}
)


def config_fields() -> dict[str, str]:
    """``{field_name: group}`` for every fingerprinted field, recursively."""
    out: dict[str, str] = {}
    cfg = Config()
    for group in dataclasses.fields(cfg):
        sub = getattr(cfg, group.name)
        if dataclasses.is_dataclass(sub):
            for f in dataclasses.fields(sub):
                out[f.name] = group.name
        else:
            out[group.name] = "(top level)"
    return out


class Walker(ast.NodeVisitor):
    """Per-function: which config fields are read, which functions are called."""

    def __init__(self, fields: dict[str, str]) -> None:
        self.fields = fields
        self.reads: dict[str, set[str]] = defaultdict(set)
        self.calls: dict[str, set[str]] = defaultdict(set)
        self.methods: dict[str, set[str]] = defaultdict(set)
        self.scope = "<module>"

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        """Instantiating a class calls its ``__init__``, and the AST records a
        call to the CLASS name. Without this the model's own constructor —
        `Reckoner(cfg)`, which reads d_model, n_heads, n_layers and the rest —
        reads as unreachable, and the census calls seven live fields dead."""
        for child in node.body:
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                self.methods[node.name].add(child.name)
        outer, self.scope = self.scope, node.name
        self.generic_visit(node)
        self.scope = outer

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        outer, self.scope = self.scope, node.name
        self.generic_visit(node)
        self.scope = outer

    visit_AsyncFunctionDef = visit_FunctionDef  # noqa: N815

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        if node.attr in self.fields:
            self.reads[node.attr].add(self.scope)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        func = node.func
        name = getattr(func, "id", None) or getattr(func, "attr", None)
        if name:
            self.calls[self.scope].add(name)
        self.generic_visit(node)


def build() -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, set[str]], set[str]]:
    """``(library_reads, script_reads, calls, defined)``.

    Library and script reads are separated because **reporting a value is not
    acting on it.** `shakedown.py` copies `rehearsal_frac` into its results JSON;
    that is provenance, and counting it as consumption would have hidden F-31
    exactly as the guard reads did. Behaviour for a fingerprinted campaign field
    lives in `src/reckoner`; a field only ever read by a script is reported, not
    consumed.
    """
    fields = config_fields()
    reads: dict[str, set[str]] = defaultdict(set)
    script_reads: dict[str, set[str]] = defaultdict(set)
    calls: dict[str, set[str]] = defaultdict(set)
    methods: dict[str, set[str]] = defaultdict(set)
    defined: set[str] = set()

    for root in SOURCES:
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    defined.add(node.name)
            w = Walker(fields)
            w.visit(tree)
            in_library = root.name == "reckoner"
            for field, scopes in w.reads.items():
                target = reads if in_library else script_reads
                target[field] |= scopes - GUARD_READERS
            for scope, names in w.calls.items():
                calls[scope] |= names
            for cls, names in w.methods.items():
                methods[cls] |= names
                defined.add(cls)

    # A call to a class name reaches that class's methods: construction runs
    # __init__, and an object built by reachable code has reachable methods.
    for cls, names in methods.items():
        calls[cls] |= names
    return reads, script_reads, calls, defined


def reachable(calls: dict[str, set[str]], defined: set[str]) -> set[str]:
    """Fixpoint from the entry points. Module scope is always reachable."""
    seen = {"<module>"} | (ENTRY_POINTS & defined)
    frontier = list(seen)
    while frontier:
        current = frontier.pop()
        for callee in calls.get(current, ()):
            if callee in defined and callee not in seen:
                seen.add(callee)
                frontier.append(callee)
    return seen


def spec_backing(field: str) -> list[str]:
    """Documents that name this field. A dead key the spec SPECIFIES is a defect
    fix; a dead key no page asked for is a change, and wiring it by default would
    alter behaviour nobody registered."""
    return sorted(d.name for d in DOCS if field in d.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "runs" / "config_census.json")
    args = parser.parse_args()

    fields = config_fields()
    reads, script_reads, calls, defined = build()
    live_scopes = reachable(calls, defined)

    rows = []
    for field, group in sorted(fields.items()):
        scopes = reads.get(field, set())
        reaching = scopes & live_scopes
        reported = sorted(script_reads.get(field, set()))
        if reaching:
            status = "LIVE"
        elif scopes:
            status = "DEAD"  # read in the library, from unreachable code
        elif reported:
            status = "REPORTED"  # scripts copy it into artifacts; nothing acts
        else:
            status = "UNREAD"
        rows.append(
            {
                "field": f"{group}.{field}",
                "status": status,
                "library_readers": sorted(scopes),
                "reachable_readers": sorted(reaching),
                "reported_in_scripts": reported,
                "spec_backing": spec_backing(field),
            }
        )

    # The detector's reference-vector check lives in
    # `tests/test_config_liveness.py`, against a SYNTHETIC known-dead field.
    # It was originally asserted here against `train.rehearsal_frac`, the field
    # F-31 proved dead by hand — and then this round wired that field, so the
    # reference vector moved and the assertion fired on its own success. A
    # detector validated against a live repo field is validated against a moving
    # target; the synthetic case cannot be fixed out from under it.

    dead = [r for r in rows if r["status"] != "LIVE"]
    print(f"\n  CONFIG CENSUS — {len(rows)} fingerprinted fields\n")
    print(f"    LIVE  : {len(rows) - len(dead)}")
    print(
        f"    DEAD  : {sum(1 for r in dead if r['status'] == 'DEAD')}  (read, but only from unreachable code)"
    )
    print(
        f"    REPORTED: {sum(1 for r in dead if r['status'] == 'REPORTED')}  (scripts copy it to artifacts; nothing acts on it)"
    )
    print(f"    UNREAD: {sum(1 for r in dead if r['status'] == 'UNREAD')}  (no read site at all)")
    if dead:
        print(f"\n  {'field':<38} {'status':<9} {'spec-backed':<12} read in")
        for r in dead:
            backing = "YES" if r["spec_backing"] else "no"
            where = ", ".join(r["library_readers"] or r["reported_in_scripts"]) or "-"
            print(f"    {r['field']:<38} {r['status']:<9} {backing:<12} {where}")
        print(
            "\n  spec-backed dead key  -> DEFECT FIX (the page specifies it; the wire is missing)"
        )
        print("  unbacked dead key     -> CHANGE (wiring it alters behaviour no page asked for):")
        print("                           delete or amend, never wire-by-default")
    args.out.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
