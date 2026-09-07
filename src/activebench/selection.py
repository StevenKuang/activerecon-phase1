"""Validated selection of cells in a frozen campaign."""


def add_selection_arguments(parser):
    parser.add_argument("--group", choices=["mesh", "gs"])
    parser.add_argument("--scene", help="Scene name, e.g. interior_0007 or apartment_1")
    parser.add_argument("--method", help="Method id, e.g. r3con-pano or random")
    parser.add_argument("--condition", choices=["d0", "dyn"], help="Static or dynamic; omit for both")
    parser.add_argument("--seed", type=int, help="Acquisition seed; frozen Phase 1 uses 0")
    parser.add_argument("--cell", action="append", help="Exact cell id; repeat to select several")


def select_cells(campaign, args, include_missing=False):
    all_cells = campaign["cells"]
    selected = [c for c in all_cells if (include_missing or c["status"] == "retained")
                and all(getattr(args, field, None) is None or c[field] == getattr(args, field)
                        for field in ("group", "scene", "method", "condition", "seed"))
                and (not args.cell or c["id"] in args.cell)]
    if not selected:
        raise ValueError("selection contains no eligible cells; check scene/method/condition (GLEAM is GS-only)")
    if args.cell and set(args.cell) - {c["id"] for c in selected}:
        raise ValueError("unknown, excluded or filtered-out cell requested")
    return selected
