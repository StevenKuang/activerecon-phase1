"""Serve an already-exported web-demo bundle over HTTP with byte-range support.

Paged RAD reconstructions are streamed with HTTP range requests, so a plain
``python -m http.server`` silently breaks them: it ignores the ``Range``
header, answers 200 with the whole file, and the splat layer never renders
(the replay and point cloud still show, which makes it look like the LOD
build was skipped). ``export_web_demo.py --serve`` uses the right handler,
by running a full export first. This script serves the existing bundle.

    python scripts/serve_bundle.py exports/campaign_v6/interior_0007_dyn_compare
    python scripts/serve_bundle.py <bundle> --port 8091
"""

import argparse
import functools
import http.server
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from activebench.web_live import RangeRequestHandler  # noqa: E402

DEFAULT_PORT = 8090


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path,
                        help="bundle directory (the one holding index.html)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    bundle = args.bundle.resolve()
    if not (bundle / "index.html").exists():
        parser.error("%s: missing index.html; serve a directory produced by the exporter" % bundle)

    handler = functools.partial(RangeRequestHandler, directory=str(bundle))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print("serving %s at http://127.0.0.1:%d (Ctrl-C to stop)" % (bundle, args.port))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
