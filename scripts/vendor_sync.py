"""Re-vendor projection math from a spatial monorepo checkout.

Usage:
    python scripts/vendor_sync.py /path/to/spatial

Copies the source functions byte-for-byte and stamps each file with the
spatial commit they came from, so SDK math is always traceable to the
production pipeline.
"""

import re
import subprocess
import sys
from pathlib import Path

VENDOR_DIR = Path(__file__).parent.parent / "src" / "cyvl" / "_vendor"

# (vendor file, source file, function names to extract — None = whole file)
SOURCES = [
    ("transform.py", "packages/projection/src/projection/transform.py", None),
    (
        "frustum.py",
        "packages/projection/src/projection/projection.py",
        ["create_xyz_point_mask"],
    ),
]


def extract_functions(source: str, names: list[str]) -> str:
    """Pull top-level function definitions (with decorators) out of a module."""
    blocks = []
    for name in names:
        match = re.search(
            rf"^def {name}\(.*?(?=^\S)", source, flags=re.DOTALL | re.MULTILINE
        )
        if match is None:
            raise SystemExit(f"Function {name} not found in source")
        blocks.append(match.group(0).rstrip() + "\n")
    return "\n\n".join(blocks)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    spatial = Path(sys.argv[1]).resolve()
    sha = subprocess.run(
        ["git", "-C", str(spatial), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    for vendor_name, source_rel, functions in SOURCES:
        source = (spatial / source_rel).read_text()
        if functions is not None:
            body = "import numpy as np\n\n\n" + extract_functions(source, functions)
            origin = f"{source_rel} ({', '.join(functions)})"
        else:
            body = source
            origin = source_rel
        header = (
            f"# Vendored from cyvl spatial monorepo:\n"
            f"#   {origin}\n"
            f"#   @ {sha}\n"
            f"# Do not edit by hand — regenerate with scripts/vendor_sync.py.\n"
        )
        (VENDOR_DIR / vendor_name).write_text(header + body)
        print(f"vendored {vendor_name} from {origin} @ {sha[:12]}")


if __name__ == "__main__":
    main()
