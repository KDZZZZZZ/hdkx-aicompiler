from __future__ import annotations

import argparse
from pathlib import Path

from .importer import import_onnx
from .spec import save_imported_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a KXC ONNX import spec and params file.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()

    imported = import_onnx(args.model, default_batch=args.batch)
    save_imported_model(imported, args.json, args.params)


if __name__ == "__main__":
    main()
