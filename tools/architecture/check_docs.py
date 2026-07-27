#!/usr/bin/env python3
"""Validate repository documentation links and index coverage."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit


IGNORED_DIRECTORIES = {
    ".agents",
    ".claude",
    ".codegraph",
    ".git",
    ".omo",
    ".omx",
    ".pytest_cache",
    ".vscode",
    "build",
    "build-local",
    "out",
    "third_party",
}

TEXT_SUFFIXES = {
    ".cc",
    ".cmake",
    ".cpp",
    ".h",
    ".hpp",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".txt",
}

MARKDOWN_LINK = re.compile(
    r"!?\[[^\]\n]*\]\((?P<target><[^>\n]+>|[^)\n]+)\)"
)
REFERENCE_LINK = re.compile(
    r"^\s*\[[^\]\n]+\]:\s*(?P<target><[^>\n]+>|\S+)",
    re.MULTILINE,
)
DOC_PATH_REFERENCE = re.compile(r"docs[\\/][A-Za-z0-9_./\\-]+\.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check local Markdown links and docs/README.md coverage."
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Repository root. Defaults to the current directory.",
    )
    return parser.parse_args()


def is_ignored(path: Path, root: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return True
    return any(part in IGNORED_DIRECTORIES for part in relative.parts)


def markdown_files(root: Path) -> list[Path]:
    candidates: list[Path] = []
    root_readme = root / "README.md"
    if root_readme.is_file():
        candidates.append(root_readme)
    docs = root / "docs"
    if docs.is_dir():
        candidates.extend(path for path in docs.rglob("*.md") if path.is_file())
    tools = root / "tools"
    if tools.is_dir():
        candidates.extend(path for path in tools.rglob("*.md") if path.is_file())
    return sorted({path.resolve() for path in candidates if not is_ignored(path, root)})


def destination(raw_target: str) -> str:
    value = raw_target.strip()
    if value.startswith("<") and value.endswith(">"):
        return value[1:-1].strip()
    return value.split(maxsplit=1)[0]


def resolve_local_link(source: Path, raw_target: str, root: Path) -> Path | None:
    target = destination(raw_target)
    if not target or target.startswith("#"):
        return None
    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc:
        return None
    path_text = unquote(parsed.path)
    if not path_text:
        return None
    candidate = Path(path_text)
    if candidate.is_absolute():
        raise ValueError("absolute local paths are not allowed")
    return (source.parent / candidate).resolve()


def iter_text_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file() or is_ignored(path, root):
            continue
        if path.name == "CMakeLists.txt" or path.suffix.lower() in TEXT_SUFFIXES:
            yield path


def main() -> int:
    root = Path(parse_args().root).resolve()
    errors: list[str] = []
    sources = markdown_files(root)

    if not (root / "README.md").is_file():
        errors.append("README.md: missing repository documentation entry point")
    if not (root / "docs" / "ARCHITECTURE.md").is_file():
        errors.append("docs/ARCHITECTURE.md: missing architecture authority")
    if not (root / "docs" / "README.md").is_file():
        errors.append("docs/README.md: missing documentation index")

    index_targets: set[Path] = set()
    index = (root / "docs" / "README.md").resolve()
    for source in sources:
        try:
            text = source.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            errors.append(
                f"{source.relative_to(root)}: documentation is not valid UTF-8: {error}"
            )
            continue
        matches = list(MARKDOWN_LINK.finditer(text))
        matches.extend(REFERENCE_LINK.finditer(text))
        for match in matches:
            raw_target = match.group("target")
            try:
                resolved = resolve_local_link(source, raw_target, root)
            except (OSError, ValueError) as error:
                errors.append(
                    f"{source.relative_to(root)}: invalid local link "
                    f"{destination(raw_target)!r}: {error}"
                )
                continue
            if resolved is None:
                continue
            if not resolved.exists():
                errors.append(
                    f"{source.relative_to(root)}: missing local link target "
                    f"{destination(raw_target)!r}"
                )
            if source == index:
                index_targets.add(resolved)

    docs_root = root / "docs"
    if docs_root.is_dir() and index.is_file():
        for path in sorted(docs_root.rglob("*.md")):
            resolved = path.resolve()
            if resolved == index:
                continue
            if resolved not in index_targets:
                errors.append(
                    f"docs/README.md: does not index {path.relative_to(root)}"
                )

    for path in iter_text_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for match in DOC_PATH_REFERENCE.finditer(text):
            referenced = match.group(0).replace("\\", "/")
            candidate = (root / referenced).resolve()
            if not candidate.is_file():
                errors.append(
                    f"{path.relative_to(root)}: references missing {referenced}"
                )

    if errors:
        for error in sorted(set(errors)):
            print(f"ERROR: {error}")
        print(f"Documentation check failed: {len(set(errors))} issue(s).")
        return 1

    print(
        "Documentation check passed: "
        f"{len(sources)} Markdown file(s), no broken local links or stale docs paths."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
