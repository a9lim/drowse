from __future__ import annotations

import argparse
import hashlib
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
SENTINEL = PurePosixPath("assets/stale-dashboard-sentinel.js")


def _dashboard_path(member: str) -> PurePosixPath | None:
    parts = PurePosixPath(member).parts
    for index in range(len(parts) - 2):
        if parts[index : index + 3] == ("drowse", "web", "dist"):
            relative = parts[index + 3 :]
            return PurePosixPath(*relative) if relative else None
    return None


def _wheel_dashboard_files(path: Path) -> dict[PurePosixPath, str]:
    with zipfile.ZipFile(path) as archive:
        return {
            relative: hashlib.sha256(archive.read(member)).hexdigest()
            for member in archive.infolist()
            if not member.is_dir() and (relative := _dashboard_path(member.filename)) is not None
        }


def _sdist_dashboard_files(path: Path) -> dict[PurePosixPath, str]:
    with tarfile.open(path) as archive:
        files: dict[PurePosixPath, str] = {}
        for member in archive.getmembers():
            relative = _dashboard_path(member.name)
            if not member.isfile() or relative is None:
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                raise SystemExit(f"could not read sdist member {member.name}")
            files[relative] = hashlib.sha256(extracted.read()).hexdigest()
        return files


def _one_archive(dist_dir: Path, pattern: str) -> Path:
    matches = sorted(dist_dir.glob(pattern))
    if len(matches) != 1:
        raise SystemExit(f"expected one {pattern} archive in {dist_dir}, found {len(matches)}")
    return matches[0]


def seed() -> None:
    dashboard_cache = ROOT / "build/lib/drowse/web/dist"
    sentinel = dashboard_cache / Path(SENTINEL)
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text("stale build cache sentinel\n", encoding="utf-8")
    (dashboard_cache / "index.html").write_text("stale cached dashboard\n", encoding="utf-8")


def check(dist_dir: Path) -> None:
    wheel_files = _wheel_dashboard_files(_one_archive(dist_dir, "*.whl"))
    sdist_files = _sdist_dashboard_files(_one_archive(dist_dir, "*.tar.gz"))
    source_root = ROOT / "drowse/web/dist"
    source_files = {
        PurePosixPath(path.relative_to(source_root).as_posix())
        for path in source_root.rglob("*") if path.is_file()
    }

    unexpected = sorted(wheel_files.keys() - sdist_files.keys())
    missing = sorted(sdist_files.keys() - wheel_files.keys())
    unpackaged = sorted(source_files - wheel_files.keys())
    content_mismatch = sorted(path for path in wheel_files.keys() & sdist_files.keys() if wheel_files[path] != sdist_files[path])
    absent_from_source = sorted(path for path in wheel_files if not (source_root / Path(path)).is_file())
    source_mismatch = sorted(
        path
        for path, digest in wheel_files.items()
        if (source := source_root / Path(path)).is_file() and hashlib.sha256(source.read_bytes()).hexdigest() != digest
    )
    if unexpected or missing or unpackaged or content_mismatch or absent_from_source or source_mismatch or SENTINEL in wheel_files:
        details = [
            *(f"wheel-only dashboard file: {path}" for path in unexpected),
            *(f"sdist-only dashboard file: {path}" for path in missing),
            *(f"dashboard source file missing from wheel: {path}" for path in unpackaged),
            *(f"wheel/sdist dashboard content mismatch: {path}" for path in content_mismatch),
            *(f"dashboard file absent from source: {path}" for path in absent_from_source),
            *(f"wheel/source dashboard content mismatch: {path}" for path in source_mismatch),
        ]
        raise SystemExit("package dashboard isolation failed\n" + "\n".join(details))

    print(f"package dashboard isolation verified ({len(wheel_files)} files)")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("seed")
    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("dist_dir", nargs="?", type=Path, default=Path("dist"))
    args = parser.parse_args()

    if args.command == "seed":
        seed()
    else:
        dist_dir = args.dist_dir if args.dist_dir.is_absolute() else ROOT / args.dist_dir
        check(dist_dir)


if __name__ == "__main__":
    main()
