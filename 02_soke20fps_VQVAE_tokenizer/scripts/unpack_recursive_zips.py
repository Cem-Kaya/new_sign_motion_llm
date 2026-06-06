#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


REQUIRED_MARKERS = [
    Path("CSL-Daily-Fittings") / "csl_clean.train",
    Path("Neural-Sign-Actors") / "train_poses" / "poses",
    Path("Neural-Sign-Actors") / "val_poses" / "poses",
    Path("Neural-Sign-Actors") / "test_poses" / "poses",
    Path("phoenix_poses") / "train",
    Path("phoenix_poses") / "dev",
    Path("phoenix_poses") / "test",
]
CSL_POSE_MARKERS = [
    Path("CSL-Daily-Fittings") / "csl-daily_pose" / "csl-daily_pose",
    Path("CSL-Daily-Fittings") / "New folder" / "csl-daily_pose" / "csl-daily_pose",
    Path("CSL-Daily-Fittings") / "New folder" / "csl-daily_pose",
    Path("CSL-Daily-Fittings") / "csl-daily_pose",
]


def log(message: str) -> None:
    print(message, flush=True)


def default_workers() -> int:
    return max(1, min(4, os.cpu_count() or 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract SOKE_DATA ZIPs and nested ZIPs into a Colab-friendly data root."
    )
    parser.add_argument(
        "zip_root",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing CSL-Daily-Fittings.zip, Neural-Sign-Actors.zip, and phoenix_poses.zip.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Destination data root. Defaults to zip_root/..; use /content/SOKE_DATA on Colab for fast local extraction.",
    )
    parser.add_argument("-j", "--workers", type=int, default=default_workers())
    parser.add_argument(
        "--extractor",
        choices=("auto", "7z", "7zz", "unzip", "python"),
        default="auto",
        help="Extractor backend. auto prefers 7z/7zz, then unzip, then Python zipfile.",
    )
    parser.add_argument("--force", action="store_true", help="Remove and rebuild destination dataset folders.")
    parser.add_argument("--keep-nested-zips", action="store_true", help="Keep nested ZIP files after extraction.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_extractor(requested: str) -> str:
    if requested == "python":
        return "python"
    if requested == "auto":
        for candidate in ("7z", "7zz", "unzip"):
            executable = shutil.which(candidate)
            if executable:
                return executable
        return "python"
    executable = shutil.which(requested)
    if not executable:
        raise RuntimeError(f"Requested extractor '{requested}' was not found in PATH.")
    return executable


def top_level_zips(zip_root: Path) -> list[Path]:
    names = ["CSL-Daily-Fittings.zip", "Neural-Sign-Actors.zip", "phoenix_poses.zip"]
    found = [zip_root / name for name in names if (zip_root / name).exists()]
    if found:
        return found
    return sorted(p for p in zip_root.iterdir() if p.is_file() and p.suffix.lower() == ".zip")


def layout_ready(output_root: Path) -> bool:
    return all((output_root / marker).exists() for marker in REQUIRED_MARKERS) and any(
        (output_root / marker).exists() for marker in CSL_POSE_MARKERS
    )


def missing_layout_markers(output_root: Path) -> list[str]:
    missing = [str(marker) for marker in REQUIRED_MARKERS if not (output_root / marker).exists()]
    if not any((output_root / marker).exists() for marker in CSL_POSE_MARKERS):
        missing.append("one of: " + ", ".join(str(marker) for marker in CSL_POSE_MARKERS))
    return missing


def safe_extract_with_python(zip_path: Path, output_dir: Path) -> None:
    output_root = output_dir.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            member_name = member.filename.replace("\\", "/").lstrip("/")
            if not member_name:
                continue
            target = (output_dir / member_name).resolve()
            if not str(target).startswith(str(output_root)):
                raise RuntimeError(f"Unsafe path in {zip_path}: {member.filename}")
            if member_name.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)


def extract_zip(zip_path: Path, output_dir: Path, extractor: str, dry_run: bool) -> None:
    if dry_run:
        log(f"would extract: {zip_path} -> {output_dir}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    if extractor == "python":
        safe_extract_with_python(zip_path, output_dir)
    elif Path(extractor).name == "unzip":
        # Some source archives contain a harmless absolute root entry named "/".
        # Info-ZIP treats that as an error unless the entry is excluded.
        subprocess.run(["unzip", "-q", "-o", str(zip_path), "-d", str(output_dir), "-x", "/"], check=True)
    else:
        subprocess.run(
            [
                extractor,
                "x",
                "-y",
                "-aoa",
                "-mmt=on",
                "-bd",
                "-bso0",
                "-bsp0",
                f"-o{output_dir}",
                str(zip_path),
            ],
            check=True,
        )


def run_parallel(paths: list[Path], workers: int, task) -> None:
    if not paths:
        return
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(paths)))) as executor:
        futures = {executor.submit(task, path): path for path in paths}
        for future in as_completed(futures):
            path = futures[future]
            try:
                future.result()
            except Exception as exc:
                raise RuntimeError(f"Failed while processing {path}: {exc}") from exc


def extract_top_level(zips: list[Path], output_root: Path, extractor: str, workers: int, force: bool, dry_run: bool) -> list[Path]:
    output_dirs = [output_root / zip_path.stem for zip_path in zips]

    def task(zip_path: Path) -> None:
        output_dir = output_root / zip_path.stem
        if output_dir.exists() and force:
            if dry_run:
                log(f"would remove existing output: {output_dir}")
            else:
                log(f"remove existing output: {output_dir}")
                shutil.rmtree(output_dir)
        if output_dir.exists() and any(output_dir.iterdir()) and not force:
            log(f"reuse top-level output: {output_dir}")
            return
        log(f"extract top-level: {zip_path.name} -> {output_dir}")
        extract_zip(zip_path, output_dir, extractor, dry_run)

    run_parallel(zips, workers, task)
    return output_dirs


def iter_nested_zips(roots: list[Path], processed: set[Path]) -> list[Path]:
    nested: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                if not filename.lower().endswith(".zip"):
                    continue
                path = Path(dirpath) / filename
                resolved = path.resolve()
                if resolved not in processed:
                    nested.append(path)
    return sorted(nested)


def extract_nested_until_done(
    roots: list[Path],
    extractor: str,
    workers: int,
    keep_nested_zips: bool,
    dry_run: bool,
) -> None:
    processed: set[Path] = set()
    round_number = 1
    while True:
        nested_zips = iter_nested_zips(roots, processed)
        if not nested_zips:
            return
        log(f"nested round {round_number}: {len(nested_zips)} ZIP file(s)")

        def task(zip_path: Path) -> None:
            output_dir = zip_path.with_suffix("")
            if output_dir.exists() and any(output_dir.iterdir()):
                log(f"reuse nested output: {output_dir}")
            else:
                log(f"extract nested: {zip_path} -> {output_dir}")
                extract_zip(zip_path, output_dir, extractor, dry_run)
            if not keep_nested_zips:
                if dry_run:
                    log(f"would delete nested ZIP: {zip_path}")
                elif zip_path.exists():
                    zip_path.unlink()

        run_parallel(nested_zips, workers, task)
        processed.update(path.resolve() for path in nested_zips)
        round_number += 1


def main() -> int:
    args = parse_args()
    zip_root = args.zip_root.expanduser().resolve()
    output_root = (args.output_root if args.output_root is not None else zip_root.parent).expanduser().resolve()
    if args.workers < 1:
        raise RuntimeError("--workers must be at least 1.")
    if not zip_root.is_dir():
        raise RuntimeError(f"ZIP root does not exist: {zip_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    if layout_ready(output_root) and not args.force:
        log(f"SOKE_DATA layout already exists: {output_root}")
        return 0

    extractor = resolve_extractor(args.extractor)
    zips = top_level_zips(zip_root)
    if not zips:
        raise RuntimeError(f"No top-level ZIP files found in {zip_root}")

    log(f"zip_root: {zip_root}")
    log(f"output_root: {output_root}")
    log(f"extractor: {extractor}")
    log(f"workers: {args.workers}")
    output_dirs = extract_top_level(zips, output_root, extractor, args.workers, args.force, args.dry_run)
    extract_nested_until_done(output_dirs, extractor, args.workers, args.keep_nested_zips, args.dry_run)

    if not args.dry_run and not layout_ready(output_root):
        raise RuntimeError(
            f"Extraction finished, but expected layout is incomplete under {output_root}: {missing_layout_markers(output_root)}"
        )

    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
