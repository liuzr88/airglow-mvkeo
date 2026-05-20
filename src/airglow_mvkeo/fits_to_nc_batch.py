#!/usr/bin/env python3
"""Batch runner for airglow FITS-to-NetCDF conversion."""

from __future__ import annotations

import argparse
import platform
import subprocess
from pathlib import Path
from typing import Iterable

from .fits_to_nc import convert_year


DEFAULT_CHANNELS = ("OH", "O5", "O2", "O6", "bias", "dark", "Na")


def is_mounted(path: Path) -> bool:
    path = path.resolve()
    return any(Path(partition).resolve() == path for partition in mounted_paths())


def mounted_paths() -> set[str]:
    if platform.system() == "Windows":
        return set()
    paths: set[str] = set()
    with Path("/proc/mounts").open("r", encoding="utf-8") if Path("/proc/mounts").exists() else open(
        "/dev/null", "r", encoding="utf-8"
    ) as mounts:
        for line in mounts:
            fields = line.split()
            if len(fields) >= 2:
                paths.add(fields[1].replace("\\040", " "))

    if platform.system() == "Darwin":
        result = subprocess.run(["mount"], check=True, text=True, capture_output=True)
        for line in result.stdout.splitlines():
            marker = " on "
            if marker not in line:
                continue
            rest = line.split(marker, 1)[1]
            paths.add(rest.split(" (", 1)[0])
    return paths


def normalize_smb_share(share: str) -> str:
    if share.startswith("smb://"):
        return "//" + share[len("smb://") :]
    return share


def ensure_mounted(args: argparse.Namespace) -> None:
    if not args.mount_if_needed:
        return
    if args.mount_point is None or args.nas_share is None:
        raise ValueError("--mount-if-needed requires --nas-share and --mount-point")

    mount_point = args.mount_point
    mount_point.mkdir(parents=True, exist_ok=True)
    if is_mounted(mount_point):
        print(f"NAS already mounted at {mount_point}")
        return

    share = normalize_smb_share(args.nas_share)
    system = platform.system()

    if system == "Darwin":
        # macOS will use Keychain or prompt interactively when credentials are
        # not already available. Prefer smb://user@server/share in that case.
        command = ["mount_smbfs", share, str(mount_point)]
    elif system == "Linux":
        command = ["mount", "-t", "cifs", share, str(mount_point)]
        options = []
        if args.cifs_credentials:
            options.append(f"credentials={args.cifs_credentials}")
        if args.cifs_options:
            options.append(args.cifs_options)
        if options:
            command.extend(["-o", ",".join(options)])
    else:
        raise RuntimeError(f"Automatic mounting is not supported on {system}")

    print(f"Mounting {share} at {mount_point}")
    subprocess.run(command, check=True)
    if not is_mounted(mount_point):
        raise RuntimeError(f"Mount command finished, but {mount_point} is not mounted")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("years", nargs="+", type=int, help="One or more years to convert, e.g. 2023 2024.")
    parser.add_argument(
        "--channels",
        nargs="+",
        default=DEFAULT_CHANNELS,
        help="Airglow channels to convert.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root containing year folders, e.g. /Volumes/NAS/ALOASI.",
    )
    parser.add_argument(
        "--nc-root",
        type=Path,
        required=True,
        help="Root where NetCDF year folders should be written, e.g. /Volumes/NAS/ALOASI/NC.",
    )
    parser.add_argument("--overwrite", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--legacy-int16-ceiling",
        action="store_true",
        help="Reproduce older outputs where unsigned 16-bit FITS values were capped at 32767.",
    )
    parser.add_argument(
        "--mount-if-needed",
        action="store_true",
        help="Mount an SMB/CIFS NAS share before conversion if the mount point is not already mounted.",
    )
    parser.add_argument(
        "--nas-share",
        help="NAS SMB/CIFS share, e.g. //server/share or smb://user@server/share.",
    )
    parser.add_argument(
        "--mount-point",
        type=Path,
        help="Local mount point, e.g. /Volumes/ALOASI_NAS or /mnt/aloasi_nas.",
    )
    parser.add_argument(
        "--cifs-credentials",
        type=Path,
        help="Linux only: path to a root-readable CIFS credentials file.",
    )
    parser.add_argument(
        "--cifs-options",
        help="Linux only: extra comma-separated CIFS options, e.g. uid=1000,gid=1000,vers=3.0.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    ensure_mounted(args)
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"Data root is not mounted or does not exist: {args.data_root}")
    args.nc_root.mkdir(parents=True, exist_ok=True)

    for year in args.years:
        for channel in args.channels:
            print(f"\n=== {year} {channel} ===")
            convert_year(
                year=year,
                airglow_flag=channel,
                overwrite=bool(args.overwrite),
                data_root=args.data_root,
                nc_root=args.nc_root,
                legacy_int16_ceiling=args.legacy_int16_ceiling,
            )


if __name__ == "__main__":
    main()
