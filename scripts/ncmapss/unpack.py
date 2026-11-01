"""Unpack N-CMAPSS with the minimum disk, and say up front what it will cost.

The archive is nested and hostile to partial reads:

    full.bin (15.76 GB)                 outer zip, one folder + one member
      data_set.zip (15.76 GB)           the member, stored with method=8 (deflate),
                                        sizes only in a zip64 extra field
        N-CMAPSS_DS01-005.h5            the files we actually want
        ...

Because the inner zip is deflated, its member headers are not reachable without inflating
everything before them. So the choice is either to inflate the whole 15.76 GB to disk first,
or to walk the local file headers of the inner stream and write only the members asked for.

This script does the second. It needs `full.bin` and roughly the size of the members you
select, instead of 32 GB.

Verified facts about the outer archive, read straight from its central directory
(100 KB tail via an HTTP range request, no full download needed):

    entries : 2
    [0] '17. Turbofan Engine Degradation Simulation Data Set 2/'   method=0 stored
    [1] '17. Turbofan Engine Degradation Simulation Data Set 2/data_set.zip'  method=8 deflate
    zip64   : yes; both sizes live in the 0x000A extra field

Usage:
    python scripts/ncmapss/unpack.py --list
    python scripts/ncmapss/unpack.py --members N-CMAPSS_DS02
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
import zlib
from pathlib import Path

DEFAULT = Path("/tmp/ncmapss/full.bin")
CHUNK = 1 << 22  # 4 MiB read size; inflate is the bottleneck, not IO


def walk_local_headers(fh, want=(), out_dir=None, log_every=20):
    """Stream a zip from a non-seekable position, writing only `want` members.

    Local file headers (PK\\x03\\x04) are read in order; each member's deflate stream is
    consumed in 4 MiB chunks and discarded unless the name matches. Names are known before
    the data because this archive carries no data descriptors (flag bit 3 clear).
    """
    written, seen = [], []
    t0 = time.time()
    while True:
        sig = fh.read(4)
        if len(sig) < 4 or sig != b"PK\x03\x04":
            break
        hdr = fh.read(26)
        (ver, flag, meth, mt, md, crc, csz, usz, nlen, elen) = struct.unpack("<HHHHHIIIHH", hdr)
        name = fh.read(nlen).decode("utf-8", "replace")
        fh.read(elen)
        if flag & 0x08:
            raise RuntimeError(
                f"{name!r} uses a data descriptor; sizes are not in the local header and "
                "this stream walker cannot delimit it. Fall back to inflating data_set.zip.")
        seen.append({"name": name, "method": meth, "csz": csz, "usz": usz})
        sink = None
        if out_dir is not None and any(w in name for w in want) and not name.endswith("/"):
            out_dir.mkdir(parents=True, exist_ok=True)
            sink = open(out_dir / Path(name).name, "wb")
        if meth == 0:
            left = csz
            while left:
                b = fh.read(min(CHUNK, left))
                if not b:
                    break
                if sink:
                    sink.write(b)
                left -= len(b)
        elif meth == 8:
            d = zlib.decompressobj(-15)
            left = csz
            produced = 0
            while left:
                b = fh.read(min(CHUNK, left))
                if not b:
                    break
                left -= len(b)
                try:
                    chunk = d.decompress(b)
                except zlib.error as e:
                    raise RuntimeError(f"inflate failed inside {name!r}: {e}") from e
                if sink and chunk:
                    sink.write(chunk)
                produced += len(chunk)
            if sink:
                tail = d.flush()
                if tail:
                    sink.write(tail)
                    produced += len(tail)
            if sink and produced != usz:
                raise RuntimeError(f"{name!r}: inflated {produced} bytes, header says {usz}")
        else:
            raise RuntimeError(f"{name!r}: unsupported method {meth}")
        if sink:
            sink.close()
            written.append({"name": name, "bytes": usz})
            print(f"  wrote {Path(name).name}  {usz/1e9:.3f} GB  "
                  f"({time.time() - t0:.0f}s elapsed)")
        elif len(seen) % log_every == 0:
            print(f"  scanned {len(seen)} members  ({time.time() - t0:.0f}s elapsed)")
    return seen, written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=str(DEFAULT))
    ap.add_argument("--members", nargs="*", default=[],
                    help="substrings of member names to extract; empty means list only")
    ap.add_argument("--out", default="/tmp/ncmapss/extracted")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    z = Path(a.zip)
    if not z.exists():
        sys.exit(f"missing {z}; run scripts/ncmapss/fetch.sh first")
    print(f"{z} = {z.stat().st_size} bytes")

    # step 1: outer zip -> data_set.zip, streamed, never fully on disk twice
    out = Path(a.out)
    with open(z, "rb") as fh:
        seen, written = walk_local_headers(fh, want=("data_set.zip",), out_dir=out)
    print(f"outer: {len(seen)} entries, wrote {written}")
    (out / "outer_manifest.json").write_text(json.dumps(seen, indent=1), encoding="utf-8")

    inner = out / Path(seen[-1]["name"]).name
    if not inner.exists():
        sys.exit("data_set.zip was not produced")

    # step 2: inner zip -> the members asked for
    print(f"\n{inner.name}: scanning for {a.members or '(list only)'}")
    with open(inner, "rb") as fh:
        seen2, written2 = walk_local_headers(
            fh, want=tuple(a.members), out_dir=out if a.members else None)
    for e in seen2:
        print(f"  {e['name']:44s} {e['usz']/1e9:7.3f} GB  method={e['method']}")
    (out / "inner_manifest.json").write_text(json.dumps(seen2, indent=1), encoding="utf-8")
    print(f"\nwrote {len(written2)} files to {out}")
    print("the intermediate data_set.zip can now be deleted to reclaim 15.76 GB")


if __name__ == "__main__":
    main()
