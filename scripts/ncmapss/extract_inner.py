"""Extract members from the *inner* zip of N-CMAPSS without ever writing it to disk.

`unpack.py` stages data_set.zip (15.76 GB) and then re-reads it. This does both
inflations in one pass: the outer deflate stream is wrapped in a file-like reader,
and the inner zip walker consumes from that reader. Peak disk is therefore the
members requested, not the archive.

    full.bin  --inflate-->  data_set.zip bytes  --walk-->  N-CMAPSS_DSxx.h5

The outer archive carries no data descriptor (flag bit 3 is clear), so member sizes
are known from the local header before the data arrives and the walker can delimit
each member. `--list` does a full pass and writes the manifest without extracting;
combine `--list` with `--members` to do both in one pass.

Usage:
    python scripts/ncmapss/extract_inner.py --list
    python scripts/ncmapss/extract_inner.py --members DS02
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
import zlib
from pathlib import Path

OUTER_DATA_OFFSET = 200          # 84 + 30 + 66 (name) + 20 (zip64 extra), verified
OUTER_CSZ = 15_760_442_787
CHUNK = 1 << 22


class InflatingReader:
    """A minimal buffered file-like object over a deflate stream.

    Only `read(n)` is needed by the walker, and only forward reads happen, so this
    keeps one buffer and tops it up on demand. `read(n)` returns fewer than n bytes
    only at end of stream.
    """

    def __init__(self, fh, data_offset, csz, chunk=CHUNK):
        self.fh = fh
        self.left = csz
        self.chunk = chunk
        self.d = zlib.decompressobj(-15)
        self.buf = bytearray()
        fh.seek(data_offset)
        self.consumed = 0
        self.produced = 0

    def _pump(self) -> bool:
        if self.left <= 0:
            return False
        b = self.fh.read(min(self.chunk, self.left))
        if not b:
            self.left = 0
            return False
        self.left -= len(b)
        self.consumed += len(b)
        out = self.d.decompress(b)
        self.produced += len(out)
        self.buf += out
        return True

    def read(self, n: int) -> bytes:
        while len(self.buf) < n and self._pump():
            pass
        out = bytes(self.buf[:n])
        del self.buf[:n]
        return out


def walk(reader, want=(), out_dir=None, log_every=1):
    """Walk local file headers of a zip stream, writing only names containing `want`."""
    seen, written = [], []
    t0 = time.time()
    while True:
        sig = reader.read(4)
        if len(sig) < 4 or sig != b"PK\x03\x04":
            break
        hdr = reader.read(26)
        if len(hdr) < 26:
            break
        (ver, flag, meth, mt, md, crc, csz, usz, nlen, elen) = struct.unpack("<HHHHHIIIHH", hdr)
        name = reader.read(nlen).decode("utf-8", "replace")
        reader.read(elen)
        if flag & 0x08:
            raise RuntimeError(f"{name!r} uses a data descriptor; cannot delimit it")
        rec = {"name": name, "method": meth, "csz": csz, "usz": usz, "crc": f"{crc:08x}"}
        seen.append(rec)
        print(f"  [{len(seen):2d}] {name:44s} {usz/1e9:7.3f} GB  method={meth}  "
              f"({time.time()-t0:.0f}s)")

        take = out_dir is not None and any(w in name for w in want) and not name.endswith("/")
        sink = open(out_dir / Path(name).name, "wb") if take else None
        if take:
            out_dir.mkdir(parents=True, exist_ok=True)

        if meth == 0:
            left = csz
            while left:
                b = reader.read(min(CHUNK, left))
                if not b:
                    break
                left -= len(b)
                if sink:
                    sink.write(b)
        elif meth == 8:
            d = zlib.decompressobj(-15)
            left = csz
            produced = 0
            crc_run = 0
            while left:
                b = reader.read(min(CHUNK, left))
                if not b:
                    break
                left -= len(b)
                out = d.decompress(b)
                if sink and out:
                    sink.write(out)
                produced += len(out)
                if sink:
                    crc_run = zlib.crc32(out, crc_run)
            if sink:
                tail = d.flush()
                if tail:
                    sink.write(tail)
                    produced += len(tail)
                    crc_run = zlib.crc32(tail, crc_run)
                sink.close()
                if produced != usz:
                    raise RuntimeError(f"{name!r}: inflated {produced}, header says {usz}")
                if f"{crc_run:08x}" != f"{crc:08x}":
                    raise RuntimeError(f"{name!r}: CRC {crc_run:08x} != header {crc:08x}")
            rec["verified"] = bool(sink) and produced == usz
        else:
            raise RuntimeError(f"{name!r}: unsupported method {meth}")
        if sink:
            written.append(rec)
            print(f"       wrote {Path(name).name} {usz/1e9:.3f} GB  verified CRC+size")
    return seen, written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="/tmp/ncmapss/full.bin")
    ap.add_argument("--members", nargs="*", default=[])
    ap.add_argument("--out", default="/tmp/ncmapss/extracted")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    z = Path(a.zip)
    if not z.exists():
        sys.exit(f"missing {z}")
    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    with open(z, "rb") as fh:
        r = InflatingReader(fh, OUTER_DATA_OFFSET, OUTER_CSZ)
        seen, written = walk(r, want=tuple(a.members), out_dir=out if a.members else None)
    dt = time.time() - t0
    print(f"\n{len(seen)} members, {len(written)} extracted, "
          f"{r.produced/1e9:.2f} GB inflated in {dt:.0f}s "
          f"({r.produced/1e6/max(dt,1e-9):.0f} MB/s)")
    (out / "inner_manifest.json").write_text(json.dumps(seen, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
