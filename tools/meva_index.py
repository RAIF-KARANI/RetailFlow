"""Index the public MEVA bucket (file names and sizes only, no video) -> outputs/meva_index.csv.

    python tools/meva_index.py

MEVA (Kitware / IARPA, CC BY 4.0, https://mevadata.org) is hosted in the public
S3 bucket `mevadata-public-01`. S3's ListObjectsV2 REST call works over plain
HTTPS without an account, so no AWS tool is needed. Clip names look like
    2018-03-07.16-50-00.16-55-00.admin.G329.r13.avi
    date      .start   .end     .site .camera
"""

from __future__ import annotations

import csv
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

BUCKET = "https://mevadata-public-01.s3.amazonaws.com"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
NAME = re.compile(r"(\d{4}-\d\d-\d\d)\.(\d\d-\d\d-\d\d)\.(\d\d-\d\d-\d\d)\.([\w-]+)\.(G\d+)")


def list_keys(prefix: str):
    token = None
    while True:
        q = {"list-type": "2", "prefix": prefix}
        if token:
            q["continuation-token"] = token
        with urllib.request.urlopen(f"{BUCKET}/?{urllib.parse.urlencode(q)}", timeout=60) as r:
            root = ET.fromstring(r.read())
        for c in root.findall("s3:Contents", NS):
            yield c.find("s3:Key", NS).text, int(c.find("s3:Size", NS).text)
        if root.find("s3:IsTruncated", NS).text != "true":
            return
        token = root.find("s3:NextContinuationToken", NS).text


def main() -> None:
    rows = []
    for key, size in list_keys("drops-123-r13/"):
        m = NAME.search(key)
        if not m or not key.endswith((".avi", ".mp4")):
            continue
        date, start, end, site, cam = m.groups()
        rows.append({"key": key, "mb": round(size / 1e6, 1), "date": date, "start": start,
                     "end": end, "site": site, "camera": cam})
    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/meva_index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} clips, {sum(r['mb'] for r in rows) / 1000:.0f} GB -> outputs/meva_index.csv")


if __name__ == "__main__":
    main()
