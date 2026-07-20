#!/usr/bin/env bash
# Stamp a build id into a demo index.html before uploading it to an HF Space.
#
# Why: the version footer used to read the X-Repo-Commit response header, which
# HF emits only from its STATIC file host. The native C++ Space serves the same
# UI from its own app server (/ui/), so that header never arrives and the
# footer always showed "(local dev)". Stamping the id into the document itself
# works on both Spaces and locally.
#
# Usage: tools/stamp_build.sh <index.html> [build-id]
# Default build id: <short-sha>[-dirty] from the repo containing this script.
set -euo pipefail

f=${1:?usage: stamp_build.sh <index.html> [build-id]}
[ -f "$f" ] || { echo "no such file: $f" >&2; exit 1; }

if [ $# -ge 2 ]; then
  id=$2
else
  here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  id=$(git -C "$here" rev-parse --short=8 HEAD)
  git -C "$here" diff --quiet || id="${id}-dirty"
fi

# Stamp <meta name="build-id"> AND the ?v= cache-busting query on the app
# script. Both must move together: leaving ?v= stale means browsers keep
# serving the previous app-*.js from cache, so a deploy silently has no effect
# for anyone who visited before (this is exactly how a shipped fix appeared
# not to work).
python3 - "$f" "$id" <<'PY'
import re, sys
path, bid = sys.argv[1], sys.argv[2]
s = open(path, encoding="utf-8").read()
s, n = re.subn(r'(<meta name="build-id" content=")[^"]*(">)',
               lambda m: m.group(1) + bid + m.group(2), s, count=1)
if not n:
    sys.exit('no <meta name="build-id"> found in ' + path)
s, nv = re.subn(r'(src="app-[a-z]+\.js)\?v=[^"]*(")',
                lambda m: m.group(1) + "?v=" + bid + m.group(2), s)
if not nv:
    print(f"warning: no app-*.js?v= query found in {path}", file=sys.stderr)
open(path, "w", encoding="utf-8").write(s)
print(f"stamped {path} -> {bid} (meta + {nv} cache-bust query)")
PY
