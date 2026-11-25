#!/bin/bash
# One-time (and after a Docker restart): build the images, the internal network and the proxy.
set -euo pipefail
export PATH=/Applications/Docker.app/Contents/Resources/bin:$PATH
cd "$(dirname "$0")"
docker build -q -t ar-proposer:2.1.326 .
docker build -q -t ar-egress:1 egress
docker network inspect ar-proposer-net >/dev/null 2>&1 || docker network create --internal ar-proposer-net
docker rm -f ar-egress >/dev/null 2>&1 || true
docker run -d --name ar-egress --restart unless-stopped --network bridge ar-egress:1 >/dev/null
docker network connect ar-proposer-net ar-egress
echo "egress proxy up"
