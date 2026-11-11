#!/usr/bin/env python3
"""Download the default ResNet-18 backbone; optionally cache the LeRobot wheel."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def download(url, destination, expected_prefix):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and digest(destination).startswith(expected_prefix):
        print(f'Verified cached asset: {destination.name}', flush=True)
    else:
        temporary = destination.with_suffix(destination.suffix + '.part')
        print(f'Downloading {destination.name}', flush=True)
        urllib.request.urlretrieve(url, temporary)
        checksum = digest(temporary)
        if not checksum.startswith(expected_prefix):
            raise RuntimeError(f'Checksum mismatch for {destination.name}: {checksum}')
        temporary.replace(destination)
    return {'url': url, 'path': str(destination.relative_to(ROOT)),
            'sha256': digest(destination), 'bytes': destination.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-wheel', action='store_true',
                        help='Also cache a verified LeRobot wheel for local/offline reuse')
    args = parser.parse_args()
    assets = []
    if args.include_wheel:
        package = json.load(urllib.request.urlopen('https://pypi.org/pypi/lerobot/0.7.0/json', timeout=60))
        wheel = next(item for item in package['urls'] if item['filename'].endswith('.whl'))
        assets.append(download(wheel['url'], ROOT / 'vendor' / wheel['filename'], wheel['digests']['sha256']))
    assets.append(download('https://download.pytorch.org/models/resnet18-f37072fd.pth',
                           ROOT / '.cache/torch/hub/checkpoints/resnet18-f37072fd.pth', 'f37072fd'))
    (ROOT / 'reports').mkdir(exist_ok=True)
    (ROOT / 'reports/assets.json').write_text(json.dumps(assets, indent=2) + '\n')


if __name__ == '__main__':
    main()
