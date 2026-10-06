#!/usr/bin/env python3
"""Apply the attached functional-folder layout after validating known file hashes."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('repo', type=Path)
p.add_argument('--check', action='store_true', help='validate only; do not write files')
a = p.parse_args()
repo = a.repo.resolve()
package = Path(__file__).resolve().parent
manifest = json.loads((package / 'manifest.json').read_text())
if not (repo / 'research/Lyapunov_uav/proposed').is_dir():
    raise SystemExit('Pass the existing repository root containing research/Lyapunov_uav/proposed')

def target(relative):
    path = repo / relative
    if repo not in path.resolve().parents:
        raise SystemExit(f'Path leaves repository: {relative}')
    return path

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None

errors = []
# Validate every input before the first write or removal.
for item in manifest['files']:
    source = package / 'files' / item['path']
    if digest(source) != item['replacement_sha256']:
        errors.append(f'Package integrity failed: {item["path"]}')
    if digest(target(item['path'])) not in item['allowed_before_sha256']:
        errors.append(f'Local file has other edits: {item["path"]}')
for item in manifest['remove_files']:
    if digest(target(item['path'])) not in item['allowed_before_sha256']:
        errors.append(f'Obsolete file has other edits: {item["path"]}')
if errors:
    raise SystemExit('\n'.join(errors) + '\nNo files were changed. Merge these differences first.')
if a.check:
    print(f"PASS: {len(manifest['files'])} source files and {len(manifest['remove_files'])} known old paths")
    raise SystemExit(0)
for item in manifest['files']:
    path = target(item['path'])
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(package / 'files' / item['path'], path)
removed = 0
for item in manifest['remove_files']:
    path = target(item['path'])
    if path.exists():
        path.unlink()
        removed += 1
print(f"Applied {len(manifest['files'])} source files; removed {removed} known obsolete files")
