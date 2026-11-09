#!/usr/bin/env bash
# Package a Linux release bundle from an existing build tree.
# usage: scripts/package-release.sh [RELEASE_TAG] [BUILD_ROOT] [PLATFORM_LABEL]
# The bundle links the system FFmpeg/poppler-glib of the build distribution (see README.md);
# NCCL (tensor parallelism) is bundled in lib/ and found through an $ORIGIN rpath.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
release_tag="${1:-v$(tr -d '[:space:]' < "$repo_root/VERSION")}"
build_root="${2:-$repo_root/build-linux}"
platform="${3:-linux-x64}"
dist_root="$repo_root/dist"
product_name="ninfer-rtx4090-$platform-$release_tag"
product_root="$dist_root/$product_name"
archive_name="$product_name.tar.gz"
archive_path="$dist_root/$archive_name"
checksum_path="$dist_root/SHA256SUMS-$platform-$release_tag.txt"

mkdir -p -- "$dist_root"
case "$product_root" in
  "$dist_root/$product_name") ;;
  *) printf 'Refusing to package outside dist: %s\n' "$product_root" >&2; exit 1 ;;
esac
rm -rf -- "$product_root"
rm -f -- "$archive_path"
mkdir -p -- "$product_root/lib"

# Every executable embeds the full sm_89+sm_120 kernel library (~440 MB), so the release is split
# into the runtime bundle and a -tools bundle to stay under GitHub's 2 GiB per-asset limit.
products=(
  'apps/ninfer:ninfer'
  'apps/ninfer-serve:ninfer-serve'
  'apps/ninfer_parallel_topology_probe:ninfer_parallel_topology_probe'
)
tool_products=(
  'apps/ninfer-perplexity:ninfer-perplexity'
  'bench/ninfer_bench:ninfer_bench'
)
for product in "${products[@]}"; do
  source_path="$build_root/${product%%:*}"
  destination="$product_root/${product#*:}"
  if [[ ! -f "$source_path" ]]; then
    printf 'Missing release product: %s\n' "$source_path" >&2
    exit 1
  fi
  cp -- "$source_path" "$destination"
done

# Bundle NCCL when the build links it, and point every binary at the bundled copy.
nccl="$(ldd "$product_root/ninfer-serve" | awk '/libnccl\.so/ {print $3; exit}')"
if [[ -n "$nccl" && -f "$nccl" ]]; then
  cp -L -- "$nccl" "$product_root/lib/libnccl.so.2"
  if command -v patchelf > /dev/null; then
    for product in "${products[@]}"; do
      patchelf --set-rpath '$ORIGIN/lib' "$product_root/${product#*:}"
    done
  else
    printf 'patchelf not found: run with LD_LIBRARY_PATH=<bundle>/lib\n' >&2
  fi
fi
rmdir --ignore-fail-on-non-empty -- "$product_root/lib"

cp -- "$repo_root/VERSION" "$repo_root/LICENSE" "$product_root/"
cp -- "$repo_root/docs/rtx-3090-linux.md" "$product_root/README.md"
cp -- \
  "$repo_root/scripts/download-qwen38.sh" \
  "$repo_root/scripts/run-qwen38-c1.sh" \
  "$repo_root/scripts/run-qwen38-c8.sh" \
  "$repo_root/scripts/run-qwen38-vision.sh" \
  "$repo_root/scripts/download-qwen36-35b-vision.sh" \
  "$repo_root/scripts/run-qwen36-35b-vision.sh" \
  "$product_root/"

(
  cd -- "$product_root"
  mapfile -d '' files < <(find . -type f ! -name SHA256SUMS.txt -print0 | LC_ALL=C sort -z)
  sha256sum -- "${files[@]}" > SHA256SUMS.txt
)
tar -C "$dist_root" -czf "$archive_path" "$product_name"

# Tools bundle: benchmarks and perplexity, sharing the runtime bundle's lib/ layout.
# The archive's top directory has the runtime bundle's name, so both extract into one folder.
tools_name="$product_name-tools"
tools_stage="$dist_root/.tools-stage"
tools_root="$tools_stage/$product_name"
tools_archive="$tools_name.tar.gz"
rm -rf -- "$tools_stage"
rm -f -- "$dist_root/$tools_archive"
mkdir -p -- "$tools_root"
for product in "${tool_products[@]}"; do
  source_path="$build_root/${product%%:*}"
  if [[ ! -f "$source_path" ]]; then
    printf 'Missing release product: %s\n' "$source_path" >&2
    exit 1
  fi
  cp -- "$source_path" "$tools_root/${product#*:}"
  if [[ -d "$product_root/lib" ]] && command -v patchelf > /dev/null; then
    patchelf --set-rpath '$ORIGIN/lib' "$tools_root/${product#*:}"
  fi
done
(
  cd -- "$tools_root"
  mapfile -d '' files < <(find . -type f -print0 | LC_ALL=C sort -z)
  sha256sum -- "${files[@]}" > SHA256SUMS-tools.txt
)
tar -C "$tools_stage" -czf "$dist_root/$tools_archive" "$product_name"
rm -rf -- "$tools_stage"
(
  cd -- "$dist_root"
  sha256sum -- "$archive_name" "$tools_archive" > "$(basename -- "$checksum_path")"
)
du -h -- "$archive_path" "$dist_root/$tools_archive" "$checksum_path"
