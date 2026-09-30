#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
build_dir="${XGC2_MATH_QUALITY_BUILD_DIR:-${repo_root}/.ci/cpp-quality}"

cd "${repo_root}"

mapfile -t sources < <(find include test -path include/xgc2_math/third_party -prune -o -type f \( -name '*.hpp' -o -name '*.h' -o -name '*.cpp' \) -print | sort)
tidy_sources=(
  test/math_header_test.cpp
)

for tool in cmake ctest clang-format clang-tidy cppcheck; do
  if ! command -v "${tool}" >/dev/null 2>&1; then
    echo "Missing required C++ quality tool: ${tool}" >&2
    exit 1
  fi
done

clang-format --dry-run --Werror "${sources[@]}"

rm -rf "${build_dir}"
cmake -S "${repo_root}" -B "${build_dir}" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_CXX_FLAGS="-Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow -Wnon-virtual-dtor -Wold-style-cast -Woverloaded-virtual -Werror"
cmake --build "${build_dir}" -- -j"$(nproc)"
(cd "${build_dir}" && ctest --output-on-failure)

clang-tidy --quiet -p "${build_dir}" "${tidy_sources[@]}" 2>&1 | sed -E '/^[0-9]+ warnings generated\.$/d'

cppcheck \
  --enable=warning,performance,portability,style \
  --error-exitcode=1 \
  --std=c++17 \
  --inline-suppr \
  --suppressions-list=.xgc2/cppcheck_suppressions.txt \
  -i include/xgc2_math/third_party \
  -I include \
  include test

echo "C++ quality checks passed."
