#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
S='{BasedOnStyle: Google, ColumnLimit: 80, IndentWidth: 2, ReflowComments: false, SortIncludes: false, FixNamespaceComments: false, BinPackArguments: false, BinPackParameters: AlwaysOnePerLine, AlignAfterOpenBracket: BlockIndent, AllowAllParametersOfDeclarationOnNextLine: false, AllowAllArgumentsOnNextLine: false}'
mapfile -t F < <(git ls-files -z -- '*.cpp' | tr '\0' '\n')
if [ "${1:-}" = --check ]; then
  exec clang-format --style="$S" --dry-run --Werror "${F[@]}"
fi
clang-format --style="$S" -i "${F[@]}"
