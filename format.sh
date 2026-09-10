#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
S='{BasedOnStyle: Google, ColumnLimit: 80, IndentWidth: 2, ReflowComments: false, SortIncludes: false, FixNamespaceComments: false, BinPackArguments: false, BinPackParameters: AlwaysOnePerLine, AlignAfterOpenBracket: BlockIndent, AllowAllParametersOfDeclarationOnNextLine: false, AllowAllArgumentsOnNextLine: false}'
mapfile -t F < <(git ls-files -z -- '*.cpp' '*.h' | tr '\0' '\n')
mapfile -t P < <(git ls-files -z -- '*.py' | tr '\0' '\n')
if [ "${1:-}" = --check ]; then
  clang-format --style="$S" --dry-run --Werror "${F[@]}"
  exec ruff format --quiet --check "${P[@]}"
fi
clang-format --style="$S" -i "${F[@]}"
ruff format --quiet "${P[@]}"
