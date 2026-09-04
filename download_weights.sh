#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

WEIGHTS=(
  "policies/homie/model.onnx"
  "afe6363c83c8b9ab8b1a2fed598fce7de35ff045eee1a95b336fd6720e3d3d49"
  "https://raw.githubusercontent.com/InternRobotics/OpenHomie/cefcd85fcf81f529e8be065795fb2a7273e69435/HomieDeploy/deploy.onnx"

  "policies/robomimic/model.pt"
  "d1d91b0201beeb649a4624ba40052d10fe4aebe98bf6f4847decf75dd1fee2da"
  "https://raw.githubusercontent.com/ccrpRepo/RoboMimic_Deploy/3a72ec1a55dcf155be88c9dcc4a32ecb5c11e313/policy/loco_mode/model/policy_29dof.pt"

  "policies/run_residual/model_cmg.onnx"
  "e7d1ee2e67e8d2c4e0fb179ddc328e96bb10bfbd8d636b4bdd635182b34fb97e"
  "https://raw.githubusercontent.com/PMY9527/RUN_DEPLOY/d6dba9c560c1a201dde244d1e2e600a399d8a607/robots/g1_29dof/config/policy/velocity/cmg/exported/cmg_exported_new.onnx"

  "policies/run_residual/model_residual.onnx"
  "5124422b4deb64cbf20cd9b7fb9f3bde9f1f0269fff9d4d4a93512cdcfdf9089"
  "https://raw.githubusercontent.com/PMY9527/RUN_DEPLOY/d6dba9c560c1a201dde244d1e2e600a399d8a607/robots/g1_29dof/config/policy/velocity/residual/exported/point25.onnx"
)

for ((i = 0; i < ${#WEIGHTS[@]}; i += 3)); do
  dest="${WEIGHTS[i]}"
  want="${WEIGHTS[i + 1]}"
  url="${WEIGHTS[i + 2]}"

  echo "fetch $dest"
  curl -fL -o "$dest" "$url"

  got=$(sha256sum <"$dest" | cut -d' ' -f1)
  if [ "$got" != "$want" ]; then
    echo "sha256 mismatch for $dest" >&2
    echo "  expected $want" >&2
    echo "  got      $got" >&2
    exit 1
  fi
  echo "ok    $dest $got"
done
