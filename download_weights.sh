#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

WEIGHTS=(
  "policies/amo/model.pt"
  "6d867ed2dd2261d0f02a5e81d2b7f92802be30f3d36570fb7c4b18707649ef3f"
  "https://raw.githubusercontent.com/OpenTeleVision/AMO/34caaf943660e6f9420e35f64e86dd56fb51dd0e/amo_jit.pt"

  "policies/amo/model_adapter.pt"
  "159c5f691e55f68c68e2d98e287f4f87baa54e79759c04e06af3bdbc8f8edc98"
  "https://raw.githubusercontent.com/OpenTeleVision/AMO/34caaf943660e6f9420e35f64e86dd56fb51dd0e/adapter_jit.pt"

  "policies/homie/model.onnx"
  "afe6363c83c8b9ab8b1a2fed598fce7de35ff045eee1a95b336fd6720e3d3d49"
  "https://raw.githubusercontent.com/InternRobotics/OpenHomie/cefcd85fcf81f529e8be065795fb2a7273e69435/HomieDeploy/deploy.onnx"

  "policies/rl_gym/model.pt"
  "cf668f75b90d1abf73d2b87612a6e76bccc61ff7e083b63582d3f6aaa3c1759d"
  "https://raw.githubusercontent.com/unitreerobotics/unitree_rl_gym/276801e46c5d433564f24658bac64f254b7d2d4b/deploy/pre_train/g1/motion.pt"

  "policies/robomimic/model.pt"
  "d1d91b0201beeb649a4624ba40052d10fe4aebe98bf6f4847decf75dd1fee2da"
  "https://raw.githubusercontent.com/ccrpRepo/RoboMimic_Deploy/3a72ec1a55dcf155be88c9dcc4a32ecb5c11e313/policy/loco_mode/model/policy_29dof.pt"

  "policies/run_residual/model_cmg.onnx"
  "e7d1ee2e67e8d2c4e0fb179ddc328e96bb10bfbd8d636b4bdd635182b34fb97e"
  "https://raw.githubusercontent.com/PMY9527/RUN_DEPLOY/d6dba9c560c1a201dde244d1e2e600a399d8a607/robots/g1_29dof/config/policy/velocity/cmg/exported/cmg_exported_new.onnx"

  "policies/run_residual/model_residual.onnx"
  "5124422b4deb64cbf20cd9b7fb9f3bde9f1f0269fff9d4d4a93512cdcfdf9089"
  "https://raw.githubusercontent.com/PMY9527/RUN_DEPLOY/d6dba9c560c1a201dde244d1e2e600a399d8a607/robots/g1_29dof/config/policy/velocity/residual/exported/point25.onnx"

  "policies/bfm_zero/model.onnx"
  "209097902c45621eebab2edb81070c31895fcdd558f0cf6f7f5a360fd747ab74"
  "https://huggingface.co/LeCAR-Lab/BFM-Zero/resolve/62b4206d68e026de5e5dc7efb1529bccfb95164c/model/exported/FBcprAuxModel.onnx"

  "policies/dm_agile/model.onnx"
  "94a95b934e34b4c2682127cd66eacbadaa634db8a27d67b47ad15ff3a814dad9"
  "https://huggingface.co/datamentorshf/dm-g1-agile-locomotion-rl/resolve/4a32a3d8d1979880e2fa6a37760e6cf0017f400f/exported/policy.onnx"

  "policies/dm_march/model.onnx"
  "c82f23413644ed317b11ce92bf1390e8f35370b8a9c56d6fa6255ef3086567bf"
  "https://huggingface.co/datamentorshf/dm-g1-military-march-rl/resolve/86f48548ec595aa2b277390432619fd7c2c2c841/exported/policy.onnx"

  "policies/handoff/model.onnx"
  "0c0f7975a50234a2d20935bfec6fbc7aa88b6401d8654bb4feb3aff7d2329333"
  "https://raw.githubusercontent.com/lzyang2000/HANDOFF/6454ae8811f31ed722e561cb0ca7c1e432ac7ca8/deploy/ckpt/policy.onnx"

  "policies/legged_rl_lab/model.onnx"
  "26679e18977a3c5d26b4e653aaa46ef4f30fe862e7df719a2d5b77f1ebfab746"
  "https://raw.githubusercontent.com/zihanwang0422/legged_rl_lab/386ee24bc62c8033641d65b940dc2cf1dec56545/deploy/g1_deploy/exported_policy/g1_flat_1.onnx"

  "policies/sonic/model_decoder.onnx"
  "c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed"
  "https://huggingface.co/BlackCatRoboticsAI/g1-sonic-base/resolve/5ff852359854df09caab7c3733440322234a2def/model_decoder.onnx"

  "policies/sonic/model_encoder.onnx"
  "013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3"
  "https://huggingface.co/BlackCatRoboticsAI/g1-sonic-base/resolve/5ff852359854df09caab7c3733440322234a2def/model_encoder.onnx"
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
