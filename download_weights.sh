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

  "policies/huru/model.onnx"
  "9955ef11795b34e640518e6b81f427e7415a90069be39c8d27363433dd463173"
  "https://huggingface.co/fishy233/huru-models/resolve/bc4a0aeadd51a8c99e3aa701e6535004b67cdcd2/g1_walk_mjlab.onnx"

  "policies/josabb/model.onnx"
  "08106fd7515900f4822939acbece2fef3393608428c1aeb010c2e0b4ebc6b8ea"
  "https://huggingface.co/josabb/G1-humanoid-6dof-hands-locomotion-rl/resolve/83e06e1277b30f9a694678b711c8c753c03107a6/onnx/policy.onnx"

  "policies/mturan33/model.pt"
  "449728a9e5e48210012d74db2c6bd689b53e3c298646b7b2e3b1f2b27490b2c5"
  "https://huggingface.co/mturan33/g1-dual-critic-locomanip/resolve/07a034a67155652f2fd6d906bf95d19314a83716/unified_critic_s6u.pt"

  "policies/zealot/model_v26.onnx"
  "1e21412a09f3af7fa2dbdec58de4d4600e2679862a1b24c502c0a02916bd440f"
  "https://huggingface.co/haixuantao/zealot-g1-locomotion/resolve/774250dd362e3341d8fe3b69b44d2ff34b0d37ac/g1_v26_iter42290.onnx"

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

  "policies/bfm_zero/reward_locomotion.pkl"
  "1b974e7278ab6183d809d11fbd77cca817f3714a4808737cd69f96606ab77718"
  "https://huggingface.co/LeCAR-Lab/BFM-Zero/resolve/62b4206d68e026de5e5dc7efb1529bccfb95164c/model/reward_inference/reward_locomotion.pkl"

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
  "https://huggingface.co/nvidia/GEAR-SONIC/resolve/6733128a3d8a523b1418b06bca3cdf61c8b0987f/model_decoder.onnx"

  "policies/sonic/model_encoder.onnx"
  "013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3"
  "https://huggingface.co/nvidia/GEAR-SONIC/resolve/6733128a3d8a523b1418b06bca3cdf61c8b0987f/model_encoder.onnx"

  "policies/sonic/planner_sonic.onnx"
  "39b553e197f62f077975ba38512bc04781a3fc37c2af7c6756e04629f760edea"
  "https://huggingface.co/nvidia/GEAR-SONIC/resolve/6733128a3d8a523b1418b06bca3cdf61c8b0987f/planner_sonic.onnx"

  "policies/g1_gym/model.pt"
  "aaef00b10c001612638551670e3ecacbad11c3116c2cd2dfdee1b37add02e600"
  "https://raw.githubusercontent.com/IlikeSukiyaki/G1_GYM/1e5aa629187d7da31ce1f832071bd0a478b0f9aa/legged_gym/sim2mujoco/pre_train/g1_full/4_26_policy_lstm_16000.pt"

  "policies/mimic_lite/policy.yaml"
  "47be255e107d0447832b4819b555fe6c97208687b952f58b84338e6aeabf5069"
  "https://drive.usercontent.google.com/download?id=1z5qIRs78-k6syKcTEoiEvi-Jp3VBhaYy&export=download&confirm=t"

  "policies/mimic_lite/model_roa.onnx"
  "78aec8b2738f3940fb47a58021b9584faa3f69b53476551c52a2026909db33c1"
  "https://drive.usercontent.google.com/download?id=1xtHn7tu2s-A8RQ84AuT1EqiXQBYRe9pt&export=download&confirm=t"

  "policies/stepdown/model.pt"
  "3c3ddea79f3010640493bd65c39c7bdf8784131d923f1f1401c7957ba1adade6"
  "https://huggingface.co/arushbisht12/unitree-g1-stepdown-safety/resolve/fd97d9ac2eccda849e6f7440a4c400a6cd3697ae/safety_motion.pt"

  "policies/sunny/model_raw.onnx"
  "6cc6c71dc1d900bc240ade207cc7987b786d90503b9c899edc12ab44fcc56d62"
  "https://huggingface.co/jasonsfmeitian/g1-sunny-locomotion/resolve/3a8b206d42a9e3ce015513cb2fffe2498cb2e92c/baseline/g1_deploy_walk_policy.onnx"

  "policies/sunny/scan_mean.npy"
  "0ab35d51d848dea3c7d36557a611c85ed03e0ed87bedc9618560d68d4ccdd686"
  "https://huggingface.co/jasonsfmeitian/g1-sunny-locomotion/resolve/3a8b206d42a9e3ce015513cb2fffe2498cb2e92c/baseline/g1_deploy_scan_mean.npy"

  "policies/wcompton/model.onnx"
  "134ae9450c41e4b282766825ea0295bda2cd7eda83f59781053194b87bad15e1"
  "https://huggingface.co/wcompton/legged_locomotion_rl/resolve/844b125833faa820214546f16bc2e770db1471de/g1_blind_flat_end2end/2026-07-03_00-27-14_flat_loco/policy.onnx"
)

auth=()
if [ -n "${HF_TOKEN:-}" ]; then
  auth=(-H "Authorization: Bearer $HF_TOKEN")
fi

for ((i = 0; i < ${#WEIGHTS[@]}; i += 3)); do
  dest="${WEIGHTS[i]}"
  want="${WEIGHTS[i + 1]}"
  url="${WEIGHTS[i + 2]}"

  echo "fetch $dest"
  mkdir -p "$(dirname "$dest")"
  hdr=()
  case "$dest" in policies/mturan33/*) hdr=("${auth[@]+"${auth[@]}"}") ;; esac
  code=$(curl -sL -o "$dest" -w '%{http_code}' "${hdr[@]+"${hdr[@]}"}" "$url")
  if [ "$code" = "401" ] || [ "$code" = "403" ]; then
    echo "$dest: HTTP $code -- this repository is gated." >&2
    echo "  Set HF_TOKEN to a token whose account has accepted the gate:" >&2
    echo "  curl -X POST -H \"Authorization: Bearer \$HF_TOKEN\" \\" >&2
    echo "       -H 'Content-Type: application/json' -d '{}' \\" >&2
    echo "       ${url%/resolve/*}/ask-access" >&2
    exit 1
  fi
  if [ "$code" != "200" ]; then
    echo "$dest: HTTP $code" >&2
    exit 1
  fi

  got=$(sha256sum <"$dest" | cut -d' ' -f1)
  if [ "$got" != "$want" ]; then
    echo "sha256 mismatch for $dest" >&2
    echo "  expected $want" >&2
    echo "  got      $got" >&2
    exit 1
  fi
  echo "ok    $dest $got"
done
