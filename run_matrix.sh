#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

languages=(python mojo cpp)
reasoning_efforts=(low mid high)
models=(gpt-oss-20b gpt-oss-120b)

# Check every prerequisite before starting an expensive generation run.
for language in "${languages[@]}"; do
  tasks_path="runs/${language}_tasks.jsonl"
  if [[ ! -f "${tasks_path}" ]]; then
    echo "missing prepared tasks: ${tasks_path}" >&2
    exit 2
  fi

  case "${language}" in
    python)
      image_path="containers/python-3.12-slim.sif"
      ;;
    mojo)
      image_path="containers/lc-eval-mojo.sif"
      ;;
    cpp)
      image_path="containers/lc-eval-mainstream.sif"
      ;;
  esac
  if [[ ! -f "${image_path}" ]]; then
    echo "missing judge image: ${image_path}" >&2
    exit 2
  fi
done

for model_name in "${models[@]}"; do
  for language in "${languages[@]}"; do
    for effort_label in "${reasoning_efforts[@]}"; do
      reasoning_effort="${effort_label}"
      if [[ "${reasoning_effort}" == "mid" ]]; then
        reasoning_effort="medium"
      fi

      printf '\n[%s | %s | %s] generate\n' \
        "${model_name}" "${language}" "${reasoning_effort}"
      LANGUAGE="${language}" \
      MODEL_NAME="${model_name}" \
      REASONING_EFFORT="${reasoning_effort}" \
      bash generate.sh

      printf '[%s | %s | %s] postprocess\n' \
        "${model_name}" "${language}" "${reasoning_effort}"
      LANGUAGE="${language}" \
      MODEL_NAME="${model_name}" \
      REASONING_EFFORT="${reasoning_effort}" \
      bash postprocess.sh

      printf '[%s | %s | %s] judge\n' \
        "${model_name}" "${language}" "${reasoning_effort}"
      LANGUAGE="${language}" \
      MODEL_NAME="${model_name}" \
      REASONING_EFFORT="${reasoning_effort}" \
      bash judge.sh

      printf '[%s | %s | %s] score\n' \
        "${model_name}" "${language}" "${reasoning_effort}"
      LANGUAGE="${language}" \
      MODEL_NAME="${model_name}" \
      REASONING_EFFORT="${reasoning_effort}" \
      bash score.sh
    done
  done
done
