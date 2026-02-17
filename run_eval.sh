LANG="de"
MODEL="qwen_omni"
TASK="TTS"
OUT_FOLDER="eval_outputs"
PREDICTION_FOLDER="outputs_debug"


python eval_outputs.py \
  --lang "$LANG" \
  --model "$MODEL" \
  --task "$TASK" \
  --out_folder "$OUT_FOLDER" \
  --predictions_folder "$PREDICTION_FOLDER"