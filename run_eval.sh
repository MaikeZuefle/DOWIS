LANG="de"
MODEL="qwen_omni"
TASK="S2ST"
OUT_FOLDER="eval_outputs"
PREDICTION_FOLDER="outputs"


python eval_outputs.py \
  --lang "$LANG" \
  --model "$MODEL" \
  --task "$TASK" \
  --out_folder "$OUT_FOLDER" \
  --predictions_folder "$PREDICTION_FOLDER"