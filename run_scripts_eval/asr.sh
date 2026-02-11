LANGS=("en" "de" "it" "es" "pt" "nl" ) # "ru" "cs" "hu" "sv" "fr"
MODELS=("qwen_omni" "phi_multimodal") 

TASK="ASR"
OUT_FOLDER="eval_outputs"
PREDICTION_FOLDER="outputs"

FAILED=0

for MODEL in "${MODELS[@]}"; do
  for LANG in "${LANGS[@]}"; do
    echo "▶️  Running evaluation | model=${MODEL} | lang=${LANG} | task=${TASK}"

    python eval_outputs.py \
      --lang "$LANG" \
      --model "$MODEL" \
      --task "$TASK" \
      --out_folder "$OUT_FOLDER" \
      --predictions_folder "$PREDICTION_FOLDER"

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
      echo "✅ Finished successfully | model=${MODEL} | lang=${LANG}"
    else
      echo "❌ FAILED | model=${MODEL} | lang=${LANG} (exit code ${EXIT_CODE})"
      FAILED=1
    fi

    echo "----------------------------------------"
  done
done

if [ $FAILED -eq 0 ]; then
  echo "🎉 All evaluations completed successfully!"
else
  echo "⚠️  Some evaluations failed. Check logs above."
fi
