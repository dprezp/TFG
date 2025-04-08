#!/bin/bash

echo "🚀 Ejecutando todos los métodos en serie..."

METHODS=("state_of_art" "first_prune" "best_prune" "first_relabel" "best_relabel")

for METHOD in "${METHODS[@]}"
do
  echo "▶️ Ejecutando método: $METHOD"
  python script.py --method $METHOD
  echo "✅ Finalizado: $METHOD"
done

echo "🎉 Todos los métodos han sido ejecutados."
