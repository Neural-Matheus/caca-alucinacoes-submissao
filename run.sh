#!/usr/bin/env bash
# Ponto de entrada único da solução.
#   bash run.sh <caminho_db> <pasta_txt> <arquivo_saida>
# Gera <arquivo_saida> no formato de submissão (documento_id,citacoes).
set -euo pipefail

if [ $# -ne 3 ]; then
    echo "uso: bash run.sh <caminho_db> <pasta_txt> <arquivo_saida>" >&2
    exit 2
fi

# caminhos do usuário viram absolutos antes de entrar na pasta do repositório
DB=$(realpath "$1")
TXT=$(realpath "$2")
SAIDA=$(realpath -m "$3")
RAIZ=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODELO="$RAIZ/modelos/gliner25_ft5"

[ -f "$DB" ] || { echo "base não encontrada: $DB" >&2; exit 1; }
[ -d "$TXT" ] || { echo "pasta de .txt não encontrada: $TXT" >&2; exit 1; }
[ -f "$MODELO/model.safetensors" ] || {
    echo "modelo ausente em $MODELO; rode antes: python baixar_modelo.py" >&2; exit 1; }

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

cd "$RAIZ"
python run_baseline.py "$TXT" "$TMP" --gliner "$MODELO" --db "$DB"
mkdir -p "$(dirname "$SAIDA")"
cp "$TMP/submission.csv" "$SAIDA"
echo "saída: $SAIDA"
