# -*- coding: utf-8 -*-
"""Roda o pipeline sobre uma pasta de .txt e escreve os JSONs + submission.csv.

Uso: python run_baseline.py [pasta_txt] [pasta_saida] [--gliner modelos/gliner25_ft5] [--db base.db]

Sem --gliner roda só as regras (Python puro + pandas). Com --gliner acrescenta os
spans do GLiNER fine-tuned onde as regras não acharam nada (cacador/hibrido.py);
precisa de torch + gliner2 (ver Dockerfile).
"""
import argparse
import csv
import json
from pathlib import Path

from cacador.pipeline import Cacador


def celula(doc: dict) -> str:
    partes = []
    for c in doc["citacoes"]:
        idc = (c.get("resolucao") or {}).get("id_canonico") or "-"
        conf = c.get("confianca")
        partes.append(f"{c['inicio']},{c['fim']},{c['classificacao']},{idc},"
                      f"{'-' if conf is None else f'{conf:.4f}'}")
    return "|".join(partes) if partes else "-"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pasta", nargs="?", type=Path, default=Path("txt"))
    ap.add_argument("saida", nargs="?", type=Path, default=Path("out"))
    ap.add_argument("--gliner", type=Path, help="checkpoint do GLiNER fine-tuned (modo híbrido)")
    ap.add_argument("--db", type=Path, default=Path("desafio1_bracis.db"), help="base de referência")
    args = ap.parse_args()
    pasta, saida = args.pasta, args.saida
    (saida / "json").mkdir(parents=True, exist_ok=True)

    if args.gliner:
        from cacador.hibrido import Hibrido     # import tardio: torch só no modo híbrido
        cacador = Hibrido(args.gliner, db=str(args.db))
    else:
        cacador = Cacador(db=str(args.db))
    linhas, total, diagnostico = [], 0, []
    for arq in sorted(pasta.glob("*.txt")):
        doc = cacador.processar(arq.read_text(encoding="utf-8"), arq.stem)
        total += len(doc["citacoes"])
        linhas.append((doc["documento_id"], celula(doc)))
        for c in doc["citacoes"]:            # `_caminho` é interno, fora do contrato
            diagnostico.append((doc["documento_id"], c["inicio"], c["fim"],
                                c["classificacao"], c.pop("_caminho"), c["trecho"]))
        (saida / "json" / f"{arq.stem}.json").write_text(
            json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    destino = saida / "submission.csv"
    with destino.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["documento_id", "citacoes"])
        w.writerows(linhas)
    with (saida / "diagnostico.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["documento_id", "inicio", "fim", "classificacao", "caminho", "trecho"])
        w.writerows(diagnostico)
    print(f"{destino}: {len(linhas)} documentos, {total} citações.")


if __name__ == "__main__":
    main()
