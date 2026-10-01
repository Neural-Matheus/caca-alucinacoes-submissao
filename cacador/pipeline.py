# -*- coding: utf-8 -*-
"""Orquestração: texto -> citações classificadas, verificadas e calibradas."""
import json
from pathlib import Path

from .extract import extrair
from .resolve import Base, forma_curta_cnj, tribunal_da_citacao

# Confiança por caminho de decisão. A métrica dá até +10% multiplicativo pelo
# Brier sobre os pares casados, então a confiança precisa ser a probabilidade de
# a predição estar *inteiramente* correta (classe E link). Valores medidos pelo
# calibrar.py sobre o goldenset; CONFIANCA.json sobrescreve quando existe.
CONFIANCA_PADRAO = {
    "juris/unico": 0.95,
    "juris/ambiguo": 0.75,
    "juris/ausente": 0.90,
    "sumula/unico": 0.95,
    "sumula/inferido": 0.85,
    "sumula/ausente": 0.90,
    "lei/unico": 0.95,
    "lei/ausente": 0.90,
    "lei/diploma_desconhecido": 0.60,
    "tema/ausente": 0.85,
    "incompleta/-": 0.95,
}


def _carregar_confianca(caminho="CONFIANCA.json"):
    if caminho is None:                      # calibração: parte da tabela padrão
        return dict(CONFIANCA_PADRAO)
    p = Path(caminho)
    if p.exists():
        return {**CONFIANCA_PADRAO, **json.loads(p.read_text(encoding="utf-8"))}
    return dict(CONFIANCA_PADRAO)


class Cacador:
    def __init__(self, db="desafio1_bracis.db", confianca="CONFIANCA.json"):
        self.base = Base(db)
        self.confianca = _carregar_confianca(confianca)

    def _resolver(self, c):
        """Candidato -> (classificacao, id_canonico, caminho)."""
        if c.familia == "incompleta":
            return "incompleta", None, "incompleta/-"

        if c.familia == "juris":
            did, motivo = self.base.acordao(c.dados["numero"],
                                            curto=forma_curta_cnj(c.dados["bruto"]),
                                            tribunal=tribunal_da_citacao(c.texto))
        elif c.familia == "sumula":
            did, motivo = self.base.sumula(c.dados["numero"], c.dados["tribunal"],
                                           c.dados["vinculante"])
        elif c.familia == "lei":
            did, motivo = self.base.dispositivo(c.dados["artigo"], c.dados["diploma"])
        else:                                   # tema de repercussão geral
            did, motivo = None, "ausente"

        caminho = f"{c.familia}/{motivo}"
        if did is None:
            # tem identificador e não existe na base -> alucinação
            return "inventada", None, caminho
        return "real", str(did), caminho

    def processar(self, texto: str, documento_id: str) -> dict:
        citacoes = []
        for c in extrair(texto):
            classe, id_canonico, caminho = self._resolver(c)
            cit = {
                "inicio": c.inicio,
                "fim": c.fim,
                "trecho": c.texto,
                "tipo": "lei" if c.familia == "lei" else "jurisprudencia",
                "classificacao": classe,
                "confianca": round(self.confianca.get(caminho, 0.7), 4),
                "_caminho": caminho,
            }
            if classe == "real":
                cit["resolucao"] = {"id_canonico": id_canonico}
            citacoes.append(cit)
        return {"documento_id": documento_id, "citacoes": citacoes}
