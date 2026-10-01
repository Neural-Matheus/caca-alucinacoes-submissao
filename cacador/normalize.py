# -*- coding: utf-8 -*-
"""Normalização tolerante a OCR.

O texto original nunca é reescrito: a métrica pontua offsets em codepoints do
.txt como ele veio, então qualquer reescrita exigiria carregar um mapa de
offsets. Em vez disso, normalizamos apenas o *token* já extraído, na hora de
montar a chave de busca na base.
"""
import re
import unicodedata

# homóglifos que o OCR troca por dígitos (e vice-versa) dentro de um número
HOMOGLIFOS = {
    "O": "0", "o": "0", "D": "0", "Q": "0",
    "I": "1", "l": "1", "i": "1", "|": "1",
    "Z": "2", "z": "2",
    "A": "4",
    "S": "5", "s": "5",
    "G": "6",
    "T": "7",
    "B": "8",
    "g": "9", "q": "9",
}

# separadores que podem aparecer dentro de um número (inclui NBSP e quebra de linha:
# o gerador quebra números no meio, ex. "33.-\n474")
SEPARADORES = "   \t\n.-–—/,"

DIGITO = "0-9" + "".join(HOMOGLIFOS)


def chave(token: str) -> str:
    """Token numérico bruto -> sequência de dígitos canônica."""
    corrigido = "".join(HOMOGLIFOS.get(ch, ch) for ch in token)
    return re.sub(r"\D", "", corrigido)


def texto_plano(s: str, manter: str = "") -> str:
    """Minúsculas, sem acentos e com espaços colapsados — para comparar nomes de lei.

    `manter` preserva pontuação significativa: o ponto de milhar e a barra do ano
    distinguem "Lei nº 13.105/2015" (lei 13105) de "lei 131052015".
    """
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9%s]+" % re.escape(manter), " ", s.lower()).strip()
