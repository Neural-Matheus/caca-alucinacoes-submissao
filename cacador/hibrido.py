# -*- coding: utf-8 -*-
"""Regras + GLiNER2.5 fine-tuned: o modelo só propõe spans onde as regras não acharam nada.

As regras continuam donas do que acham; o modelo cobre o que o ruído de OCR
esconde delas (sigla corrompida deixa de qualificar o cabeçalho). A classe de
um span do modelo nunca vem do modelo: sai da mesma base fechada, e o rótulo
só escolhe qual índice consultar. Misturar os índices foi o que gerou τ > 0 no
primeiro híbrido (número de lei corrompida casando com cabeçalho de acórdão).

  artigo de lei             -> _LEI_RE + base.dispositivo, senão descarta
  citação de jurisprudência -> número de processo + base.acordao, senão descarta
  julgado sem número        -> incompleta, se houver ano e nenhum número de processo

Precisa de torch + gliner2 (imagem do Dockerfile); o import é tardio para
`cacador.pipeline` continuar funcionando sem eles.
"""
import re

from .extract import _LEI_INVERTIDA_RE, _LEI_RE, _NUM_RE, _SUMULA_RE, ANO, dados_sumula
from .normalize import chave
from .pipeline import Cacador
from .resolve import forma_curta_cnj, tribunal_da_citacao

# Descrições usadas no treino: na inferência têm de ser as mesmas.
ROTULOS = {
    "citação de jurisprudência": "Referência a um processo ou recurso judicial com sigla ou "
    "nome do recurso e número, ex.: REsp nº 1.234.567/SP, AgInt no AREsp 765.432/RJ, "
    "HC 123.456/SP, Reclamação 12.345/DF, RR-1000-10.2015.5.02.0001",
    "artigo de lei": "Dispositivo legal citado, ex.: art. 489, § 1º, do CPC; "
    "art. 37 da Constituição Federal; art. 2º da Lei 9.784/99",
    "julgado sem número": "Referência a um julgado sem número de processo, identificado só por "
    "tribunal, ano e relator, ex.: precedente do STJ de 2021, Rel. Min. Fulano de Tal",
}
LIMIAR = 0.5            # indiferente entre 0,3 e 0,9 na validação v3
CONFIANCA_MODELO = 0.7  # sem calibração própria ainda


# "OAB/BA 349745": o modelo às vezes marca a inscrição do advogado como citação.
# Tolera o ruído de letras do stress_test (O->ô, A->ã, B->h) e homóglifos numéricos.
_OAB_RE = re.compile(r"(?<![A-Za-zÀ-ú])[OÔ0][AÃ4][BH8](?![A-Za-zÀ-ú])", re.I)


# número de diploma: blocos de milhar e barra com ano de 4 dígitos no fim (homóglifos
# incluídos). Processo usa UF depois da barra ("/RS") ou o formato CNJ, com pontos.
_NUMERO_DE_LEI = re.compile(r"^[0-9OoIlSs]{1,3}(?:\.[0-9OoIlSs]{3})*\s*/\s*(?:19|20)[0-9OoIl]{2}$")


# Número feito só de homóglifos ("GG.SlG" = 66.516), que o _NUM_RE não aceita porque
# exige um dígito verdadeiro por bloco. Só vale aqui, dentro de um span que o modelo
# já marcou como citação: exige separador . ou - entre blocos e nenhuma letra colada
# (nas regras, palavras como "S.A." e "DAS" virariam números).
_HOM = "OoDQIli|ZzASsGTBgq"
_NUM_HOMOGLIFO = re.compile(rf"(?<![A-Za-zÀ-ú0-9{_HOM}])[0-9{_HOM}]+(?:[\.\-–][0-9{_HOM}]+)+"
                            rf"(?![A-Za-zÀ-ú0-9])")


_D = f"[0-9{_HOM}]"
_CNJ_COMPLETO = re.compile(rf"{_D}+[-–]{_D}{{2}}\.{_D}{{4}}\.{_D}\.{_D}{{2}}\.{_D}{{4}}$")


def numero_processo(trecho):
    """-> (chave, token bruto) do maior número que não seja um ano solto, ou None."""
    ks = [(chave(m.group()), m.group()) for m in _NUM_RE.finditer(trecho)]
    for m in _NUM_HOMOGLIFO.finditer(trecho):
        tok = m.group()
        # sigla corrompida colada por hífen não é número: "TT-3400-05.2011.5.21.0009"
        # (RR -> TT, e T é homóglifo de 7) viraria 773400... Só descarta o prefixo se o
        # resto for um CNJ completo; com ruído de dígitos "lOSG-47.ZOlO..." (1056-47...)
        # o bloco inicial é parte do número
        resto = re.sub(rf"^(?:[{_HOM}]+[-–])+", "", tok)
        if resto != tok and _CNJ_COMPLETO.match(resto):
            tok = resto
        if len(chave(tok)) >= 5:
            ks.append((chave(tok), tok))
    ks = [(k, b) for k, b in ks
          if len(k) >= 3 and not (len(k) == 4 and 1900 <= int(k) <= 2099)]
    return max(ks, key=lambda kb: len(kb[0])) if ks else None


def classificar(base, trecho, rotulo):
    """-> (classe, id_canonico, caminho) ou None para descartar o span."""
    if _OAB_RE.search(trecho):
        return None                     # inscrição de advogado, nunca número de processo
    if rotulo == "artigo de lei":
        m = _LEI_RE.search(trecho)
        if m:
            artigo, diploma = m.group(1), m.group(3)
        else:
            m = _LEI_INVERTIDA_RE.search(trecho)
            if not m:
                return None
            artigo, diploma = m.group(2), m.group(1)
        did, motivo = base.dispositivo(artigo, diploma)
        return ("real", str(did), f"modelo/lei/{motivo}") if did else \
            ("inventada", None, f"modelo/lei/{motivo}")
    num = numero_processo(trecho)
    if rotulo == "citação de jurisprudência":
        # súmula só consulta súmulas: "Súmüla nº 281 do STJ" (inventada) casava com um
        # acórdão que tem "281" no cabeçalho e virava real, o erro grave da métrica
        m = _SUMULA_RE.search(trecho)
        if m:
            did, motivo = base.sumula(*dados_sumula(m))
            return ("real", str(did), f"modelo/sumula/{motivo}") if did else \
                ("inventada", None, f"modelo/sumula/{motivo}")
        if not num:
            return None
        # "13.105/2015", "9.504/1997": número de lei com ano, não de processo. Com
        # "Lei" ilegível ("Icl rº 13.105/2015") o modelo marca jurisprudência, e o
        # número casava com acórdãos que citam o CPC na ementa (inventada -> real)
        if _NUMERO_DE_LEI.search(num[1]):
            return None
        curto = forma_curta_cnj(num[1])
        # chave curta aparece em muitos cabeçalhos e casa por coincidência; sem a
        # sigla que as regras exigem, só a forma curta do TSE (N-DD) é consultada
        if len(num[0]) < 5 and not curto:
            return None
        did, motivo = base.acordao(num[0], curto=curto, tribunal=tribunal_da_citacao(trecho))
        return ("real", str(did), f"modelo/juris/{motivo}") if did else \
            ("inventada", None, f"modelo/juris/{motivo}")
    if num is None and re.search(ANO, trecho):
        return ("incompleta", None, "modelo/incompleta/-")
    return None


def spans_modelo(model, texto, limiar=LIMIAR):
    """Spans do GLiNER sem sobreposição (fica o de maior confiança)."""
    res = model.extract_entities_long(texto, ROTULOS, threshold=limiar, chunk_size=200,
                                      chunk_overlap=50, include_confidence=True,
                                      include_spans=True)
    brutos = sorted(((e["start"], e["end"], float(e["confidence"]), rot)
                     for rot, ents in res["entities"].items() for e in ents),
                    key=lambda s: -s[2])
    aceitos = []
    for s in brutos:
        if all(min(s[1], x[1]) - max(s[0], x[0]) <= 0 for x in aceitos):
            aceitos.append(s)
    return aceitos


class Hibrido(Cacador):
    def __init__(self, modelo, db="desafio1_bracis.db", confianca="CONFIANCA.json",
                 limiar=LIMIAR, device=None):
        super().__init__(db=db, confianca=confianca)
        import torch
        from gliner2 import AutoExtractor

        self.limiar = limiar
        torch.manual_seed(0)                # inferência já é determinística; seed por garantia
        self.model = AutoExtractor.from_pretrained(str(modelo))
        self.model.eval()
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

    def processar(self, texto: str, documento_id: str) -> dict:
        doc = super().processar(texto, documento_id)
        cits = doc["citacoes"]
        extras = []
        for ini, fim, _, rotulo in spans_modelo(self.model, texto, self.limiar):
            if any(min(fim, c["fim"]) - max(ini, c["inicio"]) > 0 for c in cits):
                continue
            r = classificar(self.base, texto[ini:fim], rotulo)
            if r is None:
                continue
            classe, did, caminho = r
            cit = {
                "inicio": ini,
                "fim": fim,
                "trecho": texto[ini:fim],
                "tipo": "lei" if rotulo == "artigo de lei" else "jurisprudencia",
                "classificacao": classe,
                "confianca": round(self.confianca.get(caminho, CONFIANCA_MODELO), 4),
                "_caminho": caminho,
            }
            if classe == "real":
                cit["resolucao"] = {"id_canonico": did}
            extras.append(cit)
        doc["citacoes"] = sorted(cits + extras, key=lambda c: c["inicio"])
        return doc
