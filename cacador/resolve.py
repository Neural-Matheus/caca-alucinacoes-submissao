# -*- coding: utf-8 -*-
"""Resolução de citações contra desafio1_bracis.db.

Três índices, todos exatos — é o que garante que `inventada` seja uma conclusão
e não um chute. Nenhuma busca fuzzy: um número que não bate é ausente da base.

  acórdãos     chave de dígitos -> {doc_id}   (números do cabeçalho do acórdão)
  forma curta  sequencial+DV -> {doc_id}      (CNJ eleitoral citado como "RO nº 906-78")
  súmulas      (tribunal, nº, vinculante) -> doc_id
  dispositivos (diploma, artigo) -> doc_id
"""
import re
import sqlite3
from collections import defaultdict

from .normalize import chave, texto_plano

# quantos caracteres do início do acórdão contam como "identificação própria".
# Mais que isso passa a indexar as citações que o acórdão faz de outros processos,
# o que criaria links falsos (e, pior, transformaria `inventada` em `real`).
JANELA_CABECALHO = 6000
# até onde uma chave de menos de 5 dígitos ainda é o número do próprio acórdão
POSICAO_CHAVE_CURTA = 500

_NUM_BASE = re.compile(r"\d[\d\.\-/ ]{2,}\d")

# O TSE cita pelo começo do CNJ: "RO nº 906-78" para 906-78.2010.6.02.0000. Só o
# segmento 6 (Justiça Eleitoral) entra neste índice, e ele só é consultado quando o
# token citado tem exatamente a forma N-DD: é a cautela contra um número inventado
# curto casar por coincidência (inventada -> real é o erro grave da métrica).
_CNJ_BASE = re.compile(r"(\d{1,7})\s?-\s?(\d\s?\d)\s?\.\s?((?:19|20)\s?\d\s?\d)\s?\.\s?6\s?\.")
_FORMA_CURTA = re.compile(r"\s*([0-9OoDQIliZzASsGTBgq]{1,7})\s?[-–]\s?([0-9OoDQIliZzASsGTBgq]{2})\s*")


def forma_curta_cnj(bruto: str) -> bool:
    """O token numérico citado é só sequencial-DV ("906-78")?"""
    return bool(_FORMA_CURTA.fullmatch(bruto))


# Tribunal que a própria citação declara. Um número citado que não está na base
# pode aparecer no texto de outro acórdão ("nos autos do Recurso Especial
# 1.700.760/SP", dentro de uma reclamação do STF) e virar `real` com o link
# errado. Na base de desenvolvimento o acórdão certo sempre existe e ganha o
# desempate; numa base nova, não. Três sinais: sigla explícita, segmento do CNJ
# e classe do recurso. Só filtra quando todos os sinais presentes concordam: a
# sigla sozinha engana ("(TST - Recurso Especial Eleitoral: 4414420126260190"
# é do TSE, e o gabarito diz real). HC, MS, Rcl etc. não dão sinal de classe.
_SIGLA_TRIBUNAL = re.compile(r"(?<![A-Za-z])(STF|STJ|TST|TSE|STM)(?![A-Za-z])")
_SEGMENTO_CNJ = re.compile(r"\d[-–]\s?\d\s?\d\s?\.\s?(?:19|20)\d\d\s?\.\s?(\d)\s?\.")
# CNJ sem pontuação e sem os zeros à esquerda: ...AAAA J TR OOOO, J no 8º dígito do fim
_CNJ_CORRIDO = re.compile(r"(?<!\d)\d{6,9}((?:19|20)\d\d)(\d)\d{6}(?!\d)")
_TRIBUNAL_DO_SEGMENTO = {"1": "STF", "3": "STJ", "5": "TST", "6": "TSE", "7": "STM"}
_CLASSE_TRIBUNAL = [
    ("TSE", re.compile(r"REspe|Recurso\s+Especial\s+Eleitoral|Recurso\s+Ordin[áa]rio\s+Eleitoral", re.I)),
    ("STJ", re.compile(r"(?<![A-Za-z])[AE]?REsp(?![A-Za-z])|Recurso\s+Especial(?!\s+Eleitoral)", re.I)),
    ("STF", re.compile(r"(?<![A-Za-z])A?RE(?![A-Za-z])|Recurso\s+Extraordin[áa]rio"
                       r"|(?<![A-Za-z])(?:ADI|ADPF|ADC|ADO)(?![A-Za-z])")),
    ("TST", re.compile(r"(?<![A-Za-z])(?:A?IRR|ARR|RR)(?![A-Za-z])|Recurso\s+de\s+Revista", re.I)),
]


def tribunal_da_citacao(trecho: str):
    """-> sigla do tribunal que a citação declara, ou None se não for inequívoco."""
    sinais = set(_SIGLA_TRIBUNAL.findall(trecho))
    for m in _SEGMENTO_CNJ.finditer(trecho):
        sinais.add(_TRIBUNAL_DO_SEGMENTO.get(m.group(1), "?"))
    for m in _CNJ_CORRIDO.finditer(trecho):
        sinais.add(_TRIBUNAL_DO_SEGMENTO.get(m.group(2), "?"))
    classes = {t for t, rx in _CLASSE_TRIBUNAL if rx.search(trecho)}
    if "TSE" in classes:                    # "REspe" também casa o padrão do REsp
        classes.discard("STJ")
    sinais |= classes
    return sinais.pop() if len(sinais) == 1 and "?" not in sinais else None

# nome citado -> (tipo de diploma, número) tal como a base o registra
ALIAS_DIPLOMA = {
    "cpc": ("lei", "13105"),
    "codigo de processo civil": ("lei", "13105"),
    "cpp": ("decreto lei", "3689"),
    "codigo de processo penal": ("decreto lei", "3689"),
    "clt": ("decreto lei", "5452"),
    "consolidacao das leis do trabalho": ("decreto lei", "5452"),
    "cdc": ("lei", "8078"),
    "codigo de defesa do consumidor": ("lei", "8078"),
    "codigo civil": ("lei", "10406"),
    "codigo eleitoral": ("lei", "4737"),
    "cpm": ("decreto lei", "1001"),
    "codigo penal militar": ("decreto lei", "1001"),
    "codigo penal": ("decreto lei", "2848"),
    "cf": ("constituicao", ""),
    "carta magna": ("constituicao", ""),
    "diploma consumerista": ("lei", "8078"),
    "codigo consumerista": ("lei", "8078"),
    "lindb": ("decreto lei", "4657"),
    "adct": ("adct", ""),
    "cc": ("lei", "10406"),
    "constituicao": ("constituicao", ""),
    "constituicao federal": ("constituicao", ""),
    "constituicao fedcral": ("constituicao", ""),
    "constituicao da republica": ("constituicao", ""),
}


def diploma_citado(texto: str):
    """Trecho do diploma citado -> (tipo, número) canônico, ou None."""
    t = texto_plano(texto, manter="./")
    t = re.sub(r"^l[ec]i\b", "lei", t)          # "Lci": e->c do OCR
    # "nº" chega aqui como "no", "n.º" como "n.o" (texto_plano decompõe o º)
    for rotulo, chave_ in (("decreto lei", "decreto lei"), ("lei complementar", "lei complementar"),
                           ("lc", "lei complementar"), ("lei", "lei")):
        m = re.match(rf"{rotulo}\s*(?:n\s*\.?\s*o?\.?)?\s*([\d\.]+)", t)
        if m:
            return (chave_, m.group(1).replace(".", ""))
    t_simples = texto_plano(texto)
    # CPC/73 é outra lei (5.869/1973): sem isto "art. 373 do CPC/73" casava com o CPC/2015
    if re.match(r"(?:cpc\s*(?:19)?73|codigo de processo civil de 1973)\b", t_simples):
        return ("lei", "5869")
    for alias, canon in ALIAS_DIPLOMA.items():
        if t_simples.startswith(alias):
            return canon
    return None


def _norm_artigo(artigo: str) -> str:
    """'5º' / '1.134' -> '5' / '1134' (a base indexa o número puro)."""
    return artigo.replace(".", "").rstrip("º°o ").strip()


def _diploma_da_base(cabecalho: str):
    t = texto_plano(cabecalho, manter="./")
    m = re.search(r"d[ao]s? (lei complementar|decreto lei|lei)\s*n?o?\.?\s*([\d\.]+)", t)
    if m:
        return (m.group(1), m.group(2).replace(".", ""))
    if "constituicao" in t:
        return ("constituicao", "")
    return None


class Base:
    def __init__(self, caminho="desafio1_bracis.db"):
        con = sqlite3.connect(caminho)
        self.acordaos = defaultdict(set)
        self.curtos = defaultdict(set)
        self.posicao = {}
        self.sumulas = {}
        self.dispositivos = {}
        self.ids = set()
        self.tribunal = {}

        for did, natureza, texto, tribunal in con.execute(
                "select id, natureza, texto, tribunal from documentos"):
            self.ids.add(did)
            self.tribunal[did] = tribunal
            if natureza == "acordao":
                for m in _NUM_BASE.finditer(texto[:JANELA_CABECALHO]):
                    k = chave(m.group())
                    if len(k) < 3:
                        continue
                    self.acordaos[k].add(did)
                    anterior = self.posicao.get((k, did))
                    if anterior is None or m.start() < anterior:
                        self.posicao[(k, did)] = m.start()
                for m in _CNJ_BASE.finditer(texto[:JANELA_CABECALHO]):
                    k = m.group(1) + re.sub(r"\s", "", m.group(2))
                    self.curtos[k].add(did)
                    anterior = self.posicao.get(("curto", k, did))
                    if anterior is None or m.start() < anterior:
                        self.posicao[("curto", k, did)] = m.start()
            elif natureza == "sumula":
                m = re.match(r"S[úu]mula\s+(Vinculante\s+)?n\.?\s*(\d+)\s+d[oe]\s+([A-Z]{2,4})",
                             texto)
                if m:
                    self.sumulas[(m.group(3), m.group(2), bool(m.group(1)))] = did
            elif natureza == "dispositivo":
                cab = texto.split("\n", 1)[0]
                m = re.match(r"Artigos?\s+(\d+(?:\.\d+)*)[º°]?\s", cab)
                dip = _diploma_da_base(cab)
                if m and dip:
                    self.dispositivos[(dip, _norm_artigo(m.group(1)))] = did
        con.close()

    # ---------------------------------------------------------------- consultas

    def _do_tribunal(self, cands, tribunal):
        if not cands or tribunal is None:
            return cands
        return {d for d in cands if self.tribunal.get(d) == tribunal}

    def acordao(self, k: str, curto: bool = False, tribunal=None):
        """Chave de dígitos -> (doc_id, motivo) ou (None, motivo).

        `curto`: o token citado tem a forma sequencial-DV (ver forma_curta_cnj);
        só então, e se a chave exata falhar, consulta o índice da forma curta.
        `tribunal`: o que a citação declara (tribunal_da_citacao); acórdão de
        outro tribunal não é candidato.
        """
        for tentativa in (k, k.lstrip("0")):
            cands = self.acordaos.get(tentativa)
            if len(tentativa) >= 5:         # chave curta já tem a cautela própria abaixo
                cands = self._do_tribunal(cands, tribunal)
            if cands and len(tentativa) < 5:
                # chave curta ("331" de um número que o OCR cortou) aparece em muitos
                # cabeçalhos; só vale se for única e estiver onde fica o número do próprio
                # processo. Na dúvida, ausente: errar para "inventada" não é o erro grave
                d = next(iter(cands))
                if len(cands) > 1 or self.posicao[(tentativa, d)] >= POSICAO_CHAVE_CURTA:
                    return None, "curto_incerto"
            if cands:
                if len(cands) == 1:
                    return next(iter(cands)), "unico"
                # desempate: o acórdão em que o número aparece mais cedo é o
                # dono do número; nos demais ele é apenas uma citação interna
                melhor = min(cands, key=lambda d: self.posicao[(tentativa, d)])
                return melhor, "ambiguo"
        if curto:
            k = k.lstrip("0")
            cands = self._do_tribunal(self.curtos.get(k), tribunal)
            if cands:
                if len(cands) == 1:
                    return next(iter(cands)), "curto"
                # vários acórdãos do mesmo processo (agravo, embargos...): mesmo
                # desempate do índice exato
                melhor = min(cands, key=lambda d: self.posicao[("curto", k, d)])
                return melhor, "curto_ambiguo"
        return None, "ausente"

    def sumula(self, numero: str, tribunal, vinculante: bool):
        tribunais = [tribunal] if tribunal else (["STF"] if vinculante
                                                 else ["STF", "STJ", "TST", "TSE", "STM"])
        for t in tribunais:
            did = self.sumulas.get((t, numero, vinculante))
            if did:
                return did, "unico" if tribunal else "inferido"
        return None, "ausente"

    def dispositivo(self, artigo: str, trecho_diploma: str):
        dip = diploma_citado(trecho_diploma)
        if dip is None:
            return None, "diploma_desconhecido"
        did = self.dispositivos.get((dip, _norm_artigo(artigo)))
        return (did, "unico") if did else (None, "ausente")
