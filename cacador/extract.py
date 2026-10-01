# -*- coding: utf-8 -*-
"""Extração de spans candidatos.

Cinco famílias, todas casadas direto no texto original (offsets nativos):

  sumula      Súmula / Súmula Vinculante / Tema de repercussão geral
  lei         art. N de <diploma>
  incompleta  citação sem identificador (tribunal + ano + relator)
  juris       citação com identificador processual

A família `juris` não enumera cadeias recursais (AgInt no EDcl no AgRg no...):
ancora no número e expande para a esquerda token a token, o que cobre cadeias
arbitrárias. Um candidato só é aceito se o cabeçalho contiver ao menos um token
*qualificador* (sigla ou nome de recurso) — é isso que separa uma citação do
número do próprio processo no cabeçalho do documento.
"""
import re
from dataclasses import dataclass

from .normalize import chave

WS = r"[ \t  ]"

# --------------------------------------------------------------------------- vocabulário

SIGLAS = r"""(?:
      TST | AgInt | AGINT | Agint | AgRg | AGRG | AgARR | AgR | AGR
    | ADPF | ADI | ADC | ARE | MI | Pet | Inq | APn | AP | RRAg | ROT | IRR
    | CC | SL | AMS | PEDILEF | RO-El
    | EDcl | EDs | ED | AREspEl | AREsp | ARESP | AResp | AgREspe | AgREsp | AGRESP
    | REspe | RESPE | REspEl | REsp | RESP | RHC | RMS | RSE
    | RCL | Rcl | Recl | APL | AIRR | ARR | RR | AR | AI | MS | HC | RE | SLS
    | (?<![/\-\(])RO
    | Rec\.?\s*Esp\.? | R\.?\s*Esp\.? | A\.?\s*REsp | Ag\.?\s*Int\.?
    | H\.?\s*C\.? | AG\.?\s*REG | Ag\.?\s*Reg\.? | R\s*-\s*Rp
)"""

NOMES = r"""(?:
      Reclama[çc][õoã][eo]?s?
    | Recursos? | Especial | Eleitoral | Extraordin[áa]ri[oa] | Revista | Ordin[áa]ri[oa]
    | Habeas | Corpus | Mandado | Seguran[çc]a
    | Agravos? | Instrumento | Interno | Regimental
    | Embargos | Declara[çc][ãa]o | Suspens[ãa]o | Liminar | Senten[çc]a
    | Apela[çc][ãa]o | Peti[çc][ãa]o | Rescis[óo]ria | Criminal | C[íi]vel
    | Terceiro | Segundo | Primeiro
)"""

# extensões: alargam o span mas não qualificam sozinhas
EXTENSOES = r"""(?:
      [Pp]rocessos? | Autos
    | n\.[ºo°] | N\.[ºo°] | n[ºo°\.]? | N[ºo°\.]?
    | nos | nas | no | na | NOS | NAS | NO | NA | Nos | Nas | Na
    | dos | das | do | da | de | em | e
)"""

_SIGLA_RE = re.compile(SIGLAS + r"\Z", re.X)
_NOME_RE = re.compile(NOMES + r"\Z", re.X)

# um passo de expansão à esquerda: token + separadores, ancorado no fim. Token colado
# depois de "/" é UF ("OAB/MS 41183", "Campo Grande/MS"), não sigla de recurso.
_PASSO = re.compile(
    r"(?<![A-Za-zÀ-ú0-9/])(?:" + SIGLAS + r"|" + NOMES + r"|" + EXTENSOES + r")"
    r"[\s  :\-–—,\.\*_]{0,3}\Z", re.X)
# token com hífen invisível (soft hyphen do OCR: "R\xadEsp"); só qualifica se, sem o
# hífen, for sigla ou nome de recurso
_PASSO_SHY = re.compile(r"(?<![A-Za-zÀ-ú0-9\xad])[A-Za-zÀ-ú]+(?:\xad[A-Za-zÀ-ú]+)+"
                        r"[\s\-–—,\.]{0,3}\Z")
# segmento de sigla colado por hífen (TST-ED-E-ED-RR-41200-...)
_PASSO_COLADO = re.compile(r"(?<![A-Za-zÀ-ú])[A-Za-z]{1,6}[-–]\Z")

# --------------------------------------------------------------------------- número

HOMO = "OoDQIliZzASsGTBgq"
_RUN = rf"[0-9{HOMO}]*[0-9][0-9{HOMO}]*"           # bloco com ao menos um dígito real
# primeiro bloco só de homóglifos ("l.505.883" = 1.505.883): aceito se curto e seguido
# de ponto de milhar + bloco com dígito real, sem letra, dígito ou hífen antes. Sem
# isso o milhar inicial se perdia e a chave virava outra (505883), que a base não tem.
# O hífen fica de fora por causa das siglas coladas do TST: "ED-TT-160000" com RR
# corrompido para TT não pode virar o número 77160000.
_PREFIXO = rf"(?:(?<![A-Za-zÀ-ú0-9\-–])[{HOMO}]{{1,3}}\.(?={_RUN}))?"
NUM = rf"{_PREFIXO}{_RUN}(?:[ \t  \n\.\-–—/]{{1,4}}{_RUN}){{0,9}}"
_NUM_RE = re.compile(NUM)
_UF_RE = re.compile(rf"{WS}*(?:[/\-–—]{WS}*([A-Z]{{2}})\b|\({WS}*([A-Z]{{2}}){WS}*\))")

MAX_PASSOS = 12
MAX_CABECALHO = 130


@dataclass
class Candidato:
    inicio: int
    fim: int
    familia: str          # juris | lei | sumula | tema | incompleta
    texto: str
    dados: dict           # campos extraídos, específicos da família

    @property
    def largura(self) -> int:
        return self.fim - self.inicio


# --------------------------------------------------------------------------- famílias

EXTENSO_TRIBUNAL = {"supremo tribunal federal": "STF", "superior tribunal de justica": "STJ",
                    "tribunal superior do trabalho": "TST", "tribunal superior eleitoral": "TSE",
                    "superior tribunal militar": "STM"}
_TRIB_EXTENSO = (r"Supremo\s+Tribunal\s+Federal|Superior\s+Tribunal\s+de\s+Justi[çc]a"
                 r"|Tribunal\s+Superior\s+do\s+Trabalho|Tribunal\s+Superior\s+Eleitoral"
                 r"|Superior\s+Tribunal\s+Militar")
# marcador de súmula de OUTRO tribunal: sem ele o tribunal ausente é deduzido; com ele a
# busca não deduz ("Súmula 83 deste Tribunal de Justiça" não é a Súmula 83 do STJ)
_OUTRO_TRIBUNAL = (rf"dest[ae]{WS}*(?:Corte|Tribunal|Regional|Casa)(?:{WS}+de{WS}+Justi[çc]a)?"
                   rf"|do{WS}*extinto{WS}*TFR|d[oa]{WS}*(?:TJ|TRF|TRT|TRE)[A-Z0-9]*")
_SUMULA_RE = re.compile(
    rf"(?<![A-Za-zÀ-ú])(?:(?P<sv>SV)|(?:(?:[Vv]erbete|[Ee]nunciado|(?P<nao_sumula>OJ|Orienta[çc][ãa]o{WS}+Jurisprudencial"
    rf"|Precedente{WS}+Normativo))(?:{WS}+sumular)?{WS}*(?:[nN]\.?[º°o]?\.?)?"
    rf"(?={WS}*\d)|[Ss5][ÚÙÛÜúùûüUu][MmNn][A-Za-zÀ-ÿ\.]*)"             # Súmula / Súm. / 5úmula / Verbete
    rf"(?:{WS}*\n?{WS}*(?P<vinc>Vinculante))?)"
    rf"{WS}*\n?{WS}*(?:de{WS}*)?(?:[nN]\.?[º°o]?\.?)?{WS}*\n?{WS}*"
    rf"(?P<num>[0-9{HOMO}]*[0-9][0-9{HOMO}]*)(?![A-Za-zÀ-ú0-9])"
    rf"(?:{WS}*(?:/|[-–]){WS}*(?P<trib1>[A-Z]{{2,4}})\b(?:[-–/][A-Z0-9]+)*"   # 7/STJ, 331-TST, /SDI-I/TST
    rf"|(?:{WS}*,?{WS}*\n?{WS}*(?:da{WS}*(?:S[úu]mula(?:{WS}*(?P<vinc2>Vinculante))?|S?BDI[-–]?[I1]+){WS}*)"
    rf"(?:d[oe]{WS}*\n?{WS}*(?:c(?:ol)?\.{WS}*)?(?:(?P<trib3>[A-Z]{{2,4}})\b|(?P<ext2>{_TRIB_EXTENSO})))?"
    rf"|(?:{WS}*,?{WS}*\n?{WS}*d[oe]{WS}*\n?{WS}*(?:c(?:ol)?\.{WS}*)?"
    rf"(?:(?P<trib2>[A-Z]{{2,4}})\b|(?P<ext>{_TRIB_EXTENSO}))))"
    rf"|{WS}*,?{WS}*\n?{WS}*(?P<outro>{_OUTRO_TRIBUNAL}))?",
    re.X)


_SUMULA_ABREV_RE = re.compile(rf"(?<![A-Za-zÀ-ú])S\.{WS}*(?P<num>\d+){WS}*/{WS}*(?P<trib1>[A-Z]{{2,4}})\b")


def dados_sumula(m):
    """Match de _SUMULA_RE -> (número, tribunal ou None, vinculante)."""
    g = m.groupdict()
    # OJ e Precedente Normativo não são súmulas: "OJ 331 da SBDI-1 do TST" não é a Súmula 331
    if g.get("outro") or g.get("nao_sumula"):
        tribunal = "OUTRO"                          # nunca casa com a base: não deduz
    elif g.get("ext") or g.get("ext2"):
        tribunal = EXTENSO_TRIBUNAL[texto_sem_acento(g.get("ext") or g.get("ext2"))]
    else:
        tribunal = g.get("trib1") or g.get("trib2") or g.get("trib3")
    vinc = bool(g.get("sv") or g.get("vinc") or g.get("vinc2"))
    return chave(m.group("num")).lstrip("0") or "0", tribunal, vinc


def texto_sem_acento(s):
    import unicodedata
    s = unicodedata.normalize("NFKD", " ".join(s.split()).lower())
    return "".join(c for c in s if not unicodedata.combining(c))


_TEMA_RE = re.compile(
    rf"Tem[aãáà]{WS}*([\d\.]+){WS}*\n?{WS}*d[ae]{WS}*\n?{WS}*repercuss[ãa]o{WS}*\n?{WS}*geral")

_DIPLOMA = rf"""(?:
      L[ec]i{WS}*\n?{WS}*Complementar{WS}*\n?{WS}*(?:[nN]\.?[º°o]?\.?)?{WS}*\n?{WS}*[\d\.]+(?:{WS}*/{WS}*\d{{2,4}})?
    | L[ec]i{WS}*\n?{WS}*(?:[nN]\.?[º°o]?\.?)?{WS}*\n?{WS}*[\d\.]+(?:{WS}*/{WS}*\d{{2,4}})?
        (?:,?{WS}*\n?{WS}*de{WS}*\n?{WS}*(?:\d{{1,2}}[º°]?{WS}*de{WS}*[a-zç]+{WS}*de{WS}*)?\d{{4}})?
        (?:{WS}*\n?{WS}*\([^)\n]{{3,80}}\))?
    | LC{WS}*(?:[nN]\.?[º°o]?\.?)?{WS}*[\d\.]+(?:{WS}*/{WS}*\d{{2,4}})?
    | RI(?:STF|STJ|TST|TSE|STM)
    | (?:diploma|C[óo]digo){WS}*\n?{WS}*consumerista
    | C[óo0]digo{WS}*\n?{WS}*(?:de{WS}*\n?{WS}*)?
        (?:Processo{WS}*\n?{WS}*Civil|Processo{WS}*\n?{WS}*Penal|Defesa{WS}*\n?{WS}*do
         {WS}*\n?{WS}*Consumidor|Penal{WS}*\n?{WS}*Militar|Civil|Eleitoral|Penal)
    | Constitui[çc][ãa]o(?:{WS}*\n?{WS}*(?:Federal|Fedcral|da{WS}*\n?{WS}*Rep[úu]blica))?
    | Consolida[çc][ãa]o{WS}*\n?{WS}*das{WS}*\n?{WS}*Leis{WS}*\n?{WS}*do{WS}*\n?{WS}*Trabalho
    | Decreto{WS}*[-–]?{WS}*Lei{WS}*(?:[nN]\.?[º°o]?\.?)?{WS}*[\d\.]+(?:{WS}*/{WS}*\d{{2,4}})?
    | Carta{WS}*\n?{WS}*Magna | LINDB | ADCT
    | CF{WS}*/{WS}*(?:19)?88 | CPC{WS}*/{WS}*(?:20)?15 | CPC{WS}*/{WS}*(?:19)?73
    | CLT | CPC | CPP | CDC | CF | CPM | CC
)"""

_LEI_RE = re.compile(
    rf"\bart(?:igos?|s)?[º°]?\.?{WS}*\n?{WS}*(\d+(?:\.\d+)*[º°]?(?:-[A-Z])?)"      # art. 373, artº 7º, 1º-A
    rf"(?:{WS}*(?:,|e|a){WS}*\d+(?:\.\d+)*[º°]?(?![\d/]))*"                    # arts. 186 e 927
    rf"(?:{WS}*e{WS}*(?:ss|seguintes)\.?)?"                                   # art. 14 e ss.
    rf"((?:{WS}*(?:,|{WS}e){WS}*\n?{WS}*(?:[IVXLCDM]+|§{WS}*\n?{WS}*(?:\d+[º°]?|[úu]nico)(?:{WS}*-{WS}*[A-Z])?"
    rf"|caput|par[áa]grafo{WS}*[úu]nico|['\"][a-z]['\"]|[a-z]\)"
    rf"|inc(?:iso)?s?\.?{WS}*[IVXLCDM]+|(?:letra|al[íi]nea){WS}*\n?{WS}*['\"‘’]?[a-z]['\"‘’]?"
    rf"|e{WS}*\n?{WS}*\d+(?:\.\d+)*[º°]?(?![\d/])))*)"                          # , inc. IV, letra 'b', e 1.060
    rf"(?:{WS}*,?{WS}*c/c{WS}*(?:o{WS}*)?art\.?{WS}*\d+(?:\.\d+)*[º°]?(?:{WS}*,{WS}*[IVXLCDM]+)?)?"
    rf"{WS}*,?{WS}*(?:ambos{WS}*)?\n?{WS}*(?:d[aeo]s?{WS}*\n?{WS}*)?" + "(" + _DIPLOMA + ")",
    re.X | re.I)

_LEI_INVERTIDA_RE = re.compile(
    "(" + _DIPLOMA + ")"
    rf"{WS}*,{WS}*\n?{WS}*art(?:igo)?s?\.?{WS}*\n?{WS}*(\d+(?:\.\d+)*[º°]?(?:-[A-Z])?)"
    rf"((?:{WS}*(?:,|{WS}e){WS}*\n?{WS}*(?:[IVXLCDM]+|§{WS}*(?:\d+[º°]?|[úu]nico)|caput"
    rf"|par[áa]grafo{WS}*[úu]nico|['\"][a-z]['\"]|[a-z]\)))*)"
    rf"(?:{WS}*(?:e|a){WS}*\d+(?:\.\d+)*[º°]?(?![\d/]))?",
    re.X)

_INCOMPLETA_RE = re.compile(
    r"(?:julgad[oa]|precedente|ac[óo]rd[ãa]o|Reclama[çc][ãa]o|Rcl|APL|RHC|RSE|REsp|AR"
    rf"|Recurso{WS}*\n?{WS}*(?:em{WS}*\n?{WS}*)?[A-Za-zÀ-ú]+(?:{WS}*\n?{WS}*[A-Za-zÀ-ú]+){{0,3}}"
    rf"|Agravo{WS}*\n?{WS}*(?:em|no|interno)?{WS}*\n?{WS}*[A-Za-zÀ-ú]*"
    rf"(?:{WS}*\n?{WS}*[A-Za-zÀ-ú]+){{0,2}})"
    rf"{WS}*\n?{WS}*(?:d[oe]{WS}*\n?{WS}*([A-Z]{{2,4}}))?"               # do STF
    rf"{WS}*,?{WS}*\n?{WS}*(?:[a-zà-úç]{{5,12}}{WS}*\n?{WS}*)?"
    rf"{WS}*\n?{WS}*(?:em|de){WS}*\n?{WS}*((?:19|20)\d{{2}}),?"          # em 2024
    rf"{WS}*\n?{WS}*(?:pela|da|sob|de)?{WS}*\n?{WS}*"
    rf"(?:relatoria{WS}*\n?{WS}*d[eoc]|Rel\.{WS}*\n?{WS}*(?:Min\.)?)"    # Rel. Min. / relatoria de
    rf"{WS}*\n?{WS}*((?:[A-ZÀ-Ú][\wÀ-úçãõâêôáéíóú']*(?:{WS}*\n?{WS}*)?){{1,5}})")

# Incompleta pela estrutura, não pela redação: tipo de julgado, depois (sem nenhum
# dígito) tribunal e verbo, o ano, e o marcador de relatoria com o nome. É o que todas
# as incompletas do goldenset têm; sem ano ou sem relator não é incompleta (é menção
# genérica, como "a jurisprudência do STJ"). O nome atravessa no máximo uma quebra de
# linha, para não engolir a linha seguinte ("Admar Gonzaga\n\nAdvogado: ...").
_S = rf"(?:{WS}|\n)+"
_S0 = rf"(?:{WS}|\n)*"
_PALAVRA = r"[A-Za-zÀ-ú]+"
_CABECA = rf"""(?<![A-Za-zÀ-ú])(?i:
      julgad[oa]s? | precedentes? | ac[óo]rd[ãa]os? | entendimento | julgamento
    | decis[ãa]o(?:{_S}(?:colegiada|monocr[áa]tica|plen[áa]ria))?
    | Reclama[çc][ãa]o | Rcl | APL | RHC | RSE | REsp | AREsp | AgInt | AgRg | HC
    | Recurso(?:{_S}{_PALAVRA}){{0,3}} | Agravo(?:{_S}{_PALAVRA}){{0,3}}
    | Embargos(?:{_S}{_PALAVRA}){{0,2}} | Habeas{_S}Corpus
    | Mandado{_S}de{_S}Seguran[çc]a
)(?![A-Za-zÀ-ú])"""                     # palavra inteira: "Aplica-se" não é APL
_SEM_DIGITO = r"(?:[^\d\.;:!?]){0,70}?"
# ano com homóglifo do OCR ("2O16"); chave() normaliza quando preciso
# todos os homóglifos que chave() normaliza ("ZOZ4" = 2024, "lgg8" = 1998)
ANO = r"(?<![0-9A-Za-z])(?:[1lI|][9gq]|[2Zz][0OoDQ])[0-9OoDQIli|ZzASsGTBgq]{2}(?![0-9A-Za-z])"
_RELATORIA = rf"""(?:
      (?:(?:sob|na|da|pela|com){_S})?relatoria{_S}d[aoeçc]{_S0}
    | Rel\.{_S0}
    | relator[a]?{_S}(?:(?:o|a){_S})?
    | relatad[oa]{_S}pel[oa]{_S}
)(?:(?:Ministr[oa]|Min\.|Desembargador[a]?|Des\.){_S0})?"""
_NOME_CAP = r"[A-ZÀ-Ú][\wÀ-úçãõâêôáéíóú']*"
_QUEBRA_NOME = rf"(?:{WS}+\n?{WS}*|\n{WS}*)"
_NOME = rf"{_NOME_CAP}(?:{_QUEBRA_NOME}(?:(?:de|da|do|dos|das|e){_QUEBRA_NOME})?{_NOME_CAP}){{0,5}}"
_INCOMPLETA_ESTRUTURAL_RE = re.compile(
    rf"{_CABECA}{_SEM_DIGITO}({ANO}){_SEM_DIGITO}{_RELATORIA}({_NOME})", re.X)

_CONECTIVO_FINAL = re.compile(r"(?:\s|\b(?:D[EeAa]|d[ea])\b)+\Z")


def _expandir_esquerda(texto: str, inicio: int):
    """Absorve o cabeçalho da citação. Retorna (novo_inicio, qualifica)."""
    pos, qualifica = inicio, False
    for _ in range(MAX_PASSOS):
        if inicio - pos >= MAX_CABECALHO:
            break
        m = _PASSO.search(texto, 0, pos) or _PASSO_COLADO.search(texto, 0, pos)
        if not m or m.end() != pos:
            m = _PASSO_SHY.search(texto, 0, pos)
            if not m or m.end() != pos:
                break
            limpo = re.sub(r"[^A-Za-zÀ-ú]", "", m.group())
            if not (_SIGLA_RE.match(limpo) or _NOME_RE.match(limpo)):
                break
        tok = texto[m.start():m.end()].strip(" \t  \n-–—,.")
        tok = tok.replace("\xad", "")
        if _SIGLA_RE.match(tok) or _NOME_RE.match(tok):
            qualifica = True
        pos = m.start()
    return pos, qualifica


def _juris(texto: str):
    for m in _NUM_RE.finditer(texto):
        bruto = m.group()
        if len(re.findall(r"[0-9]", bruto)) < 2:
            continue
        k = chave(bruto)
        if len(k) < 3:
            continue
        ini, qualifica = _expandir_esquerda(texto, m.start())
        if not qualifica:
            continue
        # ano solto ("Rcl de 2021, Rel. Min. ...") não é identificador
        if len(k) == 4 and 1900 <= int(k) <= 2099 and \
                re.search(r"\b(?:de|em)[\s ]*\Z", texto[:m.start()]):
            continue
        fim = m.end()
        uf = None
        mu = _UF_RE.match(texto, fim)
        if mu:
            uf = mu.group(1) or mu.group(2)
            fim = mu.end()
        yield Candidato(ini, fim, "juris", texto[ini:fim],
                        dict(numero=k, uf=uf, bruto=bruto))


def _sumulas(texto: str):
    for m in _SUMULA_ABREV_RE.finditer(texto):
        numero, tribunal, vinc = dados_sumula(m)
        yield Candidato(m.start(), m.end(), "sumula", m.group(),
                        dict(numero=numero, vinculante=vinc, tribunal=tribunal))
    for m in _SUMULA_RE.finditer(texto):
        numero, tribunal, vinc = dados_sumula(m)
        # o marcador de outro tribunal entra no span, como o tribunal em "Súmula 331 do TST"
        yield Candidato(m.start(), m.end(), "sumula", m.group(),
                        dict(numero=numero, vinculante=vinc, tribunal=tribunal))
    for m in _TEMA_RE.finditer(texto):
        yield Candidato(m.start(), m.end(), "tema", m.group(),
                        dict(numero=m.group(1)))


def _leis(texto: str):
    for m in _LEI_RE.finditer(texto):
        yield Candidato(m.start(), m.end(), "lei", m.group(),
                        dict(artigo=m.group(1), incisos=m.group(2) or "",
                             diploma=m.group(3)))
    for m in _LEI_INVERTIDA_RE.finditer(texto):              # "CF, art. 37, caput"
        yield Candidato(m.start(), m.end(), "lei", m.group(),
                        dict(artigo=m.group(2), incisos=m.group(3) or "", diploma=m.group(1)))


def _incompletas(texto: str):
    for m in _INCOMPLETA_RE.finditer(texto):
        fim = m.end()
        corte = _CONECTIVO_FINAL.search(m.group(3) or "")
        if corte:
            fim -= len(m.group(3)) - corte.start()
        yield Candidato(m.start(), fim, "incompleta", texto[m.start():fim],
                        dict(tribunal=m.group(1), ano=m.group(2), relator=m.group(3)))
    for m in _INCOMPLETA_ESTRUTURAL_RE.finditer(texto):
        yield Candidato(m.start(), m.end(), "incompleta", m.group(),
                        dict(tribunal=None, ano=m.group(1), relator=m.group(2)))


PRIORIDADE = {"sumula": 0, "tema": 0, "lei": 1, "incompleta": 2, "juris": 3}


def extrair(texto: str):
    """Todos os candidatos, sem sobreposição (a métrica rejeita IoU >= 0,5 entre predições)."""
    cands = list(_sumulas(texto)) + list(_leis(texto)) + \
        list(_incompletas(texto)) + list(_juris(texto))
    # desempate: span mais longo vence; depois, menor prioridade numérica
    cands.sort(key=lambda c: (-c.largura, PRIORIDADE[c.familia], c.inicio))
    aceitos = []
    for c in cands:
        if any(min(c.fim, a.fim) - max(c.inicio, a.inicio) > 0 for a in aceitos):
            continue
        aceitos.append(c)
    # "REsp 1.234/SP, julgado em 2020, Rel. Min. X": ano e relator são da citação
    # numerada, não uma incompleta à parte
    numeradas = [a for a in aceitos if a.familia != "incompleta"]
    aceitos = [c for c in aceitos if c.familia != "incompleta" or not any(
        0 <= c.inicio - a.fim <= 4 and not texto[a.fim:c.inicio].strip(" \t\n,;-–—()")
        for a in numeradas)]
    aceitos.sort(key=lambda c: c.inicio)
    return aceitos
