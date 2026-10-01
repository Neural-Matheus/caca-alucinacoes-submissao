# Caça-Alucinações — BRACIS 2026 × Jusbrasil

Extração e verificação de citações jurídicas: cada citação do texto é
classificada como `real` (com o `id_canonico` na base), `inventada` ou
`incompleta`.

## Execução

Ponto de entrada único:

```bash
bash run.sh <caminho_db> <pasta_txt> <arquivo_saida>
```

Gera `<arquivo_saida>` no formato de submissão (`documento_id,citacoes`).

### Com Docker (recomendado)

```bash
docker build -t cacador .
docker run --rm --gpus all --network none -v /caminho/dados:/dados cacador \
    /dados/base.db /dados/txt /dados/submission.csv
```

O `docker build` é a única etapa que acessa a internet: ele baixa os pesos do
modelo (ver abaixo). A execução roda offline (`HF_HUB_OFFLINE=1`) e foi testada
com `--network none`.

### Sem Docker

```bash
pip install -r requirements.txt   # + torch 2.6 com CUDA
python baixar_modelo.py           # -> modelos/gliner25_ft5
bash run.sh base.db txt/ submission.csv
```

## Pesos do modelo

Os pesos (1,1 GB) não cabem no GitHub e ficam no Hugging Face, em repositório
público e **revisão fixa**:

- repositório: [`MatheusOliveiraa/cacador-gliner25-ft5`](https://huggingface.co/MatheusOliveiraa/cacador-gliner25-ft5)
- revisão: `3e33557ac4cbdfac0262e9d502bd4bd9e4e1dfed`

`baixar_modelo.py` baixa essa revisão (sem token) e confere o SHA-256 de cada
arquivo; se algum não bater, o build falha.

## Requisitos

- GPU: pico de ~2,1 GB de VRAM (o texto é processado em janelas de 200 tokens,
  então o pico não cresce com o tamanho do documento). Sem GPU roda na CPU,
  mais devagar.
- Determinismo: não há amostragem. A classe sai de busca exata na base e a
  confiança de uma tabela fixa (`CONFIANCA.json`).
- Base: não há pré-processamento persistido. Os índices são montados em memória
  a partir do `.db` no formato original (tabela `documentos`), então um `.db`
  novo no mesmo formato funciona direto.

## Abordagem

Híbrido: regras determinísticas + GLiNER2.5 fine-tuned. **O modelo só propõe
trechos; a classe sempre sai da base.**

```
cacador/normalize.py   homóglifos de OCR e chaves de busca
cacador/extract.py     regras: súmula, tema, lei, incompleta, jurisprudência
cacador/resolve.py     índices exatos sobre o .db
cacador/pipeline.py    classificação + confiança
cacador/hibrido.py     spans do GLiNER onde as regras não acharam nada
```

**A classe é consequência, não predição.** Nenhuma busca fuzzy:

```
sem identificador               -> incompleta
identificador + hit na base     -> real  + id_canonico
identificador + sem hit na base -> inventada
```

Como `real` só sai de um hit confirmado na base, o erro grave da métrica
(gabarito `inventada`, predição `real`) fica estruturalmente perto de zero.

**Texto original intocado.** A métrica pontua offsets do `.txt` como ele veio.
Os padrões toleram o ruído de OCR no lugar, e só o token extraído é normalizado
na hora de consultar a base.

**Índice de acórdãos.** Exato sobre os números dos primeiros 6 000 caracteres
de cada acórdão. Mais que isso passaria a indexar as citações que o acórdão faz
de outros processos. Um índice secundário cobre a forma curta com que o TSE cita
o CNJ (`RO nº 906-78` para `906-78.2010.6.02.0000`). Se a citação declara o
tribunal sem ambiguidade (sigla, segmento do CNJ e classe do recurso
concordando), só acórdãos desse tribunal são candidatos: assim um número citado
que não está na base não casa com outro acórdão que apenas o menciona.

**Modo híbrido.** As regras casam siglas e nomes de recurso exatamente, e o
ruído de OCR (`REsp` → `RE5p`) as derruba. Um GLiNER2.5
(`fastino/gliner2.5-multi-v1`) fine-tuned em dados sintéticos propõe spans
**onde as regras não acharam nada**. Cada span passa pela mesma resolução
contra a base; o rótulo do modelo só escolhe qual índice consultar.

## Resultados no desenvolvimento

Score oficial (`kaggle_metric`), τ = 0 em todas as linhas:

| conjunto | ruído de letras | só regras | híbrido |
|---|---|---|---|
| goldenset | 0% | 1,100 | **1,100** |
| goldenset | 5% | 0,781 | 1,025 |
| goldenset | 10% | 0,597 | 0,990 |
| 4 gêneros fora do treino | 0% | 0,939 | 1,053 |
| 4 gêneros fora do treino | 10% | 0,537 | 0,966 |

O goldenset foi o único dado rotulado da organização e as regras foram escritas
sobre ele; o número que mede generalização é o dos conjuntos com ruído e dos
gêneros fora do treino.
