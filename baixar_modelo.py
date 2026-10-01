# -*- coding: utf-8 -*-
"""Baixa o GLiNER2.5 fine-tuned (ft5) do Hugging Face em revisão fixa e confere o SHA-256.

Uso: python baixar_modelo.py [destino]      (padrão: modelos/gliner25_ft5)

Roda no `docker build` (com internet); a execução da solução fica offline.
"""
import hashlib
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "MatheusOliveiraa/cacador-gliner25-ft5"
REVISAO = "3e33557ac4cbdfac0262e9d502bd4bd9e4e1dfed"   # commit do repositório no HF, nunca "main"

SHA256 = {
    "config.json": "de7998f560cb58866fface88d164cfff82cd783b638eb058892d72c7090472ad",
    "model.safetensors": "9f0f0cb11e4348aeb91bfb276c04ee29dea3e7ca174d9362941cd83c3260afb7",
    "special_tokens_map.json": "df495ab27c8851887f6c3e19540accf0e380edcf907dde0391daa3d9566e1370",
    "tokenizer_config.json": "ffc6c2c91858d4899414580562afdf3d4f891ccedab5ec7b7f5b193d94c0e468",
    "tokenizer.json": "b107e2e5998e7429491aaae7807d8b0815ff3e06fcb5ba95bc6a35c9b93185c5",
    "training_config.json": "5fa1fa5e1f8c15782cb0660a9ff1e34502ce85745e4bd67ea9cd68c252051855",
    "encoder_config/config.json": "57d49f3c6b75688db54644eb3a0acddc4dd55efd01078a9eec50870203317613",
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def main() -> None:
    destino = Path(sys.argv[1] if len(sys.argv) > 1 else "modelos/gliner25_ft5")
    snapshot_download(REPO, revision=REVISAO, local_dir=destino,
                      allow_patterns=list(SHA256))
    errados = [n for n, h in SHA256.items() if sha256(destino / n) != h]
    if errados:
        sys.exit(f"SHA-256 não confere: {', '.join(errados)}")
    print(f"{destino}: {len(SHA256)} arquivos conferidos ({REPO}@{REVISAO[:10]})")


if __name__ == "__main__":
    main()
