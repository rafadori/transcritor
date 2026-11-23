# 🎙️ Transcritor local

Converte áudios gravados em texto, rodando 100% offline na sua máquina com [faster-whisper](https://github.com/SYSTRAN/faster-whisper) e uma interface web simples em [Gradio](https://www.gradio.app/).

- Arraste vários áudios de uma vez (mp3, m4a, wav, ogg/opus do WhatsApp, vídeos...)
- Gera `.txt` (com ou sem marcação de tempo) e `.srt` (legenda)
- Usa a GPU NVIDIA quando disponível, senão cai para CPU

## Instalação

Requer Python 3.10+.

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Uso

Dê dois cliques em `Transcritor.bat` (ou rode `.venv\Scripts\python.exe transcritor.py`). A página abre no navegador.

Na primeira vez que cada modelo é usado, ele é baixado (~1,5 a 3 GB). Depois funciona offline.

As transcrições ficam salvas na pasta `transcricoes/`.

## Modelos

| Opção | Modelo | Observação |
|---|---|---|
| Rápido e bom | `large-v3-turbo` | Padrão, ótimo custo-benefício |
| Máxima qualidade | `large-v3` | Mais lento, melhor em áudio ruim |
| Leve | `medium` | Para máquinas sem GPU |
