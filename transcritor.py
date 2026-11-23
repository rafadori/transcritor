"""Transcritor de áudio local (faster-whisper + Gradio).

Abre uma página no navegador onde você arrasta áudios gravados e recebe a
transcrição em texto e legenda (.srt). Tudo roda offline na sua máquina.
"""

import os
import sys
import time
from pathlib import Path

# --- Torna as DLLs de CUDA instaladas via pip (nvidia-cublas / nvidia-cudnn) visíveis no Windows ---
if sys.platform == "win32":
    nvidia_root = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    for bin_dir in nvidia_root.glob("*/bin"):
        os.add_dll_directory(str(bin_dir))
        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ["PATH"]

import gradio as gr
from faster_whisper import WhisperModel

BASE_DIR = Path(__file__).parent
SAIDA_DIR = BASE_DIR / "transcricoes"
SAIDA_DIR.mkdir(exist_ok=True)

MODELOS = {
    "Rápido e bom (large-v3-turbo)": "large-v3-turbo",
    "Máxima qualidade (large-v3)": "large-v3",
    "Leve (medium)": "medium",
}
IDIOMAS = {"Português": "pt", "Detectar automaticamente": None, "Inglês": "en", "Espanhol": "es"}

_cache: dict[str, WhisperModel] = {}


def carregar_modelo(nome: str) -> WhisperModel:
    if nome not in _cache:
        _cache.clear()  # mantém só um modelo na VRAM
        try:
            # int8_float16 cabe folgado em 6 GB de VRAM, inclusive o large-v3
            _cache[nome] = WhisperModel(nome, device="cuda", compute_type="int8_float16")
        except Exception as e:  # sem GPU/CUDA: cai para CPU
            print(f"[aviso] GPU indisponível ({e}); usando CPU.")
            _cache[nome] = WhisperModel(nome, device="cpu", compute_type="int8")
    return _cache[nome]


def tempo_srt(seg: float) -> str:
    ms = int(round(seg * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def tempo_curto(seg: float) -> str:
    m, s = divmod(int(seg), 60)
    h, m = divmod(m, 60)
    return f"{h:02}:{m:02}:{s:02}" if h else f"{m:02}:{s:02}"


def transcrever(arquivos, modelo_label, idioma_label, com_tempo, progress=gr.Progress()):
    if not arquivos:
        raise gr.Error("Adicione pelo menos um arquivo de áudio.")

    progress(0, desc="Carregando modelo (na primeira vez ele é baixado)...")
    modelo = carregar_modelo(MODELOS[modelo_label])
    idioma = IDIOMAS[idioma_label]

    textos_tela, saidas = [], []
    for i, caminho in enumerate(arquivos):
        caminho = Path(caminho)
        inicio = time.time()
        segmentos, info = modelo.transcribe(
            str(caminho), language=idioma, vad_filter=True, beam_size=5
        )

        linhas_txt, blocos_srt = [], []
        for n, s in enumerate(segmentos, 1):
            texto = s.text.strip()
            linhas_txt.append(f"[{tempo_curto(s.start)}] {texto}" if com_tempo else texto)
            blocos_srt.append(f"{n}\n{tempo_srt(s.start)} --> {tempo_srt(s.end)}\n{texto}\n")
            frac = min(s.end / info.duration, 1.0) if info.duration else 0
            progress((i + frac) / len(arquivos), desc=f"{caminho.name}: {tempo_curto(s.end)} / {tempo_curto(info.duration)}")

        txt = "\n".join(linhas_txt)
        base = SAIDA_DIR / caminho.stem
        base.with_suffix(".txt").write_text(txt, encoding="utf-8")
        base.with_suffix(".srt").write_text("\n".join(blocos_srt), encoding="utf-8")
        saidas += [str(base.with_suffix(".txt")), str(base.with_suffix(".srt"))]

        dur = time.time() - inicio
        textos_tela.append(
            f"===== {caminho.name}  (idioma: {info.language}, áudio: {tempo_curto(info.duration)}, "
            f"levou {dur:.0f}s) =====\n{txt}"
        )

    return "\n\n".join(textos_tela), saidas


with gr.Blocks(title="Transcritor") as app:
    gr.Markdown("# 🎙️ Transcritor local\nArraste seus áudios (mp3, m4a, wav, ogg, opus, vídeos...). Nada sai da sua máquina.")
    with gr.Row():
        with gr.Column(scale=1):
            arquivos = gr.File(label="Áudios", file_count="multiple", type="filepath")
            modelo = gr.Dropdown(list(MODELOS), value=list(MODELOS)[0], label="Modelo")
            idioma = gr.Dropdown(list(IDIOMAS), value="Português", label="Idioma")
            com_tempo = gr.Checkbox(value=True, label="Mostrar marcação de tempo no texto")
            botao = gr.Button("Transcrever", variant="primary")
        with gr.Column(scale=2):
            saida_texto = gr.Textbox(label="Transcrição", lines=22, buttons=["copy"])
            saida_arquivos = gr.File(label="Baixar .txt / .srt", file_count="multiple")
    gr.Markdown(f"Os arquivos também ficam salvos em `{SAIDA_DIR}`.")

    botao.click(transcrever, [arquivos, modelo, idioma, com_tempo], [saida_texto, saida_arquivos])


if __name__ == "__main__":
    app.queue().launch(inbrowser="--no-browser" not in sys.argv, server_name="127.0.0.1")
