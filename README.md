# TCC — Monitoramento de tráfego e detecção de avanço de sinal vermelho

Sistema de visão computacional que analisa vídeos de câmeras fixas em cruzamentos para
contar veículos, contar pedestres e detectar veículos que avançam o sinal vermelho.

Base: **YOLO (Ultralytics) + ByteTrack**, com módulos de contagem (supervision),
classificação do semáforo e detecção de infração.

## Ambiente

Desenvolvido em Windows 11, Python 3.13, RTX 5060 8 GB (precisa de PyTorch com CUDA 12.8+).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -m src.check_env      # deve mostrar "CUDA disponível: True"
```

## Uso

```powershell
# 1. Marcar linhas/zonas de uma cena (gera configs/cena_01.yaml)
python -m src.scene_editor --video data/clips/cruzamento.mp4 --out configs/cena_01.yaml

# 2. Detecção + rastreamento (gera outputs/<video>_<modelo>/tracks.csv e annotated.mp4)
python -m src.detect_track --source data/clips/cruzamento.mp4 --config configs/cena_01.yaml

# Comparar tamanhos de modelo (qualidade x FPS)
python -m src.detect_track --source data/clips/cruzamento.mp4 --model yolo11s.pt --no-video
```

## Estrutura

```
data/raw/          vídeos originais dos datasets (não versionados)
data/clips/        trechos selecionados para o TCC
data/annotations/  gabarito manual (CVAT)
configs/           cena_XX.yaml: linhas, zonas e região do semáforo de cada vídeo
src/
  config.py        formato das cenas e classes usadas
  check_env.py     Fase 0: verifica GPU e bibliotecas
  scene_editor.py  Fase 1: marca regiões clicando no vídeo
  detect_track.py  Fase 2: YOLO + ByteTrack -> CSV de trajetórias + vídeo anotado
  count.py         Fase 3: contagem de veículos (a fazer)
  traffic_light.py Fase 4: estado do semáforo (a fazer)
  violation.py     Fase 5: avanço de sinal (a fazer)
  evaluate.py      Fase 7: comparação com anotação manual (a fazer)
notebooks/         testes e gráficos
outputs/           CSVs e vídeos anotados (não versionados)
```

## Andamento

- [x] Fase 0 — Ambiente (PyTorch cu128, GPU) e repositório
- [ ] Fase 1 — Dados (datasets + configuração das cenas) — ferramenta de marcação pronta
- [x] Fase 2 — Detecção e rastreamento (falta comparar n/s/m em vídeos reais)
- [ ] Fase 3 — Contagem de veículos
- [ ] Fase 4 — Estado do semáforo
- [ ] Fase 5 — Avanço de sinal vermelho
- [ ] Fase 6 — Contagem de pedestres
- [ ] Fase 7 — Avaliação
- [ ] Fase 8 — Escrita e defesa

## Privacidade

O sistema não identifica placas nem rostos. Quadros de evidência salvos devem ser desfocados (LGPD).
