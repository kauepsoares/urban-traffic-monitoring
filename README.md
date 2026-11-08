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
python -m src.detect_track --source data/clips/cruzamento.mp4 --model models/yolo11s.pt --no-video

# 3. Contagem de veículos por linha virtual (janelas de 60 s)
python -m src.count --source data/clips/cruzamento.mp4 --config configs/cena_01.yaml --window 60
```

Teste rápido com o vídeo de exemplo (rodovia, sem semáforo):

```powershell
python -m src.count --source data/raw/exemplos/vehicles.mp4 --config configs/exemplo_vehicles.yaml --window 5
```

### Semáforo (Fase 4)

```powershell
# Estado do semáforo por quadro (não usa YOLO; 30 min de vídeo em ~45 s)
python -m src.traffic_light --config configs/aic21_cam_7.yaml
```

Compensa o balanço da câmera por correlação de template, decide a cor pela contagem de pixels
coloridos em HSV e suaviza no tempo respeitando o ciclo verde → amarelo → vermelho.
Gera `states.csv`, `phases.csv`, `timeline.png` e `samples.jpg`. Método, tentativas descartadas
e resultados em [docs/fase4_semaforo.md](docs/fase4_semaforo.md).

### Avanço de sinal vermelho (Fase 5)

```powershell
# YOLO + ByteTrack + semáforo; precisa de stop_line e traffic_light_roi na cena
python -m src.violation --config configs/aic21_cam_7.yaml            # --video para o vídeo anotado
```

Um veículo comete infração quando o ponto inferior central da caixa cruza a linha de retenção,
no sentido de entrada, com o sinal vermelho (instante interpolado entre quadros, atraso da
suavização compensado, linha acompanhando o balanço da câmera). Quem para sobre a linha não é
contado até avançar. Gera `crossings.csv`, `violations.csv` e uma imagem de evidência por infração
(3 quadros, pessoas e placas desfocadas). Detalhes em [docs/fase5_infracao.md](docs/fase5_infracao.md).

**Atenção:** o semáforo marcado precisa ser o que a fila obedece. Um semáforo cuja frente aparece
na imagem controla quem se afasta da câmera; se a fila vem em direção à câmera, o semáforo dela
está fora da imagem. Cada cena declara isso em `light_validity` (`controla`, `substituto`,
`nao_controla`, `nao_verificado`); cenas `nao_controla` são recusadas.

### Contagem (Fase 3)

- Cada ID que cruza a `counting_line` é contado **uma vez**, pelo ponto inferior central da caixa
  (contato com o chão).
- Sentidos: para uma linha marcada da esquerda para a direita, `in` = cruza subindo na imagem,
  `out` = descendo. Os nomes vêm de `direction_names` no YAML da cena.
- A classe final é a **mais votada** em todos os quadros do ID (o YOLO às vezes alterna
  carro/caminhão); a classe no instante do cruzamento fica salva em `class_at_crossing`.
- Saídas: `crossings.csv` (um evento por veículo), `counts_total.csv`, `counts_per_window.csv`,
  `annotated.mp4` com placar e `run_info.yaml`.

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
  count.py         Fase 3: contagem de veículos por classe, sentido e janela de tempo
  traffic_light.py Fase 4: estado do semáforo por quadro (HSV + suavização)
  violation.py     Fase 5: avanço de sinal vermelho + evidências
  evaluate.py      Fase 7: comparação com anotação manual (a fazer)
docs/              levantamento dos dados e notas de método por fase
notebooks/         testes e gráficos
outputs/           CSVs e vídeos anotados (não versionados)
```

## Andamento

- [x] Fase 0 — Ambiente (PyTorch cu128, GPU) e repositório
- [ ] Fase 1 — Dados (datasets + configuração das cenas) — ferramenta de marcação pronta
- [x] Fase 2 — Detecção e rastreamento (falta comparar n/s/m em vídeos reais)
- [x] Fase 3 — Contagem de veículos (validada só no vídeo de exemplo; falta cena real)
- [x] Fase 4 — Estado do semáforo, versão 1 por regras HSV (falta gabarito e a versão 2 treinada)
- [x] Fase 5 — Avanço de sinal vermelho (avaliação só na cam_7 dia/amanhecer/chuva, 40 min: 1 infração limite; falta gabarito e mais positivos)
- [ ] Fase 6 — Contagem de pedestres
- [ ] Fase 7 — Avaliação
- [ ] Fase 8 — Escrita e defesa

## Privacidade

O sistema não identifica placas nem rostos. Quadros de evidência salvos devem ser desfocados (LGPD).
