# TCC — Monitoramento de tráfego e detecção de avanço de sinal vermelho

Sistema de visão computacional que analisa vídeos de câmeras fixas em cruzamentos para
contar veículos, contar pedestres e detectar veículos que avançam o sinal vermelho.

Base: **YOLO (Ultralytics) + ByteTrack**, com módulos de contagem (supervision),
classificação do semáforo e detecção de infração.

## Como rodar o projeto

Desenvolvido em Windows 11, Python 3.13 e RTX 5060 8 GB. Funciona com qualquer GPU NVIDIA com
CUDA 12.8+ e, mais devagar, só na CPU (`--device cpu`). Todos os comandos rodam na raiz do
repositório, no PowerShell.

### 1. Ambiente Python

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # se o PowerShell bloquear: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # GPUs RTX 50xx precisam do cu128
pip install -r requirements.txt
python -m src.check_env               # deve mostrar "CUDA disponível: True"
```

Sem GPU, instale o PyTorch padrão (`pip install torch torchvision`) e acrescente `--device cpu` aos
comandos.

### 2. Pesos do YOLO

Os pesos não são versionados (`models/` está no `.gitignore`). Para baixar os três tamanhos
usados no TCC do repositório oficial da Ultralytics (~64 MB no total):

```powershell
cd models
python -c "from ultralytics.utils.downloads import attempt_download_asset as d; [d(m) for m in ('yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt')]"
cd ..
```

### 3. Dados (AI City Challenge 2021, Track 1)

Os vídeos não podem ser redistribuídos (licença do dataset) e não estão no repositório. Peça o
acesso ao *AIC21 Track 1 — Vehicle Counting* no site do AI City Challenge
(<https://www.aicitychallenge.org/>), aceite a licença e extraia o pacote assim:

```
data/raw/AIC21_Track1_Vehicle_Counting/
    Dataset_A/cam_7.mp4, cam_7_dawn.mp4, cam_7_rain.mp4, cam_10.mp4 ... cam_19.mp4
    ROIs/  movement_description/  counting_gt_sample/  ...
```

As cenas em `configs/aic21_*.yaml` já apontam para esses caminhos. Para usar outro vídeo, crie a
cena com o `scene_editor` (seção "Uso" abaixo).

### 4. Reproduzir os resultados do TCC

Na ordem; cada comando grava em `outputs/` (tempos medidos na RTX 5060):

```powershell
# Fase 4 — estado do semáforo (cam_7, 30 min de vídeo em ~45 s)
python -m src.traffic_light --config configs/aic21_cam_7.yaml

# Fase 5 — avanço de sinal vermelho nas 3 cenas válidas (~1 min, ~1 min, ~7 min)
python -m src.violation --config configs/aic21_cam_7_dawn.yaml
python -m src.violation --config configs/aic21_cam_7_rain.yaml
python -m src.violation --config configs/aic21_cam_7.yaml

# Fase 6 — pedestres na cam_19 (~2,5 min; --video grava o vídeo anotado)
python -m src.pedestrians --config configs/aic21_cam_19.yaml --video

# Fase 3 — contagem de veículos (precisa de counting_line na cena; teste com o vídeo de exemplo)
python -m src.count --source data/raw/exemplos/vehicles.mp4 --config configs/exemplo_vehicles.yaml --window 5
```

O que conferir em cada saída:

| Saída | Arquivos principais |
| --- | --- |
| `outputs/cam_7_semaforo/` | `states.csv` (estado por quadro), `phases.csv`, `timeline.png`, `samples.jpg` |
| `outputs/cam_7*_infracoes/` | `crossings.csv` (todos os cruzamentos com veredito), `violations.csv`, `evidence/`, `run_info.yaml` |
| `outputs/cam_19_pedestres/` | `crossings.csv` (coluna `line`), `counts_total.csv`, `annotated.mp4` |
| `outputs/*_contagem/` | `crossings.csv`, `counts_total.csv`, `counts_per_window.csv`, `annotated.mp4` |

### 5. Avaliação contra o gabarito (Fase 7)

O gabarito é anotado à mão com `src.annotate` (uma janela do OpenCV: assista ao vídeo e aperte uma
tecla em cada evento; os controles aparecem no topo da tela) e salvo em `data/annotations/`.
Depois, o `src.evaluate` calcula as métricas em `outputs/avaliacao/`. Os comandos estão na seção
"Avaliação (Fase 7)" abaixo, e o protocolo, em [docs/fase7_avaliacao.md](docs/fase7_avaliacao.md).

### Problemas comuns

- **`CUDA disponível: False`**: o PyTorch instalado é o de CPU. Reinstale com o índice `cu128`
  (passo 1).
- **`Não foi possível abrir data/raw/...`**: o vídeo não está no caminho esperado (passo 3).
- **`A cena ... não tem stop_line / counting_line / pedestrian_lines`**: marque a região com
  `python -m src.scene_editor --video <vídeo> --out <cena>.yaml` (teclas `r`, `c`, `w`).
- **`light_validity: nao_controla`**: a cena foi recusada de propósito, porque o semáforo visível
  não controla aquela fila (ver Fase 5).

## Uso

```powershell
# 1. Marcar linhas/zonas de uma cena (gera configs/cena_01.yaml)
python -m src.scene_editor --video data/clips/cruzamento.mp4 --out configs/cena_01.yaml

# 2. Detecção + rastreamento (gera outputs/<video>_<modelo>/tracks.csv e annotated.mp4)
python -m src.detect_track --source data/clips/cruzamento.mp4 --config configs/cena_01.yaml

# Comparar tamanhos de modelo (FPS: docs/fase2_modelos.md)
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

Um veículo comete infração quando a frente (estimada a partir do ponto inferior central da caixa e do comprimento da classe; ver doc) cruza a linha de retenção,
no sentido de entrada, com o sinal vermelho (instante interpolado entre quadros, atraso da
suavização compensado, linha acompanhando o balanço da câmera). Quem para sobre a linha não é
contado até avançar. Gera `crossings.csv`, `violations.csv` e uma imagem de evidência por infração
(3 quadros, pessoas e placas desfocadas). Detalhes em [docs/fase5_infracao.md](docs/fase5_infracao.md).

**Atenção:** o semáforo marcado precisa ser o que a fila obedece. Um semáforo cuja frente aparece
na imagem controla quem se afasta da câmera; se a fila vem em direção à câmera, o semáforo dela
está fora da imagem. Cada cena declara isso em `light_validity` (`controla`, `substituto`,
`nao_controla`, `nao_verificado`); cenas `nao_controla` são recusadas.

### Pedestres (Fase 6)

```powershell
# conta pessoas cruzando as pedestrian_lines (eixo da rua, calçada) e/ou na crosswalk_zone
python -m src.pedestrians --config configs/aic21_cam_19.yaml --video
```

O AIC21 não tem faixa de pedestres marcada; na cam_19 a linha `travessia_rua` (eixo da rua) pega
as travessias no meio do quarteirão e `calcada_direita` mede o fluxo na calçada. Detalhes em
[docs/fase6_pedestres.md](docs/fase6_pedestres.md).

### Avaliação (Fase 7)

```powershell
# 1. Gabarito por eventos: tecla no instante de cada troca de cor / cruzamento
python -m src.annotate --config configs/aic21_cam_7.yaml --task semaforo
python -m src.annotate --config configs/aic21_cam_7.yaml --task retencao `
    --light-gt data/annotations/aic21_cam_7/semaforo.yaml --skip-green

# 2. Métricas (semaforo | contagem | infracao)
python -m src.evaluate semaforo --gt data/annotations/aic21_cam_7/semaforo.yaml --pred outputs/cam_7_semaforo/states.csv
```

Protocolo de anotação e métricas em [docs/fase7_avaliacao.md](docs/fase7_avaliacao.md).

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
data/annotations/  gabarito manual por eventos (src.annotate; versionado)
configs/           cena_XX.yaml: linhas, zonas e região do semáforo de cada vídeo
src/
  config.py        formato das cenas e classes usadas
  check_env.py     Fase 0: verifica GPU e bibliotecas
  scene_editor.py  Fase 1: marca regiões clicando no vídeo
  detect_track.py  Fase 2: YOLO + ByteTrack -> CSV de trajetórias + vídeo anotado
  count.py         Fase 3: contagem de veículos por classe, sentido e janela de tempo
  traffic_light.py Fase 4: estado do semáforo por quadro (HSV + suavização)
  violation.py     Fase 5: avanço de sinal vermelho + evidências
  pedestrians.py   Fase 6: contagem de pedestres por linhas e faixa
  annotate.py      Fase 7: anotação do gabarito (eventos com instante)
  evaluate.py      Fase 7: métricas contra o gabarito
docs/              levantamento dos dados e notas de método por fase
notebooks/         testes e gráficos
outputs/           CSVs e vídeos anotados (não versionados)
```

## Andamento

- [x] Fase 0 — Ambiente (PyTorch cu128, GPU) e repositório
- [ ] Fase 1 — Dados (datasets + configuração das cenas) — ferramenta de marcação pronta
- [x] Fase 2 — Detecção e rastreamento (FPS de n/s/m medido; falta qualidade com gabarito)
- [x] Fase 3 — Contagem de veículos (validada só no vídeo de exemplo; falta cena real)
- [x] Fase 4 — Estado do semáforo, versão 1 por regras HSV (falta gabarito e a versão 2 treinada)
- [x] Fase 5 — Avanço de sinal vermelho (avaliação só na cam_7 dia/amanhecer/chuva; cruzamento pelo instante estimado da frente do veículo; 0 infrações em 40 min; sem CARLA; falta gabarito)
- [x] Fase 6 — Contagem de pedestres (cam_19: travessias e calçada; falta gabarito)
- [ ] Fase 7 — Avaliação (ferramentas prontas; falta anotar o gabarito e rodar)
- [ ] Fase 8 — Escrita e defesa (rascunho em [docs/monografia_rascunho.md](docs/monografia_rascunho.md))

## Privacidade

O sistema não identifica placas nem rostos. Quadros de evidência salvos devem ser desfocados (LGPD).
