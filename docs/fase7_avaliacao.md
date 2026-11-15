# Fase 7 — Avaliação contra gabarito manual

Código: `src/annotate.py` (anotação) e `src/evaluate.py` (métricas).

## Por que anotar eventos, e não caixas

As três perguntas do TCC são sobre **eventos no tempo**: quando o semáforo troca de cor, quando um
veículo cruza uma linha e se esse cruzamento aconteceu no vermelho. Anotar caixas quadro a quadro
no CVAT custaria horas por minuto de vídeo e mediria outra coisa (a qualidade do detector). Por isso
o gabarito é uma lista de eventos com o quadro em que aconteceram, marcada com uma tecla enquanto o
vídeo roda (e quadro a quadro nos instantes duvidosos).

O anotador não vê as previsões do sistema, para não ser induzido. Só contam os quadros que ele
assistiu em sequência; o arquivo guarda esses intervalos e a avaliação ignora o resto.

## Protocolo de anotação

Arquivos em `data/annotations/<cena>/<tarefa>.yaml`, versionados no git (são trabalho próprio,
não dados do AIC21).

### 1. Semáforo (`--task semaforo`)

```powershell
python -m src.annotate --config configs/aic21_cam_7.yaml --task semaforo --annotator "Kauê"
```

- Marque o **primeiro quadro** em que a nova cor aparece acesa: `g` verde, `y` amarelo,
  `r` vermelho, `k` desconhecido (semáforo encoberto ou ilegível). Marque também o estado do
  início do vídeo, no quadro 0.
- Rode em 4x ou 8x até perto de uma troca, pause (`espaço`) e ande quadro a quadro (`a`/`d`) olhando o
  recorte ampliado no canto.
- Quadro de transição ambíguo (as duas lâmpadas meio acesas): considere que a nova cor começa no
  primeiro quadro em que ela está **mais acesa** que a antiga.
- Custo estimado: um ciclo da cam_7 dura ~130 s, então 30 min dão ~14 ciclos e ~42 trocas. Leva
  15 a 20 min.

### 2. Cruzamentos da linha de retenção (`--task retencao`)

```powershell
python -m src.annotate --config configs/aic21_cam_7.yaml --task retencao `
    --light-gt data/annotations/aic21_cam_7/semaforo.yaml --skip-green
```

- Um evento por veículo que cruza a `stop_line` entrando no cruzamento, no quadro em que a
  **frente do veículo** passa a linha (definição do CTB para o avanço de sinal). Tecla da classe:
  `1` carro, `2` moto, `3` ônibus, `4` caminhão.
- Com `--skip-green`, o vídeo pula as fases verdes do gabarito, mantendo 3 s no início e no fim de
  cada uma (`--margin`). Restam os trechos amarelo e vermelho (~40% do vídeo) e as transições,
  justamente onde as infrações acontecem. `n` pula para o próximo trecho.
- Veículo da faixa de conversão à esquerda: marque normalmente, porque ele cruza a linha. A
  avaliação mostra quantos desses o sistema descarta.
- Para estimar também a revocação dos cruzamentos no verde, anote sem `--skip-green` uma amostra
  (por exemplo os 5 primeiros minutos).

### 3. Contagem (`--task contagem`)

Precisa de `counting_line` na cena (`scene_editor`, tecla `c`). A tecla `t` alterna o sentido
(`in` = sobe na imagem, para uma linha marcada da esquerda para a direita).

```powershell
python -m src.annotate --config configs/aic21_cam_10.yaml --task contagem
```

Sugestão de amostra: 5 min por cena, em 3 a 4 cenas de condições diferentes (cam_7 dia, cam_7_rain,
cam_7_dawn, uma das cam_10 a cam_13).

## Métricas

```powershell
# Semáforo: states.csv da Fase 4 (ou light_states.csv da Fase 5)
python -m src.evaluate semaforo --gt data/annotations/aic21_cam_7/semaforo.yaml `
    --pred outputs/cam_7_semaforo/states.csv

# Contagem / detecção de cruzamentos (aceita vários --pred, para comparar modelos)
python -m src.evaluate contagem --gt data/annotations/aic21_cam_10/contagem.yaml `
    --pred outputs/cam_10_yolo11n_contagem/crossings.csv outputs/cam_10_yolo11s_contagem/crossings.csv `
    --names yolo11n yolo11s

# Avanço de sinal vermelho
python -m src.evaluate infracao --gt data/annotations/aic21_cam_7/retencao.yaml `
    --light-gt data/annotations/aic21_cam_7/semaforo.yaml `
    --pred outputs/full/cam_7_infracoes/crossings.csv
```

Saídas em `outputs/avaliacao/<cena>_<tarefa>/`: `metrics.yaml` e CSVs com os pares, para
discutir cada erro no texto.

| Avaliação | O que mede | Detalhes |
| --- | --- | --- |
| semáforo | acurácia por quadro, cobertura, precisão e revocação por cor, matriz de confusão | Em três versões: `raw` (quadro a quadro), `state` (suavizado, em tempo real) e `retro` (trocas reposicionadas, como a Fase 5 consulta). Quadros `desconhecido` no gabarito ficam de fora. |
| semáforo, trocas | erro (s) no instante de cada troca de cor, trocas perdidas e falsas | Erro positivo = o sistema troca depois do gabarito. Pareamento dentro de ±3 s, para a mesma cor nova. |
| contagem | precisão, revocação e F1 dos eventos, erro de contagem por classe e sentido, acerto de classe, erro absoluto médio por janela | Pareamento um para um pelo instante (±1 s, `--tol`), no mesmo sentido. `--merge-heavy` junta ônibus e caminhão. |
| infração | precisão e revocação das infrações; precisão e revocação dos cruzamentos; tabela cor do gabarito x veredito | Infração no gabarito = cruzamento (frente do veículo) com o semáforo do gabarito vermelho há pelo menos `--red-grace` s. |

O pareamento é guloso pela menor diferença de tempo. A tolerância de 1 s cobre a diferença de
definição: o gabarito marca a frente do veículo e o sistema usa o ponto inferior central da caixa.

## Atenção: frente x traseira do veículo

Na cam_7 os veículos se afastam da câmera. Assim, o ponto inferior central da caixa, que o sistema
usa como cruzamento, fica na **traseira** do veículo. O sistema registra o cruzamento quando o
veículo inteiro já passou da linha, alguns décimos de segundo depois de a frente cruzar (um carro de
4,5 m a 30 km/h leva ~0,5 s). Perto da troca amarelo → vermelho, isso pode transformar em "infração"
um carro cuja frente cruzou ainda no amarelo. É o caso da única infração detectada até agora
(865,4 s, 0,2 s depois do vermelho). A avaliação `infracao` mede esse efeito, porque o gabarito usa
a frente.

## Validação do código

`evaluate.py` foi testado com um gabarito sintético derivado das próprias saídas da cam_7 (trocas
de cor adiantadas 2 quadros, cruzamentos com ruído de 0,3 s, 5% removidos e 3 falsos). As métricas
batem com as perturbações: a versão `state` troca 0,250 s (2 quadros) depois, a revocação dos
cruzamentos fica em ~98%, e assim por diante. Os números de verdade dependem do gabarito manual.

## Resultados

_(a preencher depois da anotação)_
