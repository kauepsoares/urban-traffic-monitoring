# Fase 6 — Contagem de pedestres

Código: `src/pedestrians.py`. Cena: `configs/aic21_cam_19.yaml`.

```powershell
python -m src.pedestrians --config configs/aic21_cam_19.yaml            # 5 min em ~2,5 min
python -m src.pedestrians --config configs/aic21_cam_19.yaml --video    # + vídeo anotado
```

## Levantamento: não há faixa de pedestres no AIC21

Rodei o YOLO11s (1280 px) em um quadro a cada 5 s de todos os 31 vídeos e conferi visualmente
as cenas com mais detecções:

- **Nenhuma câmera mostra faixa de pedestres marcada.** Os pedestres andam pelas calçadas e, de vez
  em quando, atravessam a rua no meio do quarteirão.
- Cenas com mais pessoas: cam_19 (55 de 60 amostras com pessoas), cam_18, cam_17, cam_20 e cam_14.
  Nas cam_14 e cam_20 as pessoas aparecem cortadas, perto da câmera.
- Na cam_7 as "pessoas" são **falsos positivos** (placas e postes ao longe). Isso reforça que a
  cam_7 não serve para pedestres.

Por isso o módulo conta **pessoas cruzando linhas**, com a mesma lógica da contagem de veículos
(Fase 3). O modo "faixa" (`crosswalk_zone`) fica implementado para cenas que tenham faixa.

## Método

- Detecção e rastreamento: YOLO11s + ByteTrack, só a classe pessoa, `imgsz` 1280 (pessoas ao
  fundo têm ~20 px de altura) e confiança 0,25.
- **Linhas** (`pedestrian_lines`, por nome). Cada ID é contado uma vez por linha quando os pés
  (ponto inferior central da caixa) cruzam a linha. O sentido `in` é o lado esquerdo de quem olha
  de A para B; uma seta no vídeo indica esse lado.
  - `travessia_rua`: linha no eixo da rua. Quem a cruza atravessou a rua, com ou sem faixa.
  - `calcada_direita`: linha atravessando a calçada. Mede o fluxo de pedestres.
- **Faixa** (`crosswalk_zone`, opcional). Para cada ID que entra no polígono, mede o tempo dentro e
  o deslocamento ao longo do eixo maior da faixa. É travessia se percorrer ≥ 40% do comprimento;
  senão, é só presença (ex.: alguém esperando na esquina). Também gera a ocupação por quadro.

As linhas da cam_19 foram propostas a partir das trajetórias de todas as pessoas do vídeo (figura
de trajetórias) e precisam ser conferidas no `scene_editor` (tecla `w`, `--ped-line <nome>`).

## Resultado (cam_19, 5 min, dia)

| Linha | in | out | Conferência |
| --- | --- | --- | --- |
| travessia_rua | 1 | 2 | 3 travessias no meio do quarteirão, conferidas nos recortes: um pedestre sozinho (165 s) e uma dupla (152 s) |
| calcada_direita | 6 | 2 | Bate com as trajetórias brutas que cruzam a linha (inclui um grupo de 4 pessoas aos 144 s) |

119 IDs de pessoa no vídeo; desempenho de 20 FPS com `imgsz` 1280 (o vídeo tem 10 FPS).

Esses números ainda não são uma avaliação: a conferência usa o próprio rastreador. A medida de
verdade vem do gabarito da Fase 7:

```powershell
python -m src.annotate --config configs/aic21_cam_19.yaml --task contagem --line calcada_direita
python -m src.evaluate contagem --gt data/annotations/aic21_cam_19/pedestres_calcada_direita.yaml `
    --pred outputs/cam_19_pedestres/crossings.csv
```

## Limitações

- **Poucos eventos:** 3 travessias e 8 passagens na calçada em 5 min. Dá para demonstrar o método,
  mas não para estimar precisão e revocação com segurança. Juntar as cam_17 e cam_18 aumenta a
  amostra (as duas têm fluxo na calçada).
- **Grupos:** pessoas lado a lado podem virar uma caixa só ou trocar de ID, o que subconta. O grupo
  de 4 da cam_19 saiu certo.
- **Pessoas encobertas** por carros estacionados (calçada da direita na cam_19) perdem o ID; se o
  ID trocar exatamente na linha, a pessoa pode ser contada duas vezes ou nenhuma.
- **Sem faixa marcada:** não é possível medir "respeito à faixa" nem conflitos entre veículo e
  pedestre na faixa. Fica como trabalho futuro com outra fonte de vídeo.
