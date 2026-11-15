# Fase 2 — Tamanho do modelo e resolução: velocidade

YOLO11 n, s e m com ByteTrack, só as classes de veículos, confiança 0,25, 1200 quadros por vídeo
(os 50 primeiros são descartados por causa do aquecimento), RTX 5060 8 GB, sem gravar vídeo.

| Vídeo | Modelo | imgsz | FPS | Detecções/quadro | IDs criados |
| --- | --- | --- | --- | --- | --- |
| cam_7 (1280x960) | yolo11n | 640 | 37,5 | 21,3 | 972 |
| cam_7 | yolo11n | 960 | 34,4 | 31,6 | 1360 |
| cam_7 | yolo11n | 1280 | 41,1 | 41,1 | 1777 |
| cam_7 | yolo11s | 640 | 39,3 | 22,9 | 1000 |
| cam_7 | yolo11s | 960 | 32,5 | 31,8 | 1091 |
| cam_7 | yolo11s | 1280 | 34,1 | 40,3 | 1349 |
| cam_10 (1920x1080) | yolo11n | 640 | 39,6 | 10,1 | 248 |
| cam_10 | yolo11n | 960 | 36,5 | 11,4 | 230 |
| cam_10 | yolo11n | 1280 | 35,2 | 15,5 | 296 |
| cam_10 | yolo11s | 640 | 38,1 | 10,7 | 200 |
| cam_10 | yolo11s | 960 | 37,9 | 11,5 | 211 |
| cam_10 | yolo11s | 1280 | 33,0 | 17,0 | 314 |
| cam_7 | yolo11m | 640 | 32,8 | 20,2 | 804 |
| cam_7 | yolo11m | 960 | 29,1 | 32,9 | 1040 |
| cam_7 | yolo11m | 1280 | 19,0 | 42,9 | 1307 |
| cam_10 | yolo11m | 640 | 33,1 | 10,1 | 212 |
| cam_10 | yolo11m | 960 | 32,4 | 14,6 | 236 |
| cam_10 | yolo11m | 1280 | 21,5 | 21,0 | 300 |

## Leitura

- **Com n e s, o FPS quase não muda (33–41)** com o modelo nem com a resolução. Nesta GPU, o
  gargalo é a decodificação do vídeo e a parte em CPU (ByteTrack, pós-processamento), não a rede.
  Só o **m a 1280 px** passa a depender da GPU (19–21 FPS). Mesmo assim, todas as combinações ficam
  acima dos 8–10 FPS dos vídeos, então **a escolha pode ser feita pela qualidade**.
- **Resolução maior encontra mais veículos:** na cam_7, 21 → 41 detecções por quadro de 640 para
  1280. São os veículos pequenos ao fundo, que importam para a linha de retenção, distante da câmera.
- **O modelo n fragmenta mais os IDs** em resolução alta (1777 IDs contra 1349 do s na cam_7 a
  1280). Mais IDs para os mesmos veículos significa trocas de ID, que causam contagem dupla e
  cruzamentos falsos. Isso justifica o yolo11s nas Fases 5 e 6. O m fragmenta ainda menos (1307 IDs
  a 1280 na cam_7, 804 a 640) e é o candidato se o gabarito mostrar ganho de qualidade.
- Medir a qualidade (precisão e revocação da contagem) depende do gabarito da Fase 7:
  `python -m src.evaluate contagem --pred <n>/crossings.csv <s>/crossings.csv --names yolo11n yolo11s`.
