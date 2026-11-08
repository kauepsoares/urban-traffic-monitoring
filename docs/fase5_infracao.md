# Fase 5 — Detecção de avanço de sinal vermelho

Código: `src/violation.py`. Usa `src/detect_track.py` (YOLO + ByteTrack) e
`src/traffic_light.py` (Fase 4).

```powershell
python -m src.violation --config configs/aic21_cam_7.yaml                 # 30 min em ~6 min
python -m src.violation --config configs/aic21_cam_7.yaml --video         # + vídeo anotado
python -m src.violation --config configs/aic21_cam_7.yaml --red-grace 0.3 # tolerância de 0,3 s
```

A cena precisa de `stop_line` (tecla `r` no `scene_editor`), `traffic_light_roi` e, se o sentido
de entrada não for "subindo" na imagem, `stop_line_direction: out`. `exempt_zone` (opcional) é um
polígono onde a conversão no vermelho é permitida.

## Regra

1. **Cruzamento:** o ponto inferior central da caixa (contato com o chão) passa da linha de
   retenção no sentido de entrada, dentro do segmento marcado. O instante é **interpolado** entre
   os dois quadros (a 8 FPS, um carro a 50 km/h anda ~1,7 m entre quadros).
2. **Balanço da câmera:** o deslocamento medido no semáforo a cada quadro é aplicado também à
   linha de retenção (a câmera da cam_7 se desloca até ~20 px).
3. **Cor no instante do cruzamento, em retrospectiva:** a votação temporal da Fase 4 confirma a
   troca de cor com atraso (2 a 4 quadros, mais quando os quadros da transição ficam
   "desconhecidos"). Como cada cruzamento só é decidido 2 s depois, a troca é reposicionada: a nova
   cor começa no quadro seguinte ao último quadro bruto que ainda mostrava a cor antiga.
4. **Confirmação de avanço:** o veículo precisa avançar além da linha pelo menos 0,3 × a altura da
   caixa (mín. 6 px). Quem para em cima da linha fica em espera (até 60 s); se avançar depois, vale
   a cor do momento em que avançou; se não, o veredito é `parado_na_linha`.
5. **Veredito:** vermelho → `infracao`; amarelo → `amarelo`; verde → `verde`; sem leitura →
   `indeterminado`; entrou na `exempt_zone` no vermelho → `conversao_permitida`; vermelho antes da
   tolerância `--red-grace` → `vermelho_tolerancia`.
6. **Evidência:** uma imagem por infração com 3 quadros (0,5 s antes, logo após o cruzamento e no
   momento da decisão), linha e semáforo coloridos pela cor em retrospectiva, pessoas e faixa da
   placa desfocadas.

## Resultado na cam_7 (30 min, dia, `yolo11s`, imgsz 960)

| Veredito | Quantidade |
| --- | --- |
| verde | 579 |
| parado_na_linha | 23 (17 no verde, 6 no vermelho) |
| amarelo | 8 (0,1 a 3,2 s após o início do amarelo) |
| **infracao** | **1** |

Desempenho: **40,8 FPS** no pipeline completo (RTX 5060), acima dos 8 FPS do vídeo.

**A infração encontrada é um caso limite:** o carro (ID 31734, em 865,4 s) cruza a linha
**0,2 s** depois do início do vermelho, conferido quadro a quadro: no quadro 6921 os três grupos
focais estão amarelos, em 6922 já estão vermelhos, e o carro cruza entre 6922 e 6923. Com uma
tolerância de 0,3 s (comum em sistemas reais de fiscalização) ela não seria autuada. Na primeira
versão, que compensava o atraso da votação por um número fixo de quadros, esse caso saiu como
"amarelo"; foi isso que motivou o reposicionamento em retrospectiva.

Os 6 `parado_na_linha` no vermelho são carros parados na fila cuja caixa oscila sobre a linha
(avanço de 0 a 9 px). Sem a confirmação de avanço, seriam 6 falsas infrações.

## Limitações conhecidas

- **Sem gabarito ainda.** Precisão e revocação dependem de anotar manualmente os cruzamentos no
  vermelho (Fase 7). Com 1 caso positivo em 30 min, a revocação não pode ser estimada só com a
  cam_7. É preciso anotar as outras cenas (cam_7_dawn/rain, cam_10 a cam_13) e, se ainda houver
  poucos positivos, gerar casos no CARLA (plano B) para medir a revocação.
- **Rastreamento perdido logo após a linha:** 17 `parado_na_linha` aconteceram no verde, com
  avanço de até 29 px. Provavelmente o ID trocou ou sumiu antes de confirmar o avanço. No vermelho,
  isso faria uma infração passar despercebida, o que afeta a revocação.
- **Quadros irregulares no vídeo:** em alguns trechos a posição do veículo "volta" entre quadros
  seguidos (gravação a 8 FPS com quadros fora de ordem/repetidos), o que limita a precisão temporal
  a cerca de ±1 quadro (±0,125 s).
- **Detecções falsas estáticas** (ex.: um arbusto detectado como caminhão na cam_7) não cruzam a
  linha e não afetam o resultado, mas aparecem no vídeo anotado.
- O desfoque de placa é heurístico (faixa central no terço inferior da caixa); os vídeos do AIC21
  já vêm com placas e rostos borrados.
