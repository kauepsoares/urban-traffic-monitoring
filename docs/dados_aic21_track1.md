# AI City Challenge 2021 — Track 1 (Vehicle Counting): levantamento

Local: `data/raw/AIC21_Track1_Vehicle_Counting/` (não versionado).
Levantamento feito em 2026-11-08 com miniaturas de cada câmera e recortes da região
do semáforo em 6 instantes de cada vídeo.

## Licença (DataLicenseAgreement_AICityChallenge_2021.pdf)

- Após o fim do desafio: uso **não comercial e acadêmico** permitido, inclusive modelos treinados.
- **Proibido redistribuir** os dados, no todo ou em parte → vídeos nunca vão para o GitHub.
- Uma porção mínima pode aparecer em trabalhos acadêmicos, **com agradecimento** e **sem rostos
  ou placas visíveis** (os vídeos já vêm com rostos e placas borrados).
- A licença não proíbe anotação manual; a restrição "no labeling" valia para as regras da
  competição na época.

## Conteúdo

- 31 vídeos (~5 h) de 20 câmeras, 8 a 15 FPS (maioria 10 FPS).
- Variações de condição: `_dawn` (amanhecer), `_rain` (chuva), `_snow` (neve) para as câmeras 1 a 7.
  **Não há vídeo noturno.**
- Para cada câmera: ROI (polígono), movimentos de interesse (MOI) e uma imagem ilustrativa.
- Gabarito de contagem só para o 1º minuto da `cam_5` (amostra). O restante precisa ser anotado.
- Classes do gabarito: 1 = carro (4 rodas), 2 = caminhão. Motos e bicicletas não são contadas.

| Câmera | Resolução | FPS | Duração | Tipo de cena |
| --- | --- | --- | --- | --- |
| cam_1 (+dawn, rain) | 1280x960 | 10 | 5 min | rodovia / alça |
| cam_2 (+rain), cam_3 (+rain) | 1280x720 | 10 | 30 min | rodovia |
| cam_4 (+dawn, rain) | 1280x960 | 15 | 30 min | cruzamento (olho de peixe) |
| cam_5 (+dawn, rain) | 1280x960 | 10 | 30 min | cruzamento (olho de peixe) |
| cam_6 (+snow) | 1280x960 | 10 | 30 min | cruzamento (olho de peixe) |
| cam_7 (+dawn, rain) | 1280x960 | 8 (rain: 10) | 30 min | avenida com cruzamento semaforizado |
| cam_8, cam_9 | 1920x1080 | 10 | 5 min | cruzamento |
| cam_10, cam_11 | 1920x1080 | 10 | 3,5 min | aproximação de cruzamento semaforizado |
| cam_12, cam_13 | 1920x1080 | 10 | 3,3 min | aproximação de cruzamento semaforizado |
| cam_14 | 2560x1920 | 10 | 5 min | cruzamento urbano |
| cam_15 a cam_20 | 1920x1080 | 10 | 5 min | ruas urbanas |

## Semáforo visível (para as fases 4 e 5)

| Câmera | Semáforo legível? | Observação |
| --- | --- | --- |
| **cam_7** (+dawn, rain) | Sim, vários grupos focais, pequenos mas nítidos | Melhor candidata: 30 min + amanhecer + chuva |
| **cam_10** | Sim, grande e nítido (vermelho/amarelo/verde) | 3,5 min, dia |
| **cam_11** | Sim, grande e nítido; tem foco de conversão à esquerda separado | 3,5 min, dia |
| **cam_12** | Sim, 2 focos principais + seta de conversão | 3,3 min, dia |
| **cam_13** | Sim, 2 focos principais + foco de conversão | 3,3 min, dia |
| cam_4, cam_5, cam_6, cam_8 | Não: aparecem de lado ou de costas | Servem só para contagem |

Vistos em 6 instantes por vídeo, todos esses semáforos alternam entre vermelho, amarelo e verde.

**A verificar por câmera:** se o grupo focal visível controla de fato os veículos que cruzam a
linha de retenção marcada. Um semáforo cuja frente aparece para a câmera controla quem anda no
mesmo sentido que a câmera "olha", ou seja, quem se afasta dela. Também é preciso separar o foco
principal do foco de conversão à esquerda (cam_11, cam_12, cam_13).

## Consequências para o plano

- **Contagem (Fase 3):** pode usar qualquer câmera; cam_5 tem 1 minuto de gabarito para conferir
  o formato. As ROIs e MOIs do dataset podem ser reaproveitadas como referência.
- **Sinal vermelho (Fases 4 e 5):** cena principal = cam_7 (dia, amanhecer, chuva); cenas
  secundárias = cam_10 a cam_13. Cumpre o critério de "2 a 3 cruzamentos com semáforo visível".
- **Noite:** não há. Ou se aceita "amanhecer" como condição de pouca luz, ou se busca outra fonte.
- **FPS baixo (8–10):** a 50 km/h, um veículo anda ~1,4–1,7 m entre quadros. Isso limita a
  precisão do instante de cruzamento e deve ser discutido no TCC.
- **Pedestres (Fase 6):** poucos; cam_19 mostra pessoas atravessando a rua, sem faixa clara.
