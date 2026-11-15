# Monitoramento de tráfego por visão computacional: contagem de veículos e pedestres e detecção de avanço de sinal vermelho em câmeras fixas

> **Rascunho da monografia (Fase 8).** A estrutura segue o padrão ABNT de TCC. O texto foi
> montado a partir de `docs/fase*.md`. Marcações:
> `[REF: ...]` = citação a conferir e formatar; `[RESULTADO FASE 7]` = número que depende do
> gabarito manual; `[FIGURA: ...]` = figura a gerar (o caminho indica a origem).
> Para converter: `pandoc docs/monografia_rascunho.md -o monografia.docx` (ou `-o monografia.tex`).

---

## Resumo

Este trabalho apresenta um sistema de visão computacional para analisar vídeos de câmeras fixas em
vias urbanas. O sistema conta veículos por classe e sentido, conta pedestres e detecta veículos que
avançam o sinal vermelho. A detecção usa o YOLO11 e o rastreamento, o ByteTrack. O estado do
semáforo é classificado por regras de cor em HSV, com compensação do balanço da câmera e suavização
temporal restrita ao ciclo verde → amarelo → vermelho. O avanço de sinal é decidido pelo instante,
interpolado entre quadros, em que a frente do veículo cruza a linha de retenção, estimado a partir
da trajetória. Os experimentos usam o conjunto AI City Challenge 2021 (Track 1), em condições de
dia, amanhecer e chuva. [RESULTADO FASE 7: acurácia do semáforo, F1 da contagem, cruzamentos.] Um
teste empírico mostrou que, em quatro das cinco câmeras com semáforo visível, o semáforo que aparece
na imagem não é o que controla a fila observada. Isso restringe a avaliação de infrações a uma única
aproximação e expõe uma limitação pouco discutida em trabalhos semelhantes.

**Palavras-chave:** visão computacional; YOLO; rastreamento de objetos; semáforo; fiscalização de
trânsito.

## Abstract

[Tradução do resumo.]

---

## 1 Introdução

### 1.1 Contexto e motivação

- Avançar o sinal vermelho é infração gravíssima no Código de Trânsito Brasileiro (art. 208 da Lei
  nº 9.503/1997) [REF: CTB] e está entre as causas de colisões em cruzamentos [REF: dado de
  sinistros em interseções semaforizadas].
- A fiscalização eletrônica tradicional exige sensores na pista (laços indutivos) e equipamentos
  dedicados. Câmeras de monitoramento já instaladas poderiam, em princípio, fornecer contagens e
  indícios de infração só com processamento de imagem.
- Detectores de objetos em tempo real (família YOLO) e rastreadores multiobjeto (ByteTrack) tornaram
  esse processamento viável em uma GPU de consumo.

### 1.2 Objetivos

**Geral:** desenvolver e avaliar um sistema que, a partir do vídeo de uma câmera fixa, conte veículos
e pedestres e detecte avanços de sinal vermelho.

**Específicos:**
1. Detectar e rastrear veículos e pessoas (YOLO11 + ByteTrack) e comparar tamanhos de modelo e
   resoluções.
2. Contar veículos por classe, sentido e janela de tempo com uma linha virtual.
3. Classificar o estado do semáforo quadro a quadro, sem treinamento, de forma robusta a balanço da
   câmera, amanhecer e chuva.
4. Detectar avanços de sinal vermelho e gerar evidências com dados pessoais desfocados (LGPD).
5. Contar pedestres que atravessam a via e que circulam pela calçada.
6. Avaliar cada módulo contra um gabarito anotado manualmente.

### 1.3 Organização do texto

[Um parágrafo descrevendo os capítulos.]

## 2 Fundamentação teórica

### 2.1 Detecção de objetos com YOLO

- Detecção em estágio único: grade, caixas e classes em uma passada [REF: Redmon et al., 2016, "You
  Only Look Once"]. Evolução até o YOLO11 (Ultralytics) [REF: documentação/repositório Ultralytics].
- Métricas de detecção (IoU, precisão, revocação) e a troca entre resolução de entrada e objetos
  pequenos.

### 2.2 Rastreamento multiobjeto

- Rastreamento por detecção; filtro de Kalman; associação húngara.
- ByteTrack: associa também as detecções de baixa confiança [REF: Zhang et al., 2022, ByteTrack,
  ECCV]. Efeito do "buffer" de IDs perdidos (~30 quadros), que reaparece na Fase 5 como troca de ID.

### 2.3 Espaço de cor HSV e classificação de semáforos

- Separação de matiz, saturação e brilho; por que a decisão pela cor funciona melhor que a decisão
  pela posição da lâmpada (Seção 4.4).
- Abordagens treinadas (bases LISA e Bosch) [REF: bases de semáforos], citadas como alternativa.

### 2.4 Fiscalização de avanço de sinal

- Definição legal: o veículo avança o sinal quando a frente passa a linha de retenção com o vermelho
  aceso [REF: CTB e resolução do CONTRAN sobre fiscalização eletrônica].
- Sistemas comerciais: laços ou radar + câmera; tolerâncias após o início do vermelho.

### 2.5 Privacidade

- LGPD (Lei nº 13.709/2018) [REF: LGPD]: minimização e desfoque de rostos e placas nas evidências.

## 3 Trabalhos relacionados

[REF: 4 a 8 trabalhos sobre contagem de veículos com YOLO/DeepSORT/ByteTrack, detecção de avanço de
sinal por vídeo e classificação de semáforos. Para cada um: dados usados, se o semáforo é lido da
imagem ou recebido do controlador, e como a infração é avaliada.]

Ponto a destacar na comparação: trabalhos que leem o semáforo da imagem raramente verificam se ele
controla a fila analisada (Seção 5.4).

## 4 Materiais e métodos

### 4.1 Ambiente

Windows 11, Python 3.13, PyTorch com CUDA 12.8, GPU RTX 5060 8 GB. Bibliotecas: Ultralytics
(YOLO11), supervision, OpenCV, pandas. Código organizado em módulos por fase (`src/`).

### 4.2 Dados: AI City Challenge 2021, Track 1

- 31 vídeos (~5 h), 20 câmeras, 8 a 15 FPS, com rostos e placas já desfocados; uso acadêmico
  permitido e redistribuição proibida [REF: Naphade et al., 2021, AI City Challenge, CVPRW]
  (`docs/dados_aic21_track1.md`).
- Cenas usadas:

| Uso | Cenas | Condições |
| --- | --- | --- |
| Semáforo (Fase 4) | cam_7, cam_7_dawn, cam_7_rain, cam_10 a cam_13 | dia, amanhecer, chuva |
| Avanço de sinal (Fase 5) | cam_7, cam_7_dawn, cam_7_rain | dia, amanhecer, chuva |
| Pedestres (Fase 6) | cam_19 | dia |
| Contagem (Fase 3) | [cenas com `counting_line` marcada] | |

- Não há vídeo noturno; o amanhecer faz o papel de condição de pouca luz.

### 4.3 Detecção, rastreamento e contagem (Fases 2 e 3)

- YOLO11 (n, s, m) + ByteTrack, classes COCO carro, moto, ônibus, caminhão e pessoa.
- Contagem: cada ID é contado uma vez quando o ponto inferior central da caixa cruza a linha
  virtual; o sentido é dado pelo lado do cruzamento; a classe final é a mais votada ao longo da
  trajetória (corrige a alternância carro/caminhão).
- [FIGURA: cena com linha de contagem e placar — `outputs/*_contagem/annotated.mp4`]

### 4.4 Estado do semáforo (Fase 4)

1. Localização com compensação de balanço: correlação de template sobre um mapa de silhueta
   `max(255 − V, S)` em janela de ±6 px, limitada a ±25 px da referência.
2. Cor acesa pela contagem de pixels saturados e brilhantes por faixa de matiz; a vencedora precisa
   de ≥ 0,5% da área e o dobro da segunda.
3. Suavização: maioria de 5 quadros, aceita só se seguir o ciclo; troca fora do ciclo só após 3 s.
- Tabela das tentativas descartadas e do que cada uma ensinou (`docs/fase4_semaforo.md`).
- [FIGURA: `outputs/cam_7_semaforo/timeline.png` e `samples.jpg`]

### 4.5 Avanço de sinal vermelho (Fase 5)

- Cruzamento da linha de retenção, no sentido de entrada, com instante interpolado entre quadros e
  linha acompanhando o balanço da câmera.
- Filtros contra cruzamentos falsos: pulo de ID, movimento transversal e trajetória incompatível
  depois da linha (cada um motivado por um erro observado).
- **Instante da frente do veículo.** O ponto inferior da caixa é o contato com o chão mais próximo
  da câmera; quando o veículo se afasta dela, esse ponto é a traseira. A frente toca a linha quando
  a traseira está a um comprimento de veículo antes dela. O instante é procurado no histórico da
  trajetória, com a escala px/m perto da linha (calibrada pelo espaçamento de filas paradas,
  ~5,5 px/m) e o comprimento típico da classe.
- Cor no instante do cruzamento, com a troca de cor reposicionada em retrospectiva para compensar o
  atraso da votação temporal.
- Confirmação de avanço (quem para sobre a linha fica em espera) e vereditos: infração, amarelo,
  verde, parado na linha, conversão permitida, trajetória incompatível.
- **Validade do semáforo por cena** (`light_validity`) e o teste empírico: se o semáforo controla a
  fila, quase todos os cruzamentos acontecem no verde.
- Evidência: 3 quadros por infração, com pessoas e placas desfocadas.

### 4.6 Pedestres (Fase 6)

- O AIC21 não tem faixa de pedestres marcada (levantamento em todos os vídeos). Por isso, são
  contadas pessoas cruzando linhas: o eixo da rua (travessias, inclusive fora da faixa) e uma linha
  atravessando a calçada (fluxo). Há também um modo de zona para cenas com faixa.
- YOLO11s a 1280 px, porque as pessoas ao fundo têm ~20 px.

### 4.7 Protocolo de avaliação (Fase 7)

- Gabarito por eventos com instante, marcado com uma ferramenta própria (`src/annotate.py`). O
  anotador não vê as previsões, e só contam os trechos assistidos.
- Semáforo: primeiro quadro de cada cor. Cruzamentos: quadro em que a frente passa a linha
  (definição legal). Pedestres: quadro em que os pés cruzam a linha.
- Métricas (`src/evaluate.py`): acurácia por quadro e erro no instante das trocas (semáforo);
  precisão, revocação e F1 com pareamento um a um por tempo (±1 s) e erro de contagem por classe
  e sentido (contagem); precisão e revocação de infrações e tabela cor do gabarito × veredito
  (infração).

## 5 Resultados

### 5.1 Velocidade e escolha do modelo

[Tabela de `docs/fase2_modelos.md`, incluindo o yolo11m.] O FPS fica entre 33 e 41 com qualquer
modelo e resolução, porque o gargalo é a decodificação e a CPU. Todas as combinações ficam acima dos
8–10 FPS dos vídeos, então a escolha é feita pela qualidade: o yolo11s fragmenta menos os IDs.

### 5.2 Contagem de veículos

[RESULTADO FASE 7: precisão, revocação, F1, erro por classe e sentido, por cena e por modelo.]

### 5.3 Estado do semáforo

- Plausibilidade (já medida): amarelo de 3,8–4,4 s na cam_7 (padrão 3–5 s), ciclo de ~130 s
  constante, 180 recortes conferidos visualmente, todos corretos. "Desconhecido" por quadro de 9,6%
  (cam_7 de dia, lâmpada de ~4 px) e ≤ 1,6% nas demais.
- [RESULTADO FASE 7: acurácia por quadro (raw, suavizado, retrospectivo) e erro médio nas trocas.]

### 5.4 Validade do semáforo e avanço de sinal

- Tabela do teste empírico (`docs/fase5_infracao.md`): cam_11 a cam_13 com cruzamentos dezenas de
  segundos depois do vermelho, o que mostra que aquele semáforo não controla a fila.
- cam_7 nas 3 condições (40 min, 724 cruzamentos): nenhuma infração pelo instante da frente. A
  versão pela traseira apontava 1 infração (0,2 s depois do vermelho); a frente desse carro cruzou
  0,45 s antes do vermelho, no amarelo, mesmo com ±20% de erro na escala. Antecedência mediana da
  frente: 0,64–0,93 s para carros. 5 vereditos mudaram (tabela em `docs/fase5_infracao.md`).
- Erros encontrados e corrigidos durante o desenvolvimento (tabela).
- [RESULTADO FASE 7: detecção de cruzamentos e tabela cor × veredito.]

### 5.5 Pedestres

cam_19 (5 min): 3 travessias da rua fora da faixa e 8 passagens na calçada, conferidas nos recortes.
[RESULTADO FASE 7 para pedestres.]

## 6 Discussão e limitações

- **Poucas infrações reais:** a revocação de infrações não é mensurável com estes dados (não há
  dados sintéticos; decisão de escopo). A avaliação cobre a detecção dos cruzamentos e a cor
  atribuída a cada um, que são as duas partes de que o veredito depende.
- **Semáforo visível ≠ semáforo da fila:** o achado mais transferível do trabalho para a prática.
- **Frente × traseira:** um erro sistemático de ~0,5–0,9 s (mediana) que favorece falsas infrações
  na troca amarelo → vermelho. A escala px/m é provisória (±20%).
- Faixa de conversão com grupo focal próprio, rastreamento perdido logo após a linha, quadros
  irregulares (±1 quadro) e 8–10 FPS.
- Pedestres: poucos eventos, grupos e oclusão por carros estacionados; sem faixa marcada.

## 7 Conclusão

[Retomar os objetivos específicos, um a um, com o resultado de cada um. Trabalhos futuros: vídeo
com o semáforo da própria faixa e mais tráfego; calibração métrica completa (homografia);
classificador de semáforo treinado; noite.]

## Referências

[Formatar em ABNT. Lista inicial a conferir:]
- BRASIL. Lei nº 9.503, de 23 de setembro de 1997. Código de Trânsito Brasileiro.
- BRASIL. Lei nº 13.709, de 14 de agosto de 2018. Lei Geral de Proteção de Dados Pessoais.
- NAPHADE, M. et al. The 5th AI City Challenge. CVPR Workshops, 2021.
- REDMON, J. et al. You Only Look Once: Unified, Real-Time Object Detection. CVPR, 2016.
- ZHANG, Y. et al. ByteTrack: Multi-Object Tracking by Associating Every Detection Box. ECCV, 2022.
- JOCHER, G. et al. Ultralytics YOLO (software), 2024.

## Apêndice A — Reprodução

Comandos de cada fase: ver `README.md`.
