# Fase 5 — Detecção de avanço de sinal vermelho

Código: `src/violation.py`. Usa `src/detect_track.py` (YOLO + ByteTrack) e
`src/traffic_light.py` (Fase 4).

```powershell
python -m src.violation --config configs/aic21_cam_7.yaml                 # 30 min em ~6 min
python -m src.violation --config configs/aic21_cam_7.yaml --video         # + vídeo anotado
python -m src.violation --config configs/aic21_cam_7.yaml --red-grace 0.3 # tolerância de 0,3 s
```

A cena precisa de `stop_line` (tecla `r` no `scene_editor`), `traffic_light_roi`,
`stop_line_direction` (`in` = veículos sobem na imagem para uma linha marcada da esquerda para a
direita; `out` = descem) e `light_validity` (ver a seção de validade abaixo). `exempt_zone`
(opcional) é um polígono onde a conversão no vermelho é permitida.

`light_validity: nao_controla` faz o programa se recusar a rodar (só com `--force`, para estudar o
erro); `substituto` e `nao_verificado` rodam com aviso, e o valor fica gravado em `run_info.yaml`.

## Regra

1. **Cruzamento:** o ponto inferior central da caixa (contato com o chão) passa da linha de
   retenção no sentido de entrada, dentro do segmento marcado. O instante é **interpolado** entre
   os dois quadros (a 8 FPS, um carro a 50 km/h anda ~1,7 m entre quadros).
2. **Balanço da câmera:** o deslocamento medido no semáforo a cada quadro é aplicado também à
   linha de retenção (a câmera da cam_7 se desloca até ~20 px).
3. **Filtros contra cruzamentos falsos** (cada um motivado por um erro observado):
   - *pulo de ID*: deslocamento entre quadros seguidos maior que a altura da caixa;
   - *movimento transversal*: movimento com mais de 60° em relação à normal da linha (carro da rua
     transversal passando pela ponta da linha);
   - *trajetória incompatível*: ~1 s depois do cruzamento, o deslocamento desde a linha precisa
     apontar para dentro do cruzamento (até 60° da normal). Pega a troca de ID depois de uma
     lacuna: o ByteTrack guarda o ID de um veículo perdido por ~30 quadros e pode entregá-lo a
     outro veículo.
4. **Cor no instante do cruzamento, em retrospectiva:** a votação temporal da Fase 4 confirma a
   troca de cor com atraso (2 a 4 quadros, mais quando os quadros da transição ficam
   "desconhecidos"). Como cada cruzamento só é decidido 2 s depois, a troca é reposicionada: a nova
   cor começa no quadro seguinte ao último quadro bruto que ainda mostrava a cor antiga.
5. **Confirmação de avanço:** o veículo precisa avançar além da linha pelo menos 0,3 × a altura da
   caixa (mín. 6 px). Quem para em cima da linha fica em espera (até 60 s); se avançar depois, vale
   a cor do momento em que avançou; se não, o veredito é `parado_na_linha`.
6. **Veredito:** vermelho → `infracao`; amarelo → `amarelo`; verde → `verde`; sem leitura →
   `indeterminado`; entrou na `exempt_zone` no vermelho → `conversao_permitida`; vermelho antes da
   tolerância `--red-grace` → `vermelho_tolerancia`; reprovado no item 3 → `trajetoria_incompativel`.
7. **Evidência:** uma imagem por infração com 3 quadros (0,5 s antes, logo após o cruzamento e no
   momento da decisão), linha e semáforo coloridos pela cor em retrospectiva, pessoas e faixa da
   placa desfocadas.

## Validade do semáforo para cada linha

Só faz sentido medir avanço de sinal se o semáforo lido **controla** os veículos daquela linha.
Um semáforo que mostra a frente para a câmera controla quem se afasta dela. Na cam_7 é isso que
acontece (a fila anda em direção ao semáforo). Nas cam_10 a cam_13, a fila vem em direção à câmera
e o semáforo visível fica do outro lado do cruzamento, controlando o sentido oposto: ele só serve
como substituto se os dois sentidos abrirem juntos.

**Teste empírico:** se o semáforo controla a linha, quase todos os cruzamentos acontecem com ele
verde (ou amarelo). Grupos de veículos cruzando dezenas de segundos depois do início do vermelho
indicam que o semáforo não é o daquela linha.

| Cena | Condição | Duração | Cruzamentos | verde | amarelo | vermelho | `light_validity` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cam_7 | dia | 30 min | 576 | 567 | 8 | **1** | `controla` |
| cam_7_dawn | amanhecer | 5 min | 63 | 63 | 0 | 0 | `controla` |
| cam_7_rain | chuva | 5 min | 49 | 48 | 1 | 0 | `controla` |
| cam_10 | dia | 3,5 min | 64 | 63 | 1 | 0 | `substituto` |
| cam_11 | dia | 3,5 min | 54 | 41 | 0 | 13 | `nao_controla` (fila cruza 33–48 s após o vermelho) |
| cam_12 | dia | 3,3 min | 11 | 7 | 0 | 4 | `nao_controla` |
| cam_13 | dia | 3,3 min | 4 | 3 | 0 | 1 | `nao_controla` |

Cruzamentos descartados pelos filtros não entram na tabela; `parado_na_linha` também não.

**Decisão de escopo:** a avaliação de avanço de sinal usa **só as cenas `controla`** (cam_7 em
dia, amanhecer e chuva), em que o semáforo lido é comprovadamente o que a fila obedece.

- **cam_10 (`substituto`):** o resultado (63 de 64 no verde) sugere que os dois sentidos abrem
  juntos, mas 3,5 min (1–2 ciclos) não garantem isso. Muitos cruzamentos abrem um sentido alguns
  segundos antes do outro, e é justamente nas trocas de cor que as infrações acontecem. Fica como
  discussão, não como métrica.
- **cam_11 a cam_13 (`nao_controla`):** entram no texto como demonstração de que o teste empírico
  detecta um semáforo errado. Todas as quatro câmeras continuam válidas para a contagem de
  veículos, que não depende do semáforo.

## Resultado

Nas 3 cenas `controla` (40 min de vídeo, 688 cruzamentos), o sistema registrou **1 infração**:

- **cam_7, 865,4 s, ID 31734:** o carro cruza a linha **0,2 s** depois do início do vermelho,
  conferido quadro a quadro (no quadro 6921 os três grupos focais estão amarelos, em 6922 já estão
  vermelhos, e o carro cruza entre 6922 e 6923). É um caso limite: com tolerância de 0,3 s (comum em
  sistemas de fiscalização) não seria autuado. Na primeira versão, que compensava o atraso da
  votação por um número fixo de quadros, esse caso saiu como "amarelo"; foi o que motivou o
  reposicionamento em retrospectiva.

Desempenho: **41–46 FPS** no pipeline completo (RTX 5060), acima dos 8–10 FPS dos vídeos.

### Revisão: traseira x frente do veículo (Fase 7)

O ponto inferior central da caixa só é o contato com o chão **mais próximo da câmera**. Na cam_7
os veículos se afastam da câmera, então esse ponto é a **traseira**: o sistema registra o
cruzamento quando o veículo inteiro já passou da linha. Pela definição do CTB, o que vale é a
**frente** passando a linha. O atraso é L/v, o comprimento do veículo dividido pela velocidade: um
carro de 4,7 m a 50 km/h leva ~0,34 s.

Medição do caso de 865,4 s (caixa do YOLO quadro a quadro):

| Quadro | 6913 | 6916 | 6919 | 6921 | 6922 | 6923 |
| --- | --- | --- | --- | --- | --- | --- |
| Semáforo | amarelo | amarelo | amarelo | amarelo | **vermelho** | vermelho |
| Base da caixa (traseira), y | 632 | 613 | 599 | 591 | 582 | 577 (cruza) |
| Topo da caixa (teto), y | 594 | 576 (na linha) | 562 | 555 | 547 | 541 |

- A traseira cruza 0,2 s depois do início do vermelho (resultado acima).
- O **topo da caixa não serve** como "frente": a câmera está quase na horizontal, e a caixa
  (46x36 px, proporção da traseira de um carro) é praticamente só a face traseira. O teto aparece
  vários metros adiante do para-choque dianteiro. Usar o topo anteciparia demais o cruzamento.
- Primeira estimativa, feita à mão (velocidade constante, escala de 4 a 6 px/m): a frente cruzaria
  ~0,3 s antes da traseira, ~0,1 s antes do vermelho. O método implementado, a partir da trajetória
  (abaixo), dá 0,59 s de antecedência: a frente cruzou **0,45 s antes do vermelho**. A diferença vem
  dos quadros repetidos do vídeo, que reduzem o deslocamento médio por quadro. A precisão do vídeo
  (8 FPS, quadros repetidos) é de ±1 quadro (±0,125 s).

**Consequência:** o cruzamento pela traseira atrasa o instante em ~0,3–0,5 s para carros (mais
para ônibus e caminhões) quando o veículo se afasta da câmera, e isso favorece falsas infrações na
troca amarelo → vermelho. Correção proposta: estimar o instante da frente como
`t_traseira − L_classe / v`, com a escala px/m perto da linha calibrada pelo gabarito da Fase 7.
O anotador marca a frente, e a mediana de `t_traseira − t_frente` por classe dá a correção.

### Implementação: instante da frente (padrão `--anchor frente`)

A frente toca a linha quando a traseira está a um comprimento de veículo antes dela. O
`violation.py` guarda, para cada ID, a distância da traseira até a linha nos últimos 6 s e procura
o instante em que ela valia `−L_classe × stop_line_px_per_m`, interpolando entre quadros. Assim não
é preciso supor velocidade constante, e o método vale também para quem arranca. Comprimentos:
carro 4,7 m, moto 2,2 m, ônibus 12 m, caminhão 6,5 m (no COCO, "caminhão" inclui muitas picapes).
Casos especiais (coluna `front_method`):

- `extrapolado`: a trajetória começa depois desse ponto; usa a velocidade do último segundo;
- `lento`: a frente chegou mais de 3 s antes da traseira (parou ou andou devagar sobre a linha);
  usa a traseira;
- `sem_historico` / `sem_escala`: usa a traseira.

`crossings.csv` passa a ter `rear_time_s`, `front_time_s`, `lead_s` e `light_state_rear`, para
comparar as duas definições. `--anchor traseira` reproduz o comportamento anterior.

**Escala (`stop_line_px_per_m: 5.5`, provisória).** Calibrada pelo espaçamento entre traseiras de
carros parados na mesma faixa, na fila do vermelho: ~40–45 px perto da linha (y ≈ 607–630) para
~7 m (comprimento + folga típicos de fila parada), o que dá ~6 px/m, ou ~5,5 px/m na linha. A
incerteza é de cerca de ±20%. A mesma escala vale para o amanhecer e a chuva: a câmera foi
reposicionada entre as gravações, mas o tamanho dos carros e das marcas no chão perto da linha é
parecido (±10%). Recalibrar com o gabarito da Fase 7.

### Resultado: frente x traseira (cam_7, 40 min, 724 cruzamentos)

| Cena | Cruzamentos | Método `historico` | Antecedência da frente, mediana (carro / caminhão) | Vereditos que mudaram |
| --- | --- | --- | --- | --- |
| cam_7 (dia) | 603 | 517 | 0,93 s / 1,30 s | 5 |
| cam_7_dawn | 67 | 62 | 0,64 s / 1,00 s | 0 |
| cam_7_rain | 54 | 46 | 0,87 s / 1,59 s | 0 |

Os 5 vereditos que mudaram, todos na cam_7 de dia:

| ID | Traseira | Frente | Antecedência | Cor (traseira → frente) |
| --- | --- | --- | --- | --- |
| 31734 | 865,39 s | 864,80 s | 0,59 s | **vermelho → amarelo** (a frente cruzou 0,45 s antes do vermelho) |
| 22826 | 601,25 s | 600,57 s | 0,68 s | amarelo → verde |
| 41269 | 1121,34 s | 1120,63 s | 0,71 s | amarelo → verde |
| 46651 | 1251,79 s | 1250,84 s | 0,95 s | amarelo → verde |
| 68323 | 1771,93 s | 1769,00 s | 2,92 s | amarelo → verde (arrancada lenta) |

**A única infração da versão anterior deixa de existir.** A frente do carro 31734 cruzou a linha
0,45 s antes do vermelho. Com ±20% na escala, a antecedência fica entre 0,47 e 0,74 s, e a frente
cruza de 0,33 a 0,6 s antes do vermelho: no amarelo em qualquer caso. Nas 3 cenas válidas
(40 min), o sistema não registra **nenhuma infração**. Isso é coerente com a observação visual: não
há avanço de sinal claro nesses vídeos.

### Erros encontrados e corrigidos durante o desenvolvimento

| Erro | Causa | Correção |
| --- | --- | --- |
| Infração limite saía como "amarelo" | atraso da votação variável (quadros de transição "desconhecidos") | troca de cor reposicionada em retrospectiva |
| 6 falsas infrações por carros parados na fila (cam_7) | caixa oscila sobre a linha | exigir avanço mínimo além da linha |
| Falsa infração no amanhecer (jipe parado) | ByteTrack entregou o ID do jipe perdido a um carro da transversal 29 quadros depois | filtro de trajetória depois da linha |
| Descartar toda lacuna > 0,5 s (tentativa) | eliminava 9% dos cruzamentos reais na cam_7 (veículos encobertos por galhos e poste) | substituído pelo filtro de trajetória |
| Painéis "antes" e "cruzamento" iguais na evidência | memória de quadros curta quando a decisão atrasa | memória de `decide_after + 4 s` |

## Limitações conhecidas

- **Poucos positivos.** Com 1 infração (limite) em 40 min de cenas válidas, precisão e revocação
  não podem ser estimadas com segurança, e só há um cruzamento físico (cam_7). **Decisão
  (2026-11-15): sem CARLA.** A avaliação usa só vídeo real: o gabarito manual da Fase 7 mede a
  detecção dos cruzamentos e a cor atribuída a cada um, com atenção às trocas amarelo → vermelho,
  onde os erros de instante viram erros de veredito. A revocação de infrações reais fica declarada
  como não mensurável com estes dados (trabalho futuro: outra fonte de vídeo, com o semáforo da
  própria faixa visível e mais tráfego).
- **Faixa de conversão à esquerda.** A linha da cam_7 cobre também a faixa da esquerda, cujos
  veículos obedecem ao grupo focal de conversão (à esquerda, x≈795), não ao da via principal. Os 9
  cruzamentos `trajetoria_incompativel` da cam_7 são todos nessa faixa (viram à esquerda logo após a
  linha). O correto é marcar uma linha e um semáforo próprios para a faixa de conversão.
- **Veículos que viram logo após a linha** podem ser reprovados pelo filtro de trajetória; numa
  cena com conversão à direita livre, use a `exempt_zone`.
- **Rastreamento perdido logo após a linha:** 18 `parado_na_linha` na cam_7 (a maioria no verde);
  se acontecer no vermelho, uma infração pode passar despercebida.
- **Quadros irregulares no vídeo:** em alguns trechos a posição do veículo "volta" entre quadros
  seguidos (gravação a 8 FPS com quadros fora de ordem/repetidos), o que limita a precisão temporal
  a cerca de ±1 quadro (±0,125 s).
- **Linhas propostas:** as linhas da cam_7_rain (transposta da cam_7) e das cam_10 a cam_13 foram
  propostas a partir dos quadros e precisam ser conferidas no `scene_editor`; a da cam_7_rain é a
  que importa para a avaliação.
- O desfoque de placa é heurístico (faixa central no terço inferior da caixa); os vídeos do AIC21
  já vêm com placas e rostos borrados.
