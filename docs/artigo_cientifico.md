# SISTEMA DE VISÃO COMPUTACIONAL PARA MONITORAMENTO DE TRÁFEGO E MOBILIDADE URBANA INTELIGENTE

**COMPUTER VISION SYSTEM FOR TRAFFIC MONITORING AND SMART URBAN MOBILITY**

Nathan Kevyn Oliveira Ribeiro¹
Kauê Pablo Soares de Souza¹
Cibele Andréa de Godoy Fonseca²

¹ Graduandos do curso de Ciência da Computação, Centro Universitário Adventista de São Paulo (UNASP).
² Profa. Dra., orientadora, Centro Universitário Adventista de São Paulo (UNASP).

> **Notas de edição (remover antes da entrega).** Texto em conformidade com a ABNT NBR 6022:2018
> (artigo), NBR 6023:2018 (referências) e NBR 10520:2023 (citações, sistema autor-data). Os
> campos entre colchetes devem ser preenchidos. Números marcados como *resultado preliminar*
> vêm dos experimentos já executados (`docs/fase*.md`); os da avaliação contra gabarito manual
> (Fase 7) ainda não existem e aparecem apenas como resultados esperados. Conversão:
> `pandoc docs/artigo_cientifico.md -o artigo.docx`.

---

## RESUMO

O crescimento da frota e a recorrência de sinistros em interseções semaforizadas tornam a
fiscalização e o monitoramento contínuo do tráfego um problema central da mobilidade urbana. Este
artigo apresenta o projeto e a avaliação preliminar de um sistema de visão computacional que, a
partir do vídeo de câmeras fixas já instaladas em vias urbanas, conta veículos por classe e
sentido, conta pedestres e detecta veículos que avançam o sinal vermelho. A detecção de objetos
utiliza a rede neural convolucional YOLO11 e o rastreamento multiobjeto, o algoritmo ByteTrack. O
estado do semáforo é classificado sem treinamento, por regras de cor no espaço HSV, com
compensação do balanço da câmera e suavização temporal restrita ao ciclo verde, amarelo e
vermelho. A infração é decidida pelo instante, interpolado entre quadros, em que a frente do
veículo cruza a linha de retenção, conforme a definição do Código de Trânsito Brasileiro. Os
experimentos utilizam o conjunto público AI City Challenge 2021 (Track 1), em condições de dia,
amanhecer e chuva. Os resultados preliminares indicam processamento acima da taxa de quadros dos
vídeos em uma GPU de uso doméstico, classificação de semáforo coerente com os tempos de ciclo
observados e, sobretudo, que em quatro das cinco câmeras com semáforo visível o grupo focal que
aparece na imagem não controla a fila analisada, o que invalida a leitura de infrações nessas
cenas. Esse achado é proposto como uma etapa obrigatória de validação em sistemas semelhantes.
Contra um gabarito anotado manualmente (30 min, 42 trocas de cor), o classificador de semáforo
atingiu 99,85% de acurácia por quadro e encontrou todas as trocas, com erro médio de 0,06 s no
instante de cada uma. Na contagem de pedestres, o total coincidiu com o gabarito nas duas linhas
avaliadas (11 eventos), com F1 entre 0,67 e 1,00 conforme a tolerância temporal adotada. Na
detecção de cruzamentos da linha de retenção, o F1 foi de 0,55 a 0,64, limitado também por erros
do próprio gabarito, e o instante marcado pelo anotador coincidiu com o da traseira do veículo, e
não com o da frente estimada, o que indica que a frente de um veículo visto por trás não é
observável nem pelo anotador humano.

**Palavras-chave:** visão computacional; aprendizado profundo; YOLO; rastreamento de objetos;
avanço de sinal vermelho; sistemas inteligentes de transporte.

## ABSTRACT

Fleet growth and the recurrence of crashes at signalized intersections make continuous traffic
monitoring and enforcement a central problem of urban mobility. This paper presents the design
and preliminary evaluation of a computer vision system that, from the video of fixed cameras
already installed on urban roads, counts vehicles by class and direction, counts pedestrians, and
detects vehicles that run the red light. Object detection uses the YOLO11 convolutional neural
network and multi-object tracking uses ByteTrack. The traffic light state is classified without
training, by color rules in the HSV space, with camera sway compensation and temporal smoothing
constrained to the green, yellow and red cycle. A violation is decided by the instant,
interpolated between frames, at which the front of the vehicle crosses the stop line, following
the definition of the Brazilian Traffic Code. Experiments use the public AI City Challenge 2021
(Track 1) dataset under daytime, dawn and rain conditions. Preliminary results show processing
above the video frame rate on a consumer GPU, traffic light classification consistent with the
observed cycle timings and, above all, that in four of the five cameras with a visible traffic
light, the signal head shown in the image does not control the observed queue, which invalidates
violation readings in those scenes. This finding is proposed as a mandatory validation step for
similar systems. Against a manually annotated ground truth (30 min, 42 color changes), the traffic
light classifier reached 99.85% per-frame accuracy and found every change, with a mean timing
error of 0.06 s. For pedestrian counting, the total matched the ground truth on both evaluated
lines (11 events), with F1 between 0.67 and 1.00 depending on the temporal tolerance. For stop
line crossing detection, F1 ranged from 0.55 to 0.64, also limited by errors in the ground truth
itself, and the instant marked by the annotator matched the rear of the vehicle rather than the
estimated front, which indicates that the front of a vehicle seen from behind is not observable
even by a human annotator.

**Keywords:** computer vision; deep learning; YOLO; object tracking; red light running;
intelligent transportation systems.

---

## 1 INTRODUÇÃO

A Organização Mundial da Saúde estima em cerca de 1,19 milhão o número anual de mortes no
trânsito no mundo, com aumento das mortes de pedestres entre 2010 e 2021 (WORLD HEALTH
ORGANIZATION, 2023). Nas cidades, uma parcela relevante desses sinistros ocorre em interseções,
onde fluxos conflitantes são separados apenas pelo semáforo e pelo respeito do condutor a ele. No
Brasil, avançar o sinal vermelho é infração gravíssima (BRASIL, 1997, art. 208), e a literatura
internacional associa a fiscalização por câmeras à redução de violações e de colisões transversais
com vítimas (RETTING; FERGUSON; HAKKERT, 2003).

Ao mesmo tempo, a gestão da mobilidade depende de dados de demanda — volumes de veículos por
classe, sentido e horário, fluxo de pedestres — que sustentam o planejamento semafórico e as
políticas previstas na Política Nacional de Mobilidade Urbana (BRASIL, 2012). Tradicionalmente,
esses dados vêm de contagens manuais ou de sensores instalados no pavimento, como laços
indutivos, de custo elevado de implantação e manutenção. Os Sistemas Inteligentes de Transporte
(ITS, do inglês *Intelligent Transportation Systems*) propõem integrar sensoriamento, comunicação
e processamento para tornar essa gestão contínua e orientada por dados (DIMITRAKOPOULOS;
DEMESTICHAS, 2010; ZHANG *et al.*, 2011), e o vídeo é uma das fontes de dados mais ricas
disponíveis para isso (BUCH; VELASTIN; ORWELL, 2011).

O avanço recente do aprendizado profundo (LECUN; BENGIO; HINTON, 2015), em especial das redes
neurais convolucionais aplicadas à detecção de objetos em tempo real (REDMON *et al.*, 2016;
TERVEN; CÓRDOVA-ESPARZA; ROMERO-GONZÁLEZ, 2023) e dos rastreadores multiobjeto por detecção
(BEWLEY *et al.*, 2016; ZHANG *et al.*, 2022), tornou viável extrair essas informações do vídeo de
câmeras de monitoramento comuns, em uma única GPU de consumo. Este artigo descreve o projeto de um
sistema com esse objetivo, detalha as decisões metodológicas tomadas a partir de erros observados
durante o desenvolvimento e apresenta os resultados preliminares obtidos em um conjunto de dados
público.

O texto está organizado da seguinte forma: as seções 2 a 4 apresentam o problema, a justificativa
e os objetivos; a seção 5 reúne a fundamentação teórica; a seção 6, os trabalhos correlatos; a
seção 7, os materiais e métodos; a seção 8, a proposta do sistema; as seções 9 e 10, os resultados
esperados e preliminares e sua discussão; e a seção 11, as considerações finais.

## 2 PROBLEMA DE PESQUISA

A fiscalização eletrônica de avanço de sinal e a contagem de tráfego costumam exigir equipamentos
dedicados, enquanto muitas cidades já dispõem de câmeras de monitoramento cujo vídeo é apenas
observado por operadores. O problema que orienta este trabalho é: **é possível, apenas com o vídeo
de uma câmera fixa de monitoramento e com modelos pré-treinados de uso livre, contar veículos e
pedestres e identificar avanços de sinal vermelho com confiabilidade suficiente para apoiar a
gestão do trânsito?**

Desse problema derivam questões mais específicas, que estruturam a avaliação:

a) a detecção e o rastreamento, executados em hardware acessível, acompanham a taxa de quadros
   do vídeo e mantêm a identidade dos veículos o suficiente para contá-los uma única vez?

b) o estado do semáforo pode ser lido da imagem, sem treinamento específico, de forma robusta a
   variações de iluminação, chuva e balanço da câmera?

c) o instante em que o veículo cruza a linha de retenção pode ser estimado com precisão compatível
   com a definição legal da infração, considerando a baixa taxa de quadros (8 a 10 FPS) típica de
   câmeras de monitoramento?

d) o semáforo visível na imagem é, de fato, o que controla os veículos observados?

A última questão não estava no escopo inicial do projeto; ela surgiu dos próprios experimentos e
se mostrou determinante para a validade dos resultados (seção 9.4).

## 3 JUSTIFICATIVA

A justificativa do trabalho apoia-se em três argumentos.

O primeiro é de **relevância social**: sinistros em interseções semaforizadas envolvem colisões
transversais e atropelamentos, de alta gravidade, e a fiscalização tem efeito documentado sobre o
comportamento dos condutores (RETTING; FERGUSON; HAKKERT, 2003; WORLD HEALTH ORGANIZATION, 2023).

O segundo é de **viabilidade econômica**: reaproveitar câmeras existentes e modelos de código
aberto reduz o custo de obtenção de dados de tráfego em relação a sensores no pavimento, em linha
com a concepção de cidades inteligentes como sistemas que usam dados digitais para melhorar a
gestão urbana (BATTY *et al.*, 2012; ALBINO; BERARDI; DANGELICO, 2015).

O terceiro é de **contribuição científica**: trabalhos de detecção de infração por vídeo costumam
relatar métricas de detecção de veículos ou de cor do semáforo, mas raramente discutem se o
semáforo lido controla a faixa analisada, se o cruzamento é medido pela frente ou pela traseira do
veículo, ou como a baixa taxa de quadros afeta o veredito perto da troca amarelo-vermelho. Este
trabalho trata explicitamente desses pontos, e o tratamento de dados pessoais nas evidências
segue o princípio da minimização da Lei Geral de Proteção de Dados Pessoais (BRASIL, 2018).

## 4 OBJETIVOS

### 4.1 Objetivo geral

Desenvolver e avaliar um sistema de visão computacional que, a partir do vídeo de uma câmera fixa
em via urbana, conte veículos e pedestres e detecte avanços de sinal vermelho, gerando evidências
com dados pessoais desfocados.

### 4.2 Objetivos específicos

a) detectar e rastrear veículos e pessoas com YOLO11 e ByteTrack, comparando tamanhos de modelo e
   resoluções de entrada quanto à velocidade e à estabilidade das identidades;

b) contar veículos por classe, sentido e janela de tempo por meio de linhas virtuais;

c) classificar o estado do semáforo quadro a quadro, sem treinamento, de forma robusta a balanço
   da câmera, amanhecer e chuva;

d) detectar avanços de sinal vermelho pelo instante em que a frente do veículo cruza a linha de
   retenção, e verificar empiricamente se o semáforo lido controla a fila analisada;

e) contar pedestres que atravessam a via e que circulam pela calçada;

f) avaliar cada módulo contra um gabarito de eventos anotado manualmente.

## 5 FUNDAMENTAÇÃO TEÓRICA

### 5.1 Visão computacional e aprendizado profundo

A visão computacional busca extrair descrições do mundo a partir de imagens e vídeos (SZELISKI,
2022). Até a década de 2010, a análise de tráfego por vídeo baseava-se principalmente em
subtração de fundo, fluxo óptico e classificadores sobre características projetadas manualmente,
com dificuldades conhecidas em cenas urbanas: oclusões frequentes, ângulos de câmera baixos,
variedade de usuários da via e condições climáticas adversas (BUCH; VELASTIN; ORWELL, 2011).

O aprendizado profundo substituiu as características projetadas por representações aprendidas em
múltiplas camadas a partir dos dados (LECUN; BENGIO; HINTON, 2015; GOODFELLOW; BENGIO;
COURVILLE, 2016). Nas redes neurais convolucionais (CNN), cada camada aplica filtros
compartilhados em toda a imagem, seguidos de não linearidades e de redução espacial, o que explora
a localidade e a invariância à translação das imagens (LECUN *et al.*, 1998). O desempenho da
AlexNet na classificação do ImageNet (KRIZHEVSKY; SUTSKEVER; HINTON, 2017) consolidou as CNN como
base da visão computacional moderna, inclusive da detecção de objetos.

### 5.2 Detecção de objetos e a família YOLO

Detectar objetos é localizar, por caixas delimitadoras, e classificar cada instância de interesse
na imagem (ZOU *et al.*, 2023). Os detectores de dois estágios, como a R-CNN (GIRSHICK *et al.*,
2014) e a Faster R-CNN (REN *et al.*, 2017), primeiro propõem regiões candidatas e depois as
classificam. Os detectores de estágio único, introduzidos pelo YOLO (*You Only Look Once*),
tratam a detecção como uma regressão direta, em uma única passada da rede, das caixas e das
probabilidades de classe sobre uma grade da imagem, o que permite operação em tempo real (REDMON
*et al.*, 2016).

A família evoluiu com âncoras e treinamento multiescala no YOLOv2 (REDMON; FARHADI, 2017),
predição em três escalas no YOLOv3 (REDMON; FARHADI, 2018), um conjunto de técnicas de
treinamento e arquitetura no YOLOv4 (BOCHKOVSKIY; WANG; LIAO, 2020) e reparametrização e
agregação de camadas no YOLOv7 (WANG; BOCHKOVSKIY; LIAO, 2023), entre outras variantes
revisadas por Terven, Córdova-Esparza e Romero-González (2023). O YOLO11, utilizado neste
trabalho, é mantido pela Ultralytics (JOCHER; QIU, 2024) e introduz blocos C3k2 e um módulo de
atenção espacial (C2PSA), sendo oferecido em tamanhos de *nano* a *extra-large* (KHANAM; HUSSAIN,
2024). Os pesos públicos são treinados no COCO (LIN *et al.*, 2014), que inclui as classes
carro, motocicleta, ônibus, caminhão e pessoa, suficientes para este trabalho sem novo
treinamento.

O desempenho de detectores é medido pela sobreposição entre a caixa prevista $B_p$ e a caixa de
referência $B_g$, a *Intersection over Union*:

$$\mathrm{IoU} = \frac{|B_p \cap B_g|}{|B_p \cup B_g|}$$

e por precisão $P = VP/(VP+FP)$, revocação $R = VP/(VP+FN)$ e $F_1 = 2PR/(P+R)$, em que $VP$, $FP$
e $FN$ são verdadeiros positivos, falsos positivos e falsos negativos. Neste trabalho, as mesmas
métricas são aplicadas a **eventos no tempo** (cruzamentos de linha), e não a caixas (seção 7.5).

### 5.3 Rastreamento multiobjeto

Contar e julgar infrações exige saber que a caixa de um quadro e a do quadro seguinte pertencem ao
mesmo veículo. No paradigma de rastreamento por detecção, cada trajetória mantém um estado
previsto por um filtro de Kalman (KALMAN, 1960), e as detecções de cada quadro são associadas às
trajetórias resolvendo um problema de atribuição pelo método húngaro (KUHN, 1955). O SORT
aplicou esse esquema com custo baseado em IoU e alcançou alta velocidade (BEWLEY *et al.*, 2016);
o DeepSORT acrescentou um descritor de aparência para reduzir trocas de identidade em oclusões
(WOJKE; BEWLEY; PAULUS, 2017). O ByteTrack, empregado aqui, associa primeiro as detecções de alta
confiança e, em seguida, também as de baixa confiança, que costumam corresponder a objetos
parcialmente encobertos, recuperando trajetórias que seriam interrompidas (ZHANG *et al.*, 2022).
Trajetórias perdidas são mantidas por um número de quadros (*buffer*) antes de descartadas, o que
permite reencontrar o objeto, mas também pode entregar um identificador a outro veículo — efeito
que motivou filtros específicos neste trabalho (seção 8.4). Métricas como MOTA e MOTP avaliam o
rastreamento em si (BERNARDIN; STIEFELHAGEN, 2008); uma revisão ampla da área é apresentada por
Luo *et al.* (2021).

### 5.4 Reconhecimento de semáforos e o espaço de cor HSV

O reconhecimento de semáforos é estudado sobretudo na perspectiva do veículo autônomo, com
detectores treinados em bases como a LISA (JENSEN *et al.*, 2016) e a Bosch Small Traffic Lights
(BEHRENDT; NOVAK; BOTROS, 2017). Jensen *et al.* (2016) destacam as dificuldades do problema:
lâmpadas pequenas, saturação do sensor, reflexos e variação de iluminação. Em câmeras fixas, a
posição do semáforo é conhecida, e a tarefa reduz-se a decidir qual lâmpada está acesa em uma
região de interesse. O espaço HSV separa matiz (H), saturação (S) e valor ou brilho (V), o que
permite decidir pela cor de forma menos sensível a variações de iluminação do que no espaço RGB
(GONZALEZ; WOODS, 2018).

### 5.5 Sistemas inteligentes de transporte e mobilidade urbana

Os ITS integram sensores, comunicação e processamento para monitorar e controlar redes de
transporte (DIMITRAKOPOULOS; DEMESTICHAS, 2010). Zhang *et al.* (2011) descrevem a transição para
ITS orientados por dados, em que vídeo, sensores e dispositivos móveis alimentam modelos de
previsão, controle e fiscalização. No contexto de cidades inteligentes, a infraestrutura de
sensoriamento urbano é vista como instrumento de planejamento e de qualidade de vida (BATTY
*et al.*, 2012; ALBINO; BERARDI; DANGELICO, 2015). No Brasil, a Política Nacional de Mobilidade
Urbana estabelece a segurança nos deslocamentos e a eficiência da circulação como objetivos
(BRASIL, 2012). O monitoramento de interseções por visão computacional, revisado por Datondji *et
al.* (2016), é uma das aplicações mais diretas desse ecossistema.

### 5.6 Fiscalização de avanço de sinal e privacidade

Pelo Código de Trânsito Brasileiro, constitui infração gravíssima avançar o sinal vermelho do
semáforo (BRASIL, 1997, art. 208). Operacionalmente, considera-se que o veículo avança o sinal
quando sua **frente** ultrapassa a linha de retenção com a indicação vermelha acesa; sistemas de
fiscalização costumam aplicar uma tolerância de alguns décimos de segundo após o início do
vermelho. As imagens produzidas como evidência contêm dados pessoais (rostos, placas), cujo
tratamento deve observar a finalidade e a minimização previstas na LGPD (BRASIL, 2018).

## 6 TRABALHOS CORRELATOS

Song *et al.* (2019) propõem um sistema de detecção e contagem de veículos em rodovias que
segmenta a pista em regiões distante e próxima antes de aplicar o YOLOv3, melhorando a detecção de
veículos pequenos, e obtêm trajetórias e sentido por correspondência de pontos ORB; os autores
também publicam uma base com 57.290 instâncias anotadas. Dai *et al.* (2019) apresentam um
arcabouço de contagem de veículos em vídeo que combina detecção por rede profunda, rastreamento e
regras de contagem. Mandal e Adu-Gyamfi (2020) comparam combinações de detectores e rastreadores
para contagem de carros e caminhões em tempo real, mostrando que a escolha do rastreador afeta a
contagem tanto quanto a do detector.

No Brasil, Neitzke, Sano e Tampelini (2023) aplicam YOLOv5 e DeepSORT à quantificação do fluxo de
veículos nas proximidades de um polo educacional em Toledo (PR), e Krakauer e Pedrini (2018)
propõem a contagem volumétrica classificada por visão computacional como insumo para otimizar
semáforos. Em escala de *benchmark*, a Track 1 do 5º AI City Challenge avaliou a contagem de
veículos por movimento em interseções, com critérios de eficácia e eficiência computacional
(NAPHADE *et al.*, 2021) — é o conjunto de vídeos utilizado neste trabalho.

Quanto a infrações, Ong, Connie e Ong Goh (2024) usam YOLO para detectar veículos e semáforos e
DeepSORT para rastreá-los em imagens de CCTV; a linha de retenção é localizada automaticamente por
detecção de bordas e transformada de Hough, e a infração é decidida pela distância entre o veículo
rastreado e a linha durante o vermelho. Na linha do reconhecimento de semáforos, Jensen *et al.*
(2016) e Behrendt, Novak e Botros (2017) propõem bases e métodos treinados, voltados a veículos
autônomos.

O Quadro 1 sintetiza a comparação.

**Quadro 1 — Comparação entre trabalhos correlatos e a proposta**

| Trabalho | Tarefa | Detecção / rastreamento | Semáforo | Pontos não tratados |
| --- | --- | --- | --- | --- |
| Song *et al.* (2019) | contagem (rodovia) | YOLOv3 / ORB | — | sem semáforo; cena de rodovia |
| Dai *et al.* (2019) | contagem | CNN / rastreador | — | sem semáforo |
| Mandal e Adu-Gyamfi (2020) | contagem | vários / vários | — | sem semáforo |
| Neitzke, Sano e Tampelini (2023) | contagem | YOLOv5 / DeepSORT | — | sem semáforo |
| Naphade *et al.* (2021) | contagem por movimento (*benchmark*) | livre | — | não avalia infrações |
| Ong, Connie e Ong Goh (2024) | avanço de sinal, ultrapassagem | YOLO / DeepSORT | lido da imagem (YOLO) | validade do semáforo; frente × traseira |
| **Proposta** | contagem, pedestres, avanço de sinal | YOLO11 / ByteTrack | lido da imagem (HSV) | validade testada; instante da frente |

Fonte: elaborado pelos autores (2026).

A lacuna explorada por este trabalho está na última coluna: entre os trabalhos analisados que leem
o semáforo da imagem, não se encontrou verificação de que o grupo focal lido controla a fila
analisada, nem tratamento explícito da diferença entre o instante em que a traseira e a frente do
veículo cruzam a linha.

## 7 MATERIAIS E MÉTODOS

### 7.1 Caracterização da pesquisa

Trata-se de pesquisa aplicada, de abordagem quantitativa e caráter experimental, desenvolvida de
forma iterativa: cada módulo foi implementado, executado sobre os vídeos, inspecionado
visualmente e corrigido a partir dos erros observados, que são documentados como parte dos
resultados. A avaliação final compara as saídas do sistema com um gabarito anotado manualmente.

### 7.2 Ambiente computacional

O sistema foi desenvolvido em Python 3.13, sobre Windows 11, com PyTorch e CUDA 12.8, em um
computador com GPU NVIDIA RTX 5060 de 8 GB. As principais bibliotecas são Ultralytics (YOLO11 e
integração do ByteTrack), supervision (linhas e zonas de contagem), OpenCV (leitura de vídeo,
processamento de imagem e interface de anotação) e pandas (tabelas de saída). O código é
organizado em módulos independentes por etapa e cada cena é descrita por um arquivo de
configuração YAML, gerado por um editor de cenas próprio, com linhas de contagem, linha de
retenção, região do semáforo e linhas de pedestres.

### 7.3 Base de dados

Utiliza-se a Track 1 (*Vehicle Counting*) do AI City Challenge 2021 (NAPHADE *et al.*, 2021):
31 vídeos, cerca de 5 horas, de 20 câmeras fixas, com 8 a 15 quadros por segundo e rostos e placas
já desfocados na origem. A licença permite uso acadêmico não comercial e proíbe a redistribuição
dos vídeos. Um levantamento visual de todas as câmeras identificou cinco com semáforo legível
(cam_7, cam_10, cam_11, cam_12 e cam_13) e nenhuma com faixa de pedestres marcada; não há vídeos
noturnos, e o amanhecer é usado como condição de baixa luminosidade. A Tabela 1 resume as cenas.

**Tabela 1 — Cenas utilizadas por módulo**

| Módulo | Cenas | Condições | Duração |
| --- | --- | --- | --- |
| Estado do semáforo | cam_7, cam_7_dawn, cam_7_rain, cam_10 a cam_13 | dia, amanhecer, chuva | 3,3 a 30 min por cena |
| Avanço de sinal | cam_7, cam_7_dawn, cam_7_rain | dia, amanhecer, chuva | 40 min |
| Pedestres | cam_19 | dia | 5 min |
| Contagem de veículos | cenas com linha de contagem marcada | dia, amanhecer, chuva | 5 min por cena (amostra) |

Fonte: elaborado pelos autores a partir de Naphade *et al.* (2021).

### 7.4 Procedimentos experimentais

Os experimentos seguem as etapas: (i) comparação de velocidade e de fragmentação de
identidades entre YOLO11 *n*, *s* e *m* em 640, 960 e 1280 pixels, com 1.200 quadros por vídeo e
descarte dos 50 iniciais; (ii) execução do classificador de semáforo em todas as cenas com
semáforo, com conferência visual de recortes e verificação de plausibilidade da duração do
amarelo (3 a 5 s) e da regularidade do ciclo; (iii) execução do detector de infrações nas cenas
válidas, com registro de todos os cruzamentos e seus vereditos; (iv) contagem de pedestres; e
(v) avaliação contra gabarito (seção 7.5).

### 7.5 Protocolo de avaliação

Como as perguntas do trabalho são sobre **eventos no tempo** — quando o semáforo troca de cor,
quando um veículo cruza uma linha e se isso ocorreu no vermelho —, o gabarito é uma lista de
eventos com o quadro em que ocorreram, e não caixas anotadas quadro a quadro. A anotação é feita
em ferramenta própria, que reproduz o vídeo e registra o evento por tecla, com avanço quadro a
quadro nos instantes duvidosos. O anotador não vê as previsões do sistema, e apenas os trechos
efetivamente assistidos entram na avaliação. Os critérios são:

a) **semáforo**: primeiro quadro em que a nova cor aparece acesa; métricas de acurácia por
   quadro, precisão e revocação por cor e erro, em segundos, no instante de cada troca;

b) **cruzamentos da linha de retenção**: quadro em que a frente do veículo passa a linha, com a
   classe do veículo; pareamento um a um pelo instante, com tolerância de ±1 s; precisão,
   revocação e $F_1$ dos cruzamentos e tabela "cor do gabarito × veredito do sistema";

c) **contagem de veículos e pedestres**: quadro em que o ponto de contato com o chão cruza a linha
   virtual, com sentido e classe; precisão, revocação, $F_1$, erro de contagem por classe e sentido
   e erro absoluto médio por janela de tempo.

O código de avaliação foi validado com um gabarito sintético derivado das próprias saídas, com
perturbações conhecidas (trocas adiantadas em 2 quadros, ruído de 0,3 s, 5% de eventos removidos e
3 inseridos); as métricas recuperaram as perturbações aplicadas. Até a redação deste artigo, o
gabarito do semáforo da cam_7 foi concluído (30 min integrais, 42 trocas) e seus resultados estão
na seção 9.3. O gabarito de pedestres da cam_19 foi concluído para as duas linhas (5 min integrais
cada; seção 9.5). O gabarito dos cruzamentos da linha de retenção foi
concluído na terceira tentativa: a primeira registrou a chegada dos veículos à fila do vermelho, e
não a passagem pela linha, e a segunda cobriu apenas 13% do vídeo; ambas foram descartadas. A
versão final cobre as fases amarela e vermelha de todos os 14 ciclos, com 3 s de margem antes de
cada amarelo; como os veículos permanecem parados durante o vermelho, o anotador avançou para o
ciclo seguinte quando a fila se formava, e os trechos assistidos somam 5,9 min (20% do vídeo).
Apenas esses trechos entram nas métricas (seção 9.4).

## 8 PROPOSTA DE DESENVOLVIMENTO DO SISTEMA

### 8.1 Arquitetura geral

O sistema processa o vídeo quadro a quadro em um *pipeline* com cinco módulos (Figura 1). O
módulo de detecção e rastreamento é compartilhado; os demais consomem as trajetórias e, no caso
das infrações, o estado do semáforo.

**Figura 1 — Arquitetura do sistema**

```
 vídeo da câmera fixa ──► YOLO11 (detecção) ──► ByteTrack (trajetórias com ID)
        │                                              │
        │                     ┌────────────────────────┼─────────────────────┐
        ▼                     ▼                        ▼                     ▼
 classificador de      contagem de veículos     contagem de pedestres   detector de avanço
 semáforo (HSV) ─────► por linha virtual        (linhas / zonas)        de sinal vermelho
        │                                                                    │
        └──────────────── estado do semáforo por quadro ─────────────────────┘
                                                                             ▼
                               saídas: CSV de eventos, contagens por janela, evidências desfocadas
```

Fonte: elaborada pelos autores (2026).

### 8.2 Detecção, rastreamento e contagem de veículos

O YOLO11 detecta as classes carro, motocicleta, ônibus e caminhão do COCO, com confiança mínima
de 0,25, e o ByteTrack atribui um identificador a cada veículo. A contagem usa uma linha virtual:
cada identificador é contado uma única vez, quando o ponto inferior central da caixa — o contato
com o chão — cruza a linha, e o sentido é dado pelo lado de origem. Como a classe prevista pode
alternar ao longo da trajetória (por exemplo, carro e caminhão para picapes), a classe final é a
mais votada entre os quadros da trajetória. As contagens são agregadas por classe, sentido e
janela de tempo configurável.

### 8.3 Classificação do estado do semáforo

O módulo opera sobre uma região de interesse marcada no editor de cenas e não usa aprendizado. A
versão final resulta de quatro tentativas descartadas, cada uma motivada por uma falha observada
(Quadro 2), e tem três etapas:

a) **localização com compensação de balanço**: a câmera chega a oscilar cerca de 20 px com o
   vento. O grupo focal é reencontrado a cada quadro por correlação de *template* em uma janela
   de ±6 px em torno da posição anterior, limitada a ±25 px da referência. A correlação é feita
   sobre um mapa de silhueta $M = \max(255 - V,\, S)$, em que a caixa do semáforo tem a mesma forma
   qualquer que seja a lâmpada acesa; abaixo de 0,6 de correlação, mantém-se a última posição;

b) **decisão pela cor**: pixels com $S \ge 90$ e $V \ge 110$ são separados por faixa de matiz
   (vermelho, amarelo, verde-azulado). A cor com mais pixels vence se ocupar ao menos 0,5% da
   região e tiver o dobro da segunda colocada; caso contrário, o quadro é "desconhecido";

c) **suavização com regra do ciclo**: o candidato é a maioria dos últimos 5 quadros e só substitui
   o estado atual se respeitar a sequência verde → amarelo → vermelho → verde; uma troca fora do
   ciclo só é aceita se persistir por 3 s.

**Quadro 2 — Tentativas descartadas na classificação do semáforo**

| Tentativa | Falha observada |
| --- | --- |
| Lâmpada mais brilhante (canal V) por posição | caminhões claros ao fundo e caixa desalinhada geravam falsos vermelhos |
| Idem, com busca local, exigência de cor e regra do ciclo | 99% de quadros "desconhecidos" ao amanhecer: o sol baixo faz as lentes apagadas brilharem |
| Brilho colorido (S × V) por posição | vermelho pálido de dia; halo do vermelho na chuva invade a posição do amarelo |
| *Template* em tons de cinza | na chuva, o halo entra no *template* e a localização desliza ~10 px |

Fonte: elaborado pelos autores (2026).

A lição comum às tentativas é decidir pela **cor**, e não pela posição ou pelo brilho, e localizar
pela **silhueta**, e não pela imagem em tons de cinza.

### 8.4 Detecção de avanço de sinal vermelho

A regra de infração combina as trajetórias, a linha de retenção e o estado do semáforo:

a) **cruzamento**: o ponto de contato com o chão passa a linha de retenção no sentido de entrada.
   O instante é interpolado linearmente entre os dois quadros — a 8 FPS, um carro a 50 km/h
   percorre cerca de 1,7 m entre quadros. O deslocamento medido no semáforo é aplicado também à
   linha, que acompanha o balanço da câmera;

b) **filtros contra cruzamentos falsos**, cada um motivado por um erro observado: salto de
   identificador (deslocamento entre quadros maior que a altura da caixa); movimento transversal
   (mais de 60° em relação à normal da linha); e trajetória incompatível (cerca de 1 s após o
   cruzamento, o deslocamento deve apontar para dentro da interseção), que captura o caso em que o
   rastreador entrega o identificador de um veículo perdido a outro;

c) **instante da frente do veículo**: quando o veículo se afasta da câmera, o ponto inferior da
   caixa corresponde à sua **traseira**. A frente toca a linha quando a traseira está a um
   comprimento de veículo antes dela. Sendo $d(t)$ a distância assinada, em pixels, da traseira
   até a linha, $L_c$ o comprimento típico da classe $c$ (carro 4,7 m; moto 2,2 m; ônibus 12 m;
   caminhão 6,5 m) e $s$ a escala em px/m perto da linha, o instante da frente $t_f$ é obtido no
   histórico da trajetória, por interpolação, como a solução de

   $$d(t_f) = -L_c \cdot s,$$

   sem supor velocidade constante. A escala ($s \approx 5{,}5$ px/m, incerteza de ±20%) foi
   calibrada pelo espaçamento entre veículos parados na fila do vermelho;

d) **cor no instante do cruzamento, em retrospectiva**: a votação temporal confirma as trocas com
   atraso de 2 a 4 quadros. Como cada cruzamento é decidido 2 s depois, a troca é reposicionada no
   quadro seguinte ao último quadro bruto que ainda mostrava a cor anterior;

e) **confirmação de avanço**: o veículo precisa avançar além da linha ao menos 0,3 vez a altura da
   caixa; quem para sobre a linha fica em espera e, se avançar depois, vale a cor desse momento;

f) **veredito**: infração (vermelho), amarelo, verde, indeterminado, conversão permitida (zona de
   conversão livre), vermelho dentro da tolerância configurável, parado na linha ou trajetória
   incompatível;

g) **validade do semáforo**: cada cena declara se o semáforo marcado controla a fila
   (`controla`), se é um substituto plausível (`substituto`) ou se não a controla
   (`nao_controla`); cenas `nao_controla` são recusadas. O critério é físico — um semáforo cuja
   face aparece para a câmera controla os veículos que se afastam dela — e é confirmado por um
   teste empírico: se o semáforo controla a fila, quase todos os cruzamentos ocorrem no verde ou no
   amarelo; grupos de veículos cruzando dezenas de segundos após o início do vermelho indicam o
   semáforo errado.

### 8.5 Contagem de pedestres

Como nenhuma câmera do conjunto possui faixa de pedestres marcada, o módulo conta pessoas que
cruzam linhas nomeadas: uma no eixo da rua, que registra travessias inclusive fora da faixa, e
outra atravessando a calçada, que mede o fluxo. Um modo por zona, para cenas com faixa, mede o
tempo de permanência e o percurso ao longo do eixo da faixa, classificando como travessia quem
percorre ao menos 40% de seu comprimento. Usa-se o YOLO11s a 1280 px, pois as pessoas ao fundo
têm cerca de 20 px de altura.

### 8.6 Evidências e privacidade

Para cada infração é gerada uma imagem com três quadros (0,5 s antes, logo após o cruzamento e no
momento da decisão), com a linha e o semáforo coloridos pela cor reconstruída. Pessoas e a região
provável da placa são desfocadas antes da gravação, em atenção à minimização de dados pessoais
(BRASIL, 2018), embora os vídeos do conjunto já venham desfocados na origem.

## 9 RESULTADOS ESPERADOS E RESULTADOS PRELIMINARES

Esta seção separa o que já foi medido (resultados preliminares, sem gabarito) do que se espera
obter com a avaliação contra o gabarito manual.

### 9.1 Desempenho computacional

Com os modelos *n* e *s*, o *pipeline* de detecção e rastreamento operou entre 33 e 41 FPS em
todas as resoluções testadas, o que indica que o gargalo é a decodificação do vídeo e o
processamento em CPU, e não a rede. Apenas o modelo *m* a 1280 px passou a depender da GPU (19 a
21 FPS). Todas as combinações superam os 8 a 10 FPS dos vídeos. Aumentar a resolução de 640 para
1280 px dobrou as detecções por quadro na cam_7 (de 21 para 41), por incluir veículos pequenos ao
fundo, próximos à linha de retenção. O modelo *n* fragmentou mais as identidades (1.777
identificadores contra 1.349 do *s* na cam_7 a 1280 px), o que levaria a contagens duplicadas;
por isso o YOLO11s foi adotado nos módulos de infração e de pedestres. O *pipeline* completo de
infração atingiu 41 a 46 FPS (resultado preliminar).

### 9.2 Contagem de veículos

**Resultado esperado.** Espera-se medir precisão, revocação e $F_1$ dos eventos de cruzamento e o
erro de contagem por classe e sentido em amostras de 5 minutos de cenas com condições diferentes,
comparando YOLO11n e YOLO11s. A hipótese de trabalho, sustentada pela menor fragmentação de
identidades (seção 9.1), é que o modelo *s* apresente menor taxa de contagem duplicada, e que o
erro se concentre na distinção entre carro e caminhão, dada a ambiguidade das picapes no COCO.

### 9.3 Estado do semáforo

A Tabela 2 apresenta os resultados preliminares. Foram conferidos visualmente 180 recortes (60 em
cada uma das cenas cam_7, cam_7_dawn e cam_13), todos corretos. As durações de amarelo ficaram
dentro do intervalo esperado e o ciclo da cam_7 manteve-se em cerca de 130 s em todas as voltas.
Após a suavização, nenhuma cena fica sem estado atribuído, exceto antes da primeira leitura.

**Tabela 2 — Resultados preliminares do classificador de semáforo**

| Cena | Condição | Duração | Quadros "desconhecidos" (antes da suavização) | Amarelo (s) |
| --- | --- | --- | --- | --- |
| cam_7 | dia | 30 min | 9,6% | 3,8–4,4 |
| cam_7_dawn | amanhecer | 5 min | 0,1% | 4,0–4,1 |
| cam_7_rain | chuva | 5 min | 0,2% | 3,9–4,0 |
| cam_10 | dia | 3,5 min | 0,5% | 3,8–3,9 |
| cam_11 | dia | 3,5 min | 1,6% | 3,5–3,9 |
| cam_12 | dia | 3,3 min | 0,1% | 3,0–4,0 |
| cam_13 | dia | 3,3 min | 0,1% | 2,5–3,1 |

Fonte: elaborada pelos autores (2026).

O maior índice de quadros desconhecidos na cam_7 diurna decorre do tamanho da lâmpada (cerca de
4 px).

**Avaliação contra gabarito.** Os 30 minutos da cam_7 (14.400 quadros) foram anotados
integralmente, com 42 trocas de cor. A Tabela 3 apresenta as métricas das três versões do
classificador: bruta (quadro a quadro), suavizada (em tempo real) e retrospectiva (com as trocas
reposicionadas, como o detector de infrações a consulta).

**Tabela 3 — Classificador de semáforo contra o gabarito manual (cam_7, 30 min)**

| Métrica | Bruta | Suavizada | Retrospectiva |
| --- | --- | --- | --- |
| Acurácia por quadro | 90,38% | 99,34% | 99,85% |
| Acurácia nos quadros com cor decidida | 100,00% | 99,34% | 99,85% |
| Verde: precisão / revocação / F1 | 1,000 / 0,949 / 0,974 | 0,997 / 0,997 / 0,997 | 0,999 / 0,999 / 0,999 |
| Amarelo: precisão / revocação / F1 | 1,000 / 0,838 / 0,912 | 0,918 / 0,932 / 0,925 | 0,978 / 0,989 / 0,983 |
| Vermelho: precisão / revocação / F1 | 1,000 / 0,823 / 0,903 | 0,994 / 0,992 / 0,993 | 0,999 / 0,999 / 0,999 |
| Trocas encontradas / falsas | 42 de 42 / 0 | 42 de 42 / 0 | 42 de 42 / 0 |
| Erro médio no instante da troca | +0,051 s | +0,283 s | −0,062 s |
| Maior erro absoluto | 0,375 s | 0,500 s | 0,375 s |

Fonte: elaborada pelos autores (2026). Erro positivo: o sistema troca de cor depois do gabarito.

A versão bruta nunca atribuiu a cor errada: quando decide, acerta em 100% dos quadros, e a perda
de revocação vem inteiramente dos 9,6% de quadros "desconhecidos". A suavização preenche esses
quadros, mas atrasa cada troca em 0,283 s em média (cerca de 2 quadros a 8 FPS), o que derruba a
precisão do amarelo para 0,918, pois os primeiros quadros do vermelho ainda são rotulados como
amarelo. A reconstrução retrospectiva reduz esse atraso para 0,062 s em valor absoluto, abaixo da
duração de um quadro (0,125 s), e eleva o F1 do amarelo para 0,983. Esse resultado confirma a
hipótese levantada na validação sintética e justifica o uso da versão retrospectiva no detector de
infrações, cuja decisão depende justamente do instante da troca amarelo → vermelho.

### 9.4 Validade do semáforo e avanço de sinal vermelho

O teste empírico de validade (Tabela 4) mostrou que, nas cam_11 a cam_13, grupos de veículos
cruzam a linha de retenção dezenas de segundos após o início do vermelho — na cam_11, entre 33 e
48 s. Isso só é possível se o semáforo visível não controla aquela fila: nessas câmeras a fila vem
em direção à câmera, e o semáforo que aparece de frente é o do sentido oposto. Na cam_10, o
resultado (63 de 64 cruzamentos no verde) sugere que os dois sentidos abrem juntos, mas 3,5 min
não bastam para garanti-lo. Apenas a cam_7, nas três condições, tem o semáforo comprovadamente
correto.

**Tabela 4 — Teste empírico de validade do semáforo (cor no cruzamento da linha de retenção)**

| Cena | Cruzamentos | Verde | Amarelo | Vermelho | Classificação |
| --- | --- | --- | --- | --- | --- |
| cam_7 (dia) | 576 | 567 | 8 | 1 | controla |
| cam_7_dawn | 63 | 63 | 0 | 0 | controla |
| cam_7_rain | 49 | 48 | 1 | 0 | controla |
| cam_10 | 64 | 63 | 1 | 0 | substituto |
| cam_11 | 54 | 41 | 0 | 13 | não controla |
| cam_12 | 11 | 7 | 0 | 4 | não controla |
| cam_13 | 4 | 3 | 0 | 1 | não controla |

Fonte: elaborada pelos autores (2026). Contagem feita com o instante da traseira do veículo.

Nas cenas válidas, a versão inicial, que media o cruzamento pela traseira, registrou uma única
infração: um carro que cruzou 0,2 s após o início do vermelho, confirmado quadro a quadro. Com o
instante da frente (seção 8.4, item c), a frente desse carro cruzou 0,45 s **antes** do vermelho,
ainda no amarelo; mesmo com ±20% de erro na escala, o cruzamento permanece entre 0,33 e 0,6 s
antes do vermelho. Nos 40 minutos de cenas válidas (724 cruzamentos na versão pela frente), o
sistema não registra nenhuma infração, o que é coerente com a inspeção visual. A antecedência
mediana da frente em relação à traseira foi de 0,64 a 0,93 s para carros e de 1,00 a 1,59 s para
caminhões, conforme a cena (Tabela 5).

**Tabela 5 — Instante da frente × instante da traseira (cam_7)**

| Cena | Cruzamentos | Antecedência mediana da frente (carro / caminhão) | Vereditos alterados |
| --- | --- | --- | --- |
| cam_7 (dia) | 603 | 0,93 s / 1,30 s | 5 |
| cam_7_dawn | 67 | 0,64 s / 1,00 s | 0 |
| cam_7_rain | 54 | 0,87 s / 1,59 s | 0 |

Fonte: elaborada pelos autores (2026).

Os cinco vereditos alterados migraram para uma cor "anterior" no ciclo (vermelho → amarelo em um
caso, amarelo → verde em quatro), como esperado para um erro sistemático de atraso. O
desenvolvimento também registrou e corrigiu erros que teriam produzido falsas infrações: seis
veículos parados cuja caixa oscilava sobre a linha (corrigido pela confirmação de avanço) e uma
troca de identificador após 29 quadros de oclusão (corrigida pelo filtro de trajetória).

**Avaliação contra gabarito.** Nos 5,9 minutos anotados da cam_7, que concentram as transições
verde → amarelo → vermelho, o gabarito registra 14 cruzamentos. A Tabela 6 compara as duas
referências de instante: a frente estimada (versão adotada) e a traseira (ponto inferior da caixa).

**Tabela 6 — Detecção de cruzamentos da linha de retenção contra o gabarito (cam_7, 5,9 min)**

| Referência | Gabarito | Sistema | Pareados | Precisão | Revocação | F1 | Diferença mediana de instante | Cor concordante |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Frente estimada | 14 | 15 | 8 | 0,533 | 0,571 | 0,552 | −0,49 s | 6 de 8 |
| Traseira | 14 | 14 | 9 | 0,643 | 0,643 | 0,643 | 0,00 s | 7 de 9 |

Fonte: elaborada pelos autores (2026). Pareamento por instante com tolerância de ±1 s. Diferença
negativa: o sistema registra o cruzamento antes do gabarito. Cor concordante: veredito do sistema
igual à cor do semáforo no gabarito, entre os cruzamentos pareados.

Dois resultados se destacam. O primeiro é que os valores de precisão e revocação são um limite
inferior do desempenho real: a inspeção quadro a quadro das divergências mostrou que o gabarito
deixou de marcar pelo menos dois cruzamentos reais detectados pelo sistema (aos 77,7 s e aos
132,5 s) e incluiu um evento, aos 871,1 s, que corresponde a um veículo que chegou à linha e parou
no vermelho, sem atravessá-la. Com tão poucos eventos, cada erro de anotação altera as métricas em
cerca de 7 pontos percentuais.

O segundo é que **o instante marcado pelo anotador coincide com o da traseira** (diferença mediana
de 0,00 s) e não com o da frente estimada, que antecede o gabarito em 0,49 s. Embora o protocolo
peça a marcação da frente, a câmera vê os veículos por trás, e o para-choque dianteiro não é
visível: na prática, o anotador marca a passagem da parte visível do veículo, como faz o ponto
inferior da caixa. O gabarito, portanto, não permite validar a correção pela frente, que permanece
sustentada pela definição legal e pela geometria, mas não por medição direta.

Quanto às infrações, o gabarito contém dois cruzamentos com o semáforo vermelho. Um é o evento
aos 871,1 s, já descrito como erro de anotação. O outro é o caso limite discutido no início desta
seção: o anotador o marcou no primeiro quadro do vermelho (0,00 s), a versão pela traseira o
registra 0,14 s após o início do vermelho do gabarito e a versão pela frente, 0,45 s antes, no
amarelo. Com a
tolerância de 0,3 s usual em fiscalização eletrônica, nenhuma das três leituras configura infração.
Assim, os vídeos válidos não contêm infrações inequívocas, e a revocação de infrações reais
continua não mensurável com estes dados.

### 9.5 Pedestres

Na cam_19 (5 min, dia), o módulo registrou 3 travessias da rua fora de faixa e 8 passagens pela
calçada, conferidas nos recortes de vídeo, incluindo um grupo de quatro pessoas contado
corretamente, a 20 FPS com entrada de 1280 px.

**Avaliação contra gabarito.** As duas linhas foram anotadas nos 5 minutos integrais do vídeo,
marcando cada pessoa individualmente, inclusive dentro de grupos. A Tabela 7 apresenta os
resultados com a tolerância padrão do protocolo (±1 s) e com uma tolerância de ±2,5 s.

**Tabela 7 — Contagem de pedestres contra o gabarito manual (cam_19, 5 min)**

| Linha | Gabarito | Sistema | Erro de contagem | P / R / F1 (±1 s) | P / R / F1 (±2,5 s) |
| --- | --- | --- | --- | --- | --- |
| Calçada (fluxo) | 8 | 8 | 0% | 0,875 / 0,875 / 0,875 | 1,00 / 1,00 / 1,00 |
| Eixo da rua (travessias) | 3 | 3 | 0% | 0,667 / 0,667 / 0,667 | 1,00 / 1,00 / 1,00 |

Fonte: elaborada pelos autores (2026). P: precisão; R: revocação.

O total do sistema coincidiu com o gabarito nas duas linhas, inclusive no grupo de quatro pessoas
que atravessou a linha da calçada em cerca de 3 s. As duas divergências na tolerância de ±1 s
ocorreram em grupos (aos 146 s na calçada e aos 152 s na rua), em que a diferença de instante
entre o sistema e o gabarito foi de 1,8 s e 2,2 s. Essas diferenças são atribuídas à imprecisão da
marcação manual do instante quando várias pessoas cruzam a linha com poucos décimos de segundo de
intervalo, e a inspeção quadro a quadro dos recortes confirmou que os eventos detectados pelo
sistema correspondem a pessoas reais cruzando a linha. Por esse motivo, os resultados com ±2,5 s são apresentados como complementares, e não como
substitutos dos obtidos com o protocolo padrão. Com apenas 11 eventos, os números demonstram a
viabilidade do método, mas não permitem estimativas estatisticamente robustas.

## 10 DISCUSSÃO

O resultado mais relevante para a prática é que **o semáforo visível na imagem nem sempre é o
semáforo da fila analisada**. Em quatro das cinco câmeras com semáforo legível, ele controla o
sentido oposto. Um sistema que lesse esse semáforo e aplicasse a regra de infração apontaria, por
exemplo, 13 "infrações" em 3,5 minutos na cam_11, todas falsas. A verificação é simples — pela
geometria da cena e pelo teste empírico da Tabela 4 — e propõe-se que ela seja etapa obrigatória
na implantação de sistemas de fiscalização por vídeo que leiam o semáforo da imagem, algo que
não foi encontrado nos trabalhos correlatos analisados (seção 6).

O segundo ponto é a **diferença entre frente e traseira** do veículo. O ponto inferior central da
caixa, usado de forma quase universal como referência de cruzamento em contagem, corresponde à
traseira quando o veículo se afasta da câmera. Para a contagem, isso é irrelevante; para a
infração, introduz um atraso sistemático de 0,6 a 0,9 s (mediana, carros) que favorece falsas
infrações exatamente na troca amarelo → vermelho. O gabarito manual, porém, mostrou que essa
diferença não pode ser resolvida por anotação visual quando o veículo é visto por trás: o anotador
marcou o instante da traseira (diferença mediana de 0,00 s), e não o da frente. A correção pela
frente decorre da definição legal e da geometria, mas sua validação exige outra fonte de
referência — uma câmera lateral, um laço indutivo ou a calibração métrica completa (homografia do
plano da via), que também substituiria a escala px/m provisória. Na prática, a tolerância de 0,3 s
após o início do vermelho absorve o único caso limite observado em qualquer das duas referências.

O terceiro ponto é a **resolução temporal**. A 8 FPS, a precisão do instante de cruzamento é de
cerca de ±1 quadro (±0,125 s), agravada por quadros repetidos no vídeo. A interpolação entre
quadros e a reconstrução retrospectiva da troca de cor mitigam, mas não eliminam, essa limitação;
uma tolerância após o início do vermelho, prática comum em fiscalização, é coerente com essa
incerteza.

Quanto ao semáforo, a abordagem por regras em HSV mostrou-se adequada a câmeras fixas, sem
necessidade de dados rotulados, e robusta ao amanhecer e à chuva depois que a decisão passou a ser
feita pela cor e a localização, pela silhueta. A avaliação contra o gabarito da cam_7 (99,85% de
acurácia, 42 de 42 trocas, erro médio de 0,06 s) mostra que, mesmo com a lâmpada ocupando cerca de
4 px, a leitura da cor não é a principal fonte de incerteza do veredito: o erro remanescente no
instante da troca é menor que o intervalo entre quadros e que a diferença entre frente e traseira
do veículo. A classe mais difícil é o amarelo, por ser a fase mais curta e estar sempre entre duas
transições; é também a que mais se beneficia da reconstrução retrospectiva. Ressalva-se que esse
resultado se refere a uma única câmera, em condição diurna; o amanhecer e a chuva foram avaliados
apenas por inspeção visual e plausibilidade. Abordagens treinadas (JENSEN *et al.*, 2016;
BEHRENDT; NOVAK; BOTROS, 2017) permanecem como alternativa para cenas em que a lâmpada ocupa
poucos pixels, como na cam_7 diurna.

A construção do gabarito também trouxe uma lição metodológica. Anotar eventos no tempo é muito
mais barato que anotar caixas, mas é sensível à definição operacional do evento e à precisão da
marcação: a primeira versão do gabarito de cruzamentos registrou a chegada dos veículos à fila do
vermelho, e não a passagem pela linha, e, nos pedestres, o instante de pessoas em grupo divergiu
em até 2,2 s. Recomenda-se anotar os trechos de transição quadro a quadro e conferir uma amostra
das marcações antes de calcular as métricas.

Entre as limitações, destacam-se: a ausência de infrações reais nos vídeos válidos, que impede
estimar a revocação de infrações; a existência de uma única interseção com semáforo válido; a
faixa de conversão à esquerda da cam_7, que obedece a um grupo focal próprio e deveria ter linha e
semáforo separados; a ausência de vídeos noturnos e de faixas de pedestres; e o baixo número de
eventos de pedestres. Essas limitações são de dados, não de método, e orientam os trabalhos
futuros.

## 11 CONSIDERAÇÕES FINAIS

Este trabalho apresentou um sistema de visão computacional para monitoramento de tráfego que
conta veículos e pedestres e detecta avanços de sinal vermelho a partir do vídeo de câmeras fixas,
usando modelos de código aberto e hardware acessível. Em relação aos objetivos específicos: a
detecção e o rastreamento com YOLO11 e ByteTrack operam acima da taxa de quadros dos vídeos, e o
modelo *s* oferece melhor estabilidade de identidades (objetivo a); a contagem por linha virtual,
com votação de classe ao longo da trajetória, está implementada e tem protocolo de avaliação
definido (objetivo b); o classificador de semáforo por regras em HSV funcionou em dia, amanhecer e
chuva e, contra o gabarito manual da cam_7, atingiu 99,85% de acurácia por quadro, com todas as 42
trocas encontradas e erro médio de 0,06 s (objetivo c); o detector de infrações incorpora o
instante da frente do veículo e a verificação de validade do semáforo e, contra o gabarito,
detectou os cruzamentos da linha de retenção com F1 de 0,55 (frente) a 0,64 (traseira), valores
limitados também por erros do próprio gabarito, sem registrar nenhuma infração inequívoca
(objetivo d); a contagem de
pedestres por linhas, em cena sem faixa marcada, acertou o total nas duas linhas avaliadas, com F1
de 0,67 a 0,88 na tolerância padrão de ±1 s e de 1,00 na tolerância de ±2,5 s (objetivo e); e o
protocolo de avaliação por eventos, com ferramenta de anotação e métricas, está implementado, foi
validado com dados sintéticos e produziu as avaliações do semáforo, dos cruzamentos e dos
pedestres (objetivo f).

As principais contribuições são metodológicas: a verificação, geométrica e empírica, de que o
semáforo lido controla a fila analisada; a explicitação da diferença entre frente e traseira do
veículo e de seu efeito sobre o veredito, com a constatação de que ela não é verificável por
anotação visual quando o veículo é visto por trás; e o registro sistemático dos erros encontrados e de suas correções,
útil a quem desenvolver sistemas semelhantes.

Como trabalhos futuros, propõem-se: ampliar o gabarito de cruzamentos, com revisão independente
por um segundo anotador, e incluir a avaliação da contagem de veículos; obter vídeos em
que o semáforo da própria faixa esteja visível e com maior volume de tráfego, para estimar a
revocação de infrações; substituir a escala px/m por calibração métrica por homografia; comparar o
classificador por regras com um classificador treinado; incluir cenas noturnas e com faixa de
pedestres; e avaliar a operação contínua, em tempo real, integrada a uma central de monitoramento
de trânsito.

## REFERÊNCIAS

ALBINO, V.; BERARDI, U.; DANGELICO, R. M. Smart cities: definitions, dimensions, performance, and
initiatives. **Journal of Urban Technology**, [*s. l.*], v. 22, n. 1, p. 3-21, 2015. DOI:
10.1080/10630732.2014.942092.

BATTY, M. *et al.* Smart cities of the future. **The European Physical Journal Special Topics**,
[*s. l.*], v. 214, n. 1, p. 481-518, 2012. DOI: 10.1140/epjst/e2012-01703-3.

BEHRENDT, K.; NOVAK, L.; BOTROS, R. A deep learning approach to traffic lights: detection,
tracking, and classification. *In*: IEEE INTERNATIONAL CONFERENCE ON ROBOTICS AND AUTOMATION
(ICRA), 2017, Singapore. **Proceedings** [...]. Piscataway: IEEE, 2017. p. 1370-1377. DOI:
10.1109/ICRA.2017.7989163.

BERNARDIN, K.; STIEFELHAGEN, R. Evaluating multiple object tracking performance: the CLEAR MOT
metrics. **EURASIP Journal on Image and Video Processing**, [*s. l.*], v. 2008, art. 246309,
2008. DOI: 10.1155/2008/246309.

BEWLEY, A. *et al.* Simple online and realtime tracking. *In*: IEEE INTERNATIONAL CONFERENCE ON
IMAGE PROCESSING (ICIP), 2016, Phoenix. **Proceedings** [...]. Piscataway: IEEE, 2016.
p. 3464-3468. DOI: 10.1109/ICIP.2016.7533003.

BOCHKOVSKIY, A.; WANG, C.-Y.; LIAO, H.-Y. M. **YOLOv4**: optimal speed and accuracy of object
detection. [*S. l.*]: arXiv, 2020. DOI: 10.48550/arXiv.2004.10934.

BRASIL. Lei nº 9.503, de 23 de setembro de 1997. Institui o Código de Trânsito Brasileiro.
**Diário Oficial da União**: seção 1, Brasília, DF, 24 set. 1997.

BRASIL. Lei nº 12.587, de 3 de janeiro de 2012. Institui as diretrizes da Política Nacional de
Mobilidade Urbana. **Diário Oficial da União**: seção 1, Brasília, DF, 4 jan. 2012.

BRASIL. Lei nº 13.709, de 14 de agosto de 2018. Lei Geral de Proteção de Dados Pessoais (LGPD).
**Diário Oficial da União**: seção 1, Brasília, DF, 15 ago. 2018.

BUCH, N.; VELASTIN, S. A.; ORWELL, J. A review of computer vision techniques for the analysis of
urban traffic. **IEEE Transactions on Intelligent Transportation Systems**, [*s. l.*], v. 12,
n. 3, p. 920-939, 2011. DOI: 10.1109/TITS.2011.2119372.

DAI, Z. *et al.* Video-based vehicle counting framework. **IEEE Access**, [*s. l.*], v. 7,
p. 64460-64470, 2019. DOI: 10.1109/ACCESS.2019.2914254.

DATONDJI, S. R. E. *et al.* A survey of vision-based traffic monitoring of road intersections.
**IEEE Transactions on Intelligent Transportation Systems**, [*s. l.*], v. 17, n. 10,
p. 2681-2698, 2016. DOI: 10.1109/TITS.2016.2530146.

DIMITRAKOPOULOS, G.; DEMESTICHAS, P. Intelligent transportation systems. **IEEE Vehicular
Technology Magazine**, [*s. l.*], v. 5, n. 1, p. 77-84, 2010. DOI: 10.1109/MVT.2009.935537.

GIRSHICK, R. *et al.* Rich feature hierarchies for accurate object detection and semantic
segmentation. *In*: IEEE CONFERENCE ON COMPUTER VISION AND PATTERN RECOGNITION (CVPR), 2014,
Columbus. **Proceedings** [...]. Piscataway: IEEE, 2014. p. 580-587. DOI: 10.1109/CVPR.2014.81.

GONZALEZ, R. C.; WOODS, R. E. **Digital image processing**. 4. ed. New York: Pearson, 2018.

GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A. **Deep learning**. Cambridge: MIT Press, 2016.

JENSEN, M. B. *et al.* Vision for looking at traffic lights: issues, survey, and perspectives.
**IEEE Transactions on Intelligent Transportation Systems**, [*s. l.*], v. 17, n. 7,
p. 1800-1815, 2016. DOI: 10.1109/TITS.2015.2509509.

JOCHER, G.; QIU, J. **Ultralytics YOLO11**. Versão 11.0.0. [*S. l.*]: Ultralytics, 2024.
Software. Disponível em: https://github.com/ultralytics/ultralytics. Acesso em: 15 nov. 2026.

KALMAN, R. E. A new approach to linear filtering and prediction problems. **Journal of Basic
Engineering**, [*s. l.*], v. 82, n. 1, p. 35-45, 1960. DOI: 10.1115/1.3662552.

KHANAM, R.; HUSSAIN, M. **YOLOv11**: an overview of the key architectural enhancements.
[*S. l.*]: arXiv, 2024. DOI: 10.48550/arXiv.2410.17725.

KRAKAUER, E.; PEDRINI, H. **Otimização do fluxo de veículos**: contagem volumétrica
automatizada. Campinas: Instituto de Computação, Universidade Estadual de Campinas, 2018.
Relatório técnico IC-PFG-18-23. Disponível em:
https://www.ic.unicamp.br/~reltech/PFG/2018/PFG-18-23.pdf. Acesso em: 15 nov. 2026.

KRIZHEVSKY, A.; SUTSKEVER, I.; HINTON, G. E. ImageNet classification with deep convolutional
neural networks. **Communications of the ACM**, [*s. l.*], v. 60, n. 6, p. 84-90, 2017. DOI:
10.1145/3065386.

KUHN, H. W. The Hungarian method for the assignment problem. **Naval Research Logistics
Quarterly**, [*s. l.*], v. 2, n. 1-2, p. 83-97, 1955. DOI: 10.1002/nav.3800020109.

LECUN, Y. *et al.* Gradient-based learning applied to document recognition. **Proceedings of the
IEEE**, [*s. l.*], v. 86, n. 11, p. 2278-2324, 1998. DOI: 10.1109/5.726791.

LECUN, Y.; BENGIO, Y.; HINTON, G. Deep learning. **Nature**, [*s. l.*], v. 521, n. 7553,
p. 436-444, 2015. DOI: 10.1038/nature14539.

LIN, T.-Y. *et al.* Microsoft COCO: common objects in context. *In*: EUROPEAN CONFERENCE ON
COMPUTER VISION (ECCV), 2014, Zurich. **Computer Vision – ECCV 2014**. Cham: Springer, 2014.
p. 740-755. (Lecture Notes in Computer Science). DOI: 10.1007/978-3-319-10602-1_48.

LUO, W. *et al.* Multiple object tracking: a literature review. **Artificial Intelligence**,
[*s. l.*], v. 293, art. 103448, 2021. DOI: 10.1016/j.artint.2020.103448.

MANDAL, V.; ADU-GYAMFI, Y. Object detection and tracking algorithms for vehicle counting: a
comparative analysis. **Journal of Big Data Analytics in Transportation**, [*s. l.*], v. 2, n. 3,
p. 251-261, 2020. DOI: 10.1007/s42421-020-00025-w.

NAPHADE, M. *et al.* The 5th AI City Challenge. *In*: IEEE/CVF CONFERENCE ON COMPUTER VISION AND
PATTERN RECOGNITION WORKSHOPS (CVPRW), 2021, [*s. l.*]. **Proceedings** [...]. Piscataway: IEEE,
2021. p. 4258-4268. DOI: 10.1109/CVPRW53098.2021.00482.

NEITZKE, L. O.; SANO, F. K. S.; TAMPELINI, L. G. Visão computacional aplicada a quantificação do
fluxo de veículos. *In*: CONGRESSO LATINO-AMERICANO DE SOFTWARE LIVRE E TECNOLOGIAS ABERTAS
(LATINOWARE), 20., 2023, Foz do Iguaçu. **Anais** [...]. Porto Alegre: Sociedade Brasileira de
Computação, 2023. p. 158-161. DOI: 10.5753/latinoware.2023.236524.

ONG, C. S.; CONNIE, T.; ONG GOH, M. K. Traffic violation detection using computer vision
techniques. **JOIV: International Journal on Informatics Visualization**, [*s. l.*], v. 8,
n. 3-2, p. 1515, 2024. DOI: 10.62527/joiv.8.3-2.2941.

REDMON, J. *et al.* You only look once: unified, real-time object detection. *In*: IEEE
CONFERENCE ON COMPUTER VISION AND PATTERN RECOGNITION (CVPR), 2016, Las Vegas. **Proceedings**
[...]. Piscataway: IEEE, 2016. p. 779-788. DOI: 10.1109/CVPR.2016.91.

REDMON, J.; FARHADI, A. YOLO9000: better, faster, stronger. *In*: IEEE CONFERENCE ON COMPUTER
VISION AND PATTERN RECOGNITION (CVPR), 2017, Honolulu. **Proceedings** [...]. Piscataway: IEEE,
2017. p. 6517-6525. DOI: 10.1109/CVPR.2017.690.

REDMON, J.; FARHADI, A. **YOLOv3**: an incremental improvement. [*S. l.*]: arXiv, 2018. DOI:
10.48550/arXiv.1804.02767.

REN, S. *et al.* Faster R-CNN: towards real-time object detection with region proposal networks.
**IEEE Transactions on Pattern Analysis and Machine Intelligence**, [*s. l.*], v. 39, n. 6,
p. 1137-1149, 2017. DOI: 10.1109/TPAMI.2016.2577031.

RETTING, R. A.; FERGUSON, S. A.; HAKKERT, A. S. Effects of red light cameras on violations and
crashes: a review of the international literature. **Traffic Injury Prevention**, [*s. l.*],
v. 4, n. 1, p. 17-23, 2003. DOI: 10.1080/15389580309858.

SONG, H. *et al.* Vision-based vehicle detection and counting system using deep learning in
highway scenes. **European Transport Research Review**, [*s. l.*], v. 11, art. 51, 2019. DOI:
10.1186/s12544-019-0390-4.

SZELISKI, R. **Computer vision**: algorithms and applications. 2. ed. Cham: Springer, 2022.

TERVEN, J.; CÓRDOVA-ESPARZA, D.-M.; ROMERO-GONZÁLEZ, J.-A. A comprehensive review of YOLO
architectures in computer vision: from YOLOv1 to YOLOv8 and YOLO-NAS. **Machine Learning and
Knowledge Extraction**, [*s. l.*], v. 5, n. 4, p. 1680-1716, 2023. DOI: 10.3390/make5040083.

WANG, C.-Y.; BOCHKOVSKIY, A.; LIAO, H.-Y. M. YOLOv7: trainable bag-of-freebies sets new
state-of-the-art for real-time object detectors. *In*: IEEE/CVF CONFERENCE ON COMPUTER VISION AND
PATTERN RECOGNITION (CVPR), 2023, Vancouver. **Proceedings** [...]. Piscataway: IEEE, 2023.
p. 7464-7475. DOI: 10.1109/CVPR52729.2023.00721.

WOJKE, N.; BEWLEY, A.; PAULUS, D. Simple online and realtime tracking with a deep association
metric. *In*: IEEE INTERNATIONAL CONFERENCE ON IMAGE PROCESSING (ICIP), 2017, Beijing.
**Proceedings** [...]. Piscataway: IEEE, 2017. p. 3645-3649. DOI: 10.1109/ICIP.2017.8296962.

WORLD HEALTH ORGANIZATION. **Global status report on road safety 2023**. Geneva: WHO, 2023.
ISBN 978-92-4-008651-7.

ZHANG, J. *et al.* Data-driven intelligent transportation systems: a survey. **IEEE Transactions
on Intelligent Transportation Systems**, [*s. l.*], v. 12, n. 4, p. 1624-1639, 2011. DOI:
10.1109/TITS.2011.2158001.

ZHANG, Y. *et al.* ByteTrack: multi-object tracking by associating every detection box. *In*:
EUROPEAN CONFERENCE ON COMPUTER VISION (ECCV), 2022, Tel Aviv. **Computer Vision – ECCV 2022**.
Cham: Springer, 2022. p. 1-21. (Lecture Notes in Computer Science, v. 13682). DOI:
10.1007/978-3-031-20047-2_1.

ZOU, Z. *et al.* Object detection in 20 years: a survey. **Proceedings of the IEEE**, [*s. l.*],
v. 111, n. 3, p. 257-276, 2023. DOI: 10.1109/JPROC.2023.3238524.
