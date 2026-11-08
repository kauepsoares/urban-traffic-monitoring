# Fase 4 — Estado do semáforo (versão 1: regras em HSV)

Código: `src/traffic_light.py`. Configurações: `configs/aic21_cam_*.yaml`.

```powershell
python -m src.traffic_light --config configs/aic21_cam_7.yaml            # 30 min em ~45 s
python -m src.traffic_light --config configs/aic21_cam_7.yaml --video    # + vídeo anotado
```

Saídas em `outputs/<video>_semaforo/`: `states.csv` (estado por quadro), `phases.csv`
(fases contínuas), `timeline.png` e `samples.jpg` (mosaico para conferência visual).

## Método final

1. **Localização com compensação de balanço.** A câmera balança até ~20 px (vento). O grupo
   focal é reencontrado a cada quadro por correlação de template, numa janela de ±6 px em
   torno da posição anterior, limitada a ±25 px da referência. A correlação usa um mapa de
   silhueta, `max(255 − V, S)`, em que a caixa do semáforo tem a mesma forma com qualquer
   lâmpada acesa. Abaixo de 0,6 de correlação, mantém-se a última posição (ex.: caminhão
   passando atrás do semáforo).
2. **Cor acesa por contagem de pixels coloridos.** Pixels com S ≥ 90 e V ≥ 110 são separados
   por matiz: vermelho (H ≤ 12 ou ≥ 160), amarelo (20–40), verde-azulado (60–105). A cor com
   mais pixels vence, se tiver ≥ 0,5% da área (mín. 2 px) e o dobro da segunda colocada.
3. **Suavização com regra do ciclo.** O candidato é a maioria dos últimos 5 quadros. Só
   substitui o estado atual se seguir o ciclo verde → amarelo → vermelho → verde; uma troca
   fora do ciclo só é aceita se persistir por 3 s.

## Tentativas descartadas (para a seção de metodologia)

| Tentativa | Problema observado |
| --- | --- |
| Lâmpada mais brilhante (canal V) por posição (cima/meio/baixo) | Caminhões claros atrás do semáforo e caixa desalinhada geravam falsos vermelhos |
| Mesma ideia com busca local, exigência de cor e regra do ciclo | Funcionou de dia (cam_7), mas **99%** "desconhecido" ao amanhecer: o sol baixo faz as lentes apagadas brilharem quase como a acesa |
| Brilho colorido S×V por posição | Corrigiu parte do amanhecer, mas o vermelho de dia é pálido (S baixo) e o halo do vermelho na chuva vaza para a posição do amarelo |
| Template em tons de cinza | Na chuva, o halo do vermelho faz parte do template; quando o verde acende a caixa desliza ~10 px e perde a lâmpada |

O que resolveu: decidir pela **cor** (não pela posição nem pelo brilho) e localizar pela
**silhueta** (não pela imagem em cinza).

## Resultados (AIC21 Track 1, Dataset A)

Validação visual: 180 recortes conferidos (cam_7, cam_7_dawn e cam_13, 60 cada), todos
corretos. Como checagem de plausibilidade, a duração do amarelo deve ficar entre 3 e 5 s
(padrão nos EUA) e o ciclo da cam_7 deve ser constante.

| Vídeo | Condição | Duração | "Desconhecido" por quadro | Amarelo (s) | Observação |
| --- | --- | --- | --- | --- | --- |
| cam_7 | dia | 30 min | 9,6% | 3,8–4,4 (14 fases) | ciclo de ~130 s em todas as voltas; lâmpada de ~4 px |
| cam_7_dawn | amanhecer | 5 min | 0,1% | 4,0–4,1 | reflexo do sol nas lentes apagadas |
| cam_7_rain | chuva | 5 min | 0,2% | 3,9–4,0 | halo grande no vermelho |
| cam_10 | dia | 3,5 min | 0,5% | 3,8–3,9 | |
| cam_11 | dia | 3,5 min | 1,6% | 3,5–3,9 | |
| cam_12 | dia | 3,3 min | 0,1% | 3,0–4,0 | semáforo encostado na borda da imagem |
| cam_13 | dia | 3,3 min | 0,1% | 2,5–3,1 | verde curto (7–15 s), vermelho domina |

O "desconhecido" por quadro é antes da suavização; depois dela nenhum vídeo fica sem estado
(exceto o início, antes da primeira leitura). Processamento: 170–450 FPS só para o semáforo.

## Limitações e próximos passos

- Ainda não há gabarito anotado quadro a quadro; a avaliação acima é visual e por
  plausibilidade. Na Fase 7, anotar os instantes de troca de cor em uma amostra e medir a
  acurácia por quadro e o atraso na detecção de cada troca.
- A regra do ciclo assume o padrão verde → amarelo → vermelho (EUA/Brasil). Setas de
  conversão e semáforos piscantes não são tratados.
- Versão 2 (prevista no plano): classificador treinado (LISA/Bosch + recortes destes vídeos),
  para comparar com esta versão por regras.
