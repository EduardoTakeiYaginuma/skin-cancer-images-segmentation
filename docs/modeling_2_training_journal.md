# EfficientNet-B0: Jornada de Treinamento e Calibração

**Notebook:** `modeling_2.ipynb`  
**Modelo base:** EfficientNet-B0 com pré-treino ImageNet  
**Dataset:** ISIC 2020 (split 70/15/15, balanceamento 1:3 mel:non-mel no treino)

---

## Contexto

O `modeling_2.ipynb` replica o experimento `base_224x224` do `04_modeling.ipynb` (AUC 0.9039) e adiciona análise clínica detalhada. Ao longo do processo, identificamos e corrigimos três problemas encadeados: **double-weighting**, **threshold não calibrado** e **overfitting**. Este documento registra cada iteração, o diagnóstico feito e o resultado obtido.

---

## Iteração 1 — Modelo original carregado (sem re-treino)

**Configuração:**
- Modelo salvo do `04_modeling.ipynb` carregado diretamente
- `BCEWithLogitsLoss` com `pos_weight = neg/pos ≈ 3`
- `WeightedRandomSampler` com upsampling de melanoma (ratio 1:3)
- Meta de sensibilidade: **95%**

**Resultados no test set:**

| Métrica | Valor |
|---|---|
| AUC | 0.9039 |
| Threshold | **0.0038** |
| Sensibilidade | 0.9401 |
| Especificidade | 0.6430 |
| Precisão | 0.2476 |
| F1 | 0.3920 |
| FN | 10 |
| FP | 477 |
| % predito como melanoma | 42.2% |

**Diagnóstico:** O threshold de 0.0038 revelou um problema de calibração — para atingir 95% de sensibilidade, o modelo precisava de um limiar absurdamente baixo. A distribuição de probabilidades estava comprimida próximo a zero para quase todos os casos.

**Causa raiz identificada: double-weighting.** O desbalanceamento de classes estava sendo corrigido duas vezes simultaneamente:
1. `WeightedRandomSampler` → upsamples melanoma nos batches (~1:3 ratio)
2. `pos_weight ≈ 3` no `BCEWithLogitsLoss` → aplica peso adicional na loss

Essa correção dupla força o modelo a emitir logits muito negativos, comprimindo todas as probabilidades perto de zero. O AUC alto (0.90) confirma que o modelo discrimina bem, mas as probabilidades absolutas não têm interpretação confiável.

---

## Iteração 2 — Remoção do `pos_weight` (re-treino)

**Mudança aplicada:**
```python
# Antes
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg_count / pos_count], device=DEVICE))

# Depois
criterion = nn.BCEWithLogitsLoss()
```

**Meta de sensibilidade:** 95% (mantida)

**Resultados no test set:**

| Métrica | Valor | vs Iter. 1 |
|---|---|---|
| AUC | 0.8908 | −0.013 |
| Threshold | **0.0002** | **Piorou** |
| Sensibilidade | 0.9701 | +0.03 |
| Especificidade | 0.4828 | −0.16 |
| Precisão | 0.1899 | −0.06 |
| F1 | 0.3176 | −0.07 |
| FN | 5 | −5 |
| FP | 691 | +214 |

**Diagnóstico:** A remoção do `pos_weight` não resolveu a calibração — o threshold caiu de 0.0038 para 0.0002. O problema ficou ainda pior porque o `WeightedRandomSampler` apresenta ~50% de melanoma nos batches, mas sem `pos_weight` a loss trata as duas classes igualmente. O modelo aprende com uma distribuição de treino artificial e no momento da inferência (11% melanoma real) produz probabilidades ainda mais comprimidas.

**Insight importante:** A distribuição de probabilidades para melanoma teve mediana 0.97 (bem calibrada para os casos fáceis), mas 5 melanomas receberam probabilidades ≈ 0:

```
FN probabilities: [0.000012, 0.000027, 0.0001, 0.0001, 0.00014]
```

Esses 5 casos são melanomas atípicos que o modelo confunde completamente com não-melanoma. Para atingir 95% de sensibilidade e capturá-los, o threshold precisa descer até 0.0002 — arrastando junto 691 falsos alarmes.

**Conclusão:** A meta de 95% de sensibilidade com casos genuinamente difíceis força o threshold para o chão. O problema não é apenas calibração — é a combinação de casos extremamente difíceis com uma meta muito agressiva.

---

## Iteração 3 — Relaxamento da meta de sensibilidade para 85%

**Mudança aplicada:**
```python
# Antes
MELANOMA_RECALL_TARGET = 0.95

# Depois
MELANOMA_RECALL_TARGET = 0.85
```

**Sem re-treino** — apenas a seleção de threshold foi recalculada na validação.

**Resultados no test set:**

| Métrica | Valor | vs Iter. 2 |
|---|---|---|
| AUC | 0.8908 | = |
| Threshold | **0.0131** | Melhorou |
| Sensibilidade | 0.8323 | −0.14 |
| Especificidade | 0.7201 | +0.24 |
| Precisão | 0.2710 | +0.08 |
| F1 | 0.4088 | +0.09 |
| FN | 28 | +23 |
| FP | 374 | −317 |

**Três zonas (primeira versão funcional):**

| Zona | Total | % total | Melanomas | Taxa mel |
|---|---|---|---|---|
| Não melanoma | 553 | 36.8% | 2 | 0.4% |
| Possível melanoma | 437 | 29.1% | 26 | 5.9% |
| Melanoma | 513 | 34.1% | 139 | 27.1% |

**Diagnóstico:** O threshold subiu para 0.013 (melhora real, mas ainda baixo). O sistema de três zonas começa a funcionar clinicamente. O problema principal restante: overfitting forte (train AUC → 0.999, val AUC ≈ 0.88), que mantém a calibração ruim.

---

## Iteração 4 — Data Augmentation (configuração final)

**Problema identificado:** O flag `AUGMENT` existia no código mas o `__getitem__` do `SkinDataset` não o utilizava — augmentation era um placeholder sem efeito.

**Mudanças aplicadas:**

1. `AUGMENT = True` na configuração
2. Implementação real da augmentation no `SkinDataset.__getitem__`:

```python
def __getitem__(self, idx):
    row = self.records.iloc[idx]
    img = load_rgb_array(row["export_path"], IMAGE_SIZE)
    # float [0,1] antes da normalização para permitir color jitter
    t = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0)
    if self.augment:
        import torchvision.transforms.functional as TF
        if random.random() < 0.5:
            t = TF.hflip(t)
        if random.random() < 0.5:
            t = TF.vflip(t)
        t = TF.rotate(t, random.uniform(-180, 180))
        t = TF.adjust_brightness(t, 1.0 + random.uniform(-0.2, 0.2))
        t = TF.adjust_contrast(t,   1.0 + random.uniform(-0.2, 0.2))
        t = TF.adjust_saturation(t, 1.0 + random.uniform(-0.2, 0.2))
        t = t.clamp(0.0, 1.0)
    m = torch.tensor(NORM_MEAN, dtype=torch.float32).view(3, 1, 1)
    s = torch.tensor(NORM_STD,  dtype=torch.float32).view(3, 1, 1)
    t = (t - m) / s
    lbl = torch.tensor(int(row["binary_label"]), dtype=torch.float32)
    return t, lbl
```

Augmentations aplicadas (só no treino):
- Flip horizontal e vertical aleatório (p=0.5 cada)
- Rotação aleatória ±180° — lesões são invariantes a rotação
- Brightness, contraste e saturação com variação ±20%

**Meta de sensibilidade:** 85% (mantida)

**Resultados no test set:**

| Métrica | Valor | vs Iter. 3 | vs Original |
|---|---|---|---|
| AUC | **0.9015** | +0.011 | −0.002 |
| Threshold | **0.2208** | **+0.21** | **+0.22** |
| Sensibilidade | **0.9042** | +0.07 | −0.04 |
| Especificidade | **0.7433** | +0.02 | +0.10 |
| Precisão | **0.3057** | +0.03 | +0.06 |
| F1 | **0.4569** | +0.05 | +0.06 |
| FN | 16 | −12 | +6 |
| FP | **343** | −31 | **−134** |

**Três zonas (configuração final):**

| Zona | Total | % total | Melanomas | Taxa mel |
|---|---|---|---|---|
| Não melanoma | 573 | 38.1% | 1 | 0.2% |
| Possível melanoma | 436 | 29.0% | 15 | 3.4% |
| Melanoma | 494 | 32.9% | 151 | 30.6% |

**Curvas de treino — overfitting reduzido drasticamente:**

```
Sem augmentation:  época 5  → train AUC 0.989  val AUC 0.859  (gap 0.13)
Com augmentation:  época 9  → train AUC 0.933  val AUC 0.889  (gap 0.04)
```

---

## Resultado final e interpretação clínica

### Por que o threshold 0.22 importa

Nas iterações anteriores, thresholds de 0.0002 ou 0.013 não têm interpretação probabilística real. Com 0.22, é possível afirmar: *"o modelo considera que há pelo menos 22% de probabilidade de melanoma para acionar o alerta."* Isso é comunicável e auditável clinicamente.

### Sistema de triagem em três zonas

| Zona | Ação clínica sugerida | Garantia |
|---|---|---|
| **Não melanoma** (38% dos casos) | Dispensar — baixo risco | 99.8% são realmente benignos |
| **Possível melanoma** (29% dos casos) | Revisão dermatológica | 3.4% de taxa de melanoma real |
| **Melanoma** (33% dos casos) | Encaminhar com urgência | 30.6% de taxa de melanoma real |

A zona "Possível melanoma" contém 15 melanomas reais — casos borderline que o modelo honestamente não consegue classificar. São exatamente esses casos que justificam a revisão humana.

### Trade-off final vs modelo original

O resultado final aceita 6 melanomas a mais não detectados (16 vs 10 FN) em troca de:
- 134 alarmes falsos a menos (343 vs 477 FP)
- Threshold interpretável (0.22 vs 0.0038)
- Melhor especificidade (+10 p.p.)
- Melhor precisão (+6 p.p.)
- Sistema de três zonas com fronteiras significativas

---

## Lições aprendidas

1. **Double-weighting é um erro silencioso.** `WeightedRandomSampler` + `pos_weight` no mesmo pipeline corrige o desbalanceamento duas vezes, comprimindo as probabilidades sem afetar o AUC. O AUC alto pode mascarar calibração péssima.

2. **AUC não mede calibração.** Um modelo com AUC 0.90 e threshold 0.0002 é inútil clinicamente, mesmo que tecnicamente "discrimine bem".

3. **A meta de sensibilidade define o ponto operacional, não o treino.** Mudar `MELANOMA_RECALL_TARGET` não requer re-treino — apenas recalcula onde operar na curva ROC.

4. **Augmentation resolve overfitting e calibração ao mesmo tempo.** O gap train/val caiu de 0.13 para 0.04, e o threshold subiu de 0.013 para 0.22 — efeito direto da melhor generalização.

5. **Flags sem implementação são bugs silenciosos.** O `AUGMENT = True` não fazia nada até a augmentation ser implementada de fato no `__getitem__`.
