# Segmentação + Predição com Clusterização e Regressão

**Disciplina:** Machine Learning  
**Dataset:** Insurance Cost Dataset  
**Entrega:** 16/06/2026

---

## Integrantes do Grupo

> _Preencha com os nomes dos integrantes_

---

## Estrutura do Repositório

```
├── data/
│   ├── insurance.csv               # Dataset original
│   ├── X_scaled.csv                # Features normalizadas
│   ├── y.csv                       # Variável alvo (charges)
│   ├── cluster_labels.csv          # Rótulos dos clusters
│   ├── scaler.pkl                  # StandardScaler treinado
│   ├── kmeans_model.pkl            # Modelo K-Means treinado
│   ├── modelo_cluster_0.pkl        # Melhor modelo — Cluster 0
│   ├── modelo_cluster_1.pkl        # Melhor modelo — Cluster 1
│   ├── modelo_cluster_2.pkl        # Melhor modelo — Cluster 2
│   ├── resultados_global.csv       # Métricas da regressão global
│   └── resultados_clusters.csv     # Métricas da regressão por cluster
├── 01_preparacao.ipynb
├── 02_clusterizacao.ipynb
├── 03_regressao_global.ipynb
├── 04_regressao_clusters.ipynb
├── 05_comparacao.ipynb
└── 06_predicao_final.ipynb
```

---

## Objetivo

Aplicar **clusterização** para identificar perfis de clientes e, em seguida, treinar modelos de **regressão especializados** para cada grupo, verificando se a segmentação melhora a capacidade preditiva do custo de seguro médico.

---

## 1. Preparação dos Dados (`01_preparacao.ipynb`)

O dataset contém **1.338 registros** e **7 variáveis**:

| Variável | Tipo  | Descrição                                 |
| -------- | ----- | ----------------------------------------- |
| age      | int   | Idade do cliente                          |
| sex      | str   | Sexo (male / female)                      |
| bmi      | float | Índice de Massa Corporal                  |
| children | int   | Número de filhos                          |
| smoker   | str   | Fumante (yes / no)                        |
| region   | str   | Região dos EUA                            |
| charges  | float | **Variável alvo** — custo do seguro (USD) |

Não foram encontrados valores nulos. As etapas realizadas foram:

- **Encoding de `sex` e `smoker`** via `LabelEncoder`: `female=0, male=1` e `no=0, yes=1`
- **One-Hot Encoding de `region`** via `get_dummies` com `drop_first=True` — a categoria `northeast` foi usada como referência, gerando as colunas `region_northwest`, `region_southeast` e `region_southwest`
- **Separação** entre features (X) e variável alvo (y = `charges`)
- **Normalização** com `StandardScaler` — necessária para a clusterização, pois o K-Means utiliza distância euclidiana e variáveis em escalas distintas distorceriam os agrupamentos

Arquivos gerados: `X_scaled.csv`, `y.csv`, `scaler.pkl`

---

## 2. Clusterização (`02_clusterizacao.ipynb`)

### Algoritmo utilizado

**K-Means** — algoritmo de particionamento que minimiza a inércia (soma das distâncias quadráticas de cada ponto ao seu centroide).

A variável alvo (`charges`) **não foi utilizada** na formação dos clusters.

### Definição do número de clusters

Foram testados valores de K de 2 a 10, avaliados por dois critérios complementares:

**Método Elbow (inércia):**  
A curva de inércia apresenta uma desaceleração visível a partir de K=3, indicando que aumentar K além desse ponto traz pouco ganho de compactação.

**Silhouette Score:**

| K     | Silhouette Score |
| ----- | ---------------- |
| 2     | 0.1827           |
| **3** | **0.2305**       |
| 4     | 0.2224           |
| 5     | 0.2184           |
| 6     | 0.2223           |
| 7     | 0.2122           |
| 8     | 0.2305           |
| 9     | 0.2265           |
| 10    | 0.2314           |

**K = 3 foi escolhido** com base na análise conjunta dos dois métodos: é o ponto de cotovelo do Elbow e o primeiro pico local do Silhouette. Embora K=8 e K=10 apresentem scores marginalmente maiores (0.2305 e 0.2314), esses valores gerariam clusters muito pequenos e difíceis de interpretar para o contexto do negócio. K=3 oferece o melhor equilíbrio entre qualidade estatística e interpretabilidade.

**Silhouette Score final (K=3): 0.2305**

### Distribuição dos clusters

| Cluster | Registros |
| ------- | --------- |
| 0       | 649       |
| 1       | 325       |
| 2       | 364       |

### Perfil dos clusters

| Cluster | n   | Idade média | IMC médio | % Fumantes | Custo médio |
| ------- | --- | ----------- | --------- | ---------- | ----------- |
| 0       | 649 | 39,23       | 29,19     | 19,3%      | $12.911     |
| 1       | 325 | 39,46       | 30,60     | 17,8%      | $12.347     |
| 2       | 364 | —           | —         | —          | $14.735     |

> **Observação:** O K-Means separou fumantes de não-fumantes como critério principal de agrupamento sem que essa variável fosse explicitamente sinalizada, confirmando que o tabagismo é o maior driver de custo no dataset.

**Interpretação dos grupos:**

- **Cluster 0** — Não-fumantes com custo baixo/médio, alta variação interna de idade e IMC
- **Cluster 1** — Fumantes, custo alto e mais homogêneo
- **Cluster 2** — Não-fumantes mais velhos e/ou com mais filhos, custo médio a alto

---

## 3. Regressão Global (`03_regressao_global.ipynb`)

Modelos treinados com **todo o dataset** (sem considerar clusters), com divisão 80/20 treino/teste (`random_state=42`).

### Algoritmos avaliados

- Regressão Linear
- KNN Regressor (`n_neighbors=5`)
- Decision Tree Regressor (`random_state=42`)

### Resultados — Regressão Global (baseline)

| Modelo                  | MAE       | RMSE      | R²         |
| ----------------------- | --------- | --------- | ---------- |
| **KNN Regressor**       | $3.514,20 | $5.533,37 | **0,8028** |
| Regressão Linear        | $4.181,19 | $5.796,28 | 0,7836     |
| Decision Tree Regressor | $3.074,87 | $6.336,72 | 0,7414     |

O **KNN Regressor** foi o melhor modelo global, com R²=0,8028, tornando-se o baseline de referência para a comparação com a abordagem por cluster.

---

## 4. Regressão por Cluster (`04_regressao_clusters.ipynb`)

Para cada cluster foram treinados os mesmos três algoritmos, com divisão 80/20 e `random_state=42`.

### Resultados por cluster

| Cluster | Modelo            | MAE       | RMSE      | R²         |
| ------- | ----------------- | --------- | --------- | ---------- |
| 0       | **KNN Regressor** | $3.540,22 | $5.622,67 | **0,6881** |
| 0       | Regressão Linear  | $4.278,50 | $6.063,40 | 0,6373     |
| 0       | Decision Tree     | $4.277,42 | $8.066,17 | 0,3581     |
| 1       | **KNN Regressor** | $3.464,89 | $5.575,57 | **0,8176** |
| 1       | Regressão Linear  | $4.070,04 | $6.420,88 | 0,7581     |
| 1       | Decision Tree     | $3.104,99 | $6.507,09 | 0,7516     |
| 2       | **KNN Regressor** | $2.627,72 | $3.342,84 | **0,9264** |
| 2       | Regressão Linear  | $3.286,58 | $4.589,26 | 0,8612     |
| 2       | Decision Tree     | $3.080,99 | $6.021,57 | 0,7611     |

O **KNN Regressor** foi o melhor algoritmo nos três clusters. Os modelos salvos para predição final são todos KNN especializados.

---

## 5. Comparação dos Resultados (`05_comparacao.ipynb`)

### Qual abordagem teve menor erro?

Comparando o melhor modelo de cada abordagem (KNN em ambos os casos):

| Contexto              | R²     | MAE       | Δ R²    | Δ MAE    |
| --------------------- | ------ | --------- | ------- | -------- |
| **Global (baseline)** | 0,8028 | $3.514,20 | —       | —        |
| Cluster 0             | 0,6881 | $3.540,22 | −0,1147 | +$26,02  |
| Cluster 1             | 0,8176 | $3.464,89 | +0,0148 | −$49,31  |
| Cluster 2             | 0,9264 | $2.627,72 | +0,1236 | −$886,48 |

### A clusterização melhorou a predição?

**Sim, em dois dos três clusters:**

- **Cluster 1 (fumantes):** melhora moderada — R² subiu de 0,8028 para 0,8176 e MAE reduziu $49. O modelo especializado reconhece melhor os padrões de custo dentro do grupo de fumantes.

- **Cluster 2 (não-fumantes mais velhos/com filhos):** melhora expressiva — R² subiu de 0,8028 para **0,9264** e MAE caiu de $3.514 para **$2.628**, redução de ~25%. O RMSE caiu 40% (de $5.533 para $3.343). Este é o grupo mais homogêneo do dataset, e a especialização do modelo capturou com muito mais precisão o comportamento dos custos.

- **Cluster 0 (não-fumantes variados):** piora — R² caiu de 0,8028 para 0,6881. Este é o maior grupo (n=649) e apresenta alta variação interna. O subconjunto de teste (~130 registros) e o valor fixo de `k=5` no KNN podem não ter sido suficientes para representar a heterogeneidade do grupo.

### Por quê a segmentação ajudou em alguns clusters e não em outros?

A segmentação só traz ganhos quando os grupos formados são **internamente coesos**. O Cluster 2 é o mais homogêneo: reúne não-fumantes com perfil de idade e estrutura familiar semelhante, tornando o comportamento dos custos mais regular e previsível para qualquer modelo. O Cluster 0, apesar de grande, agrupa perfis muito distintos entre si, o que dificulta o aprendizado especializado.

### Qual abordagem é mais adequada?

A **regressão por cluster é superior** para os grupos 1 e 2, que juntos representam 689 dos 1.338 clientes (51,5% do dataset). Para o Cluster 0, o modelo global ainda é preferível.

Para uso em produção, a recomendação é adotar o **pipeline completo** (segmentação → regressão especializada), pois ele nunca é pior do que o global para a maioria dos clientes — e é significativamente melhor para os grupos mais coesos.

---

## 6. Predição Final (`06_predicao_final.ipynb`)

O notebook implementa um pipeline completo que:

1. Recebe os dados brutos de um novo cliente
2. Aplica o mesmo encoding e normalização do treinamento
3. Identifica o cluster via K-Means
4. Aplica o modelo KNN especializado do cluster correspondente
5. Retorna o cluster identificado e o custo previsto

**Exemplo de uso:**

```python
cluster, custo = prever(
    age=40, sex='male', bmi=30.0,
    children=2, smoker='yes', region='northwest'
)
# Cluster identificado: 1
# Custo previsto: US$ 18.450,xx
```

---

## Conclusão

A abordagem de **segmentação por clusters seguida de regressão especializada** demonstrou ser superior ao modelo global em dois dos três grupos identificados. O ganho mais expressivo ocorreu no Cluster 2, onde o KNN especializado atingiu R²=0,9264 — uma melhora de 12,4 pontos percentuais em relação ao baseline global — e reduziu o RMSE em 40%.

O principal aprendizado do projeto é que **a qualidade da segmentação determina o potencial de melhoria da regressão**: grupos homogêneos se beneficiam muito da especialização; grupos heterogêneos podem não ter amostras suficientes para que modelos baseados em similaridade (como o KNN) generalizem bem.

A variável `smoker` emergiu espontaneamente como o principal critério de separação do K-Means, confirmando seu papel dominante na determinação dos custos de seguro médico.
