# 🏥 ML — Segmentação + Predição de Custos de Seguro

**Disciplina:** Machine Learning  
**Grupo:** Bia · Valéria · Fátima · Ian · Rebeca  
**Entrega:** 16/06/2026 · **Peso:** 20 pontos

---

> ⚠️ **README temporário** — mantido por Bia (Etapa 1).  
> Cada colega deve complementar com a sua seção após mergear a própria etapa.

---

## 📁 Estrutura atual do repositório

```
ml-segmentacao-predicao/
│
├── data/
│   ├── insurance.csv              # Dataset original (não modificar!)
│   ├── X_scaled.csv               # Variáveis de entrada normalizadas (Etapa 1)
│   ├── y.csv                      # Variável alvo: charges (Etapa 1)
│   ├── scaler.pkl                 # StandardScaler treinado (necessário para Etapa 6)
│   ├── cluster_labels.csv         # Rótulos dos clusters (Etapa 2)
│   ├── kmeans_model.pkl           # Modelo K-Means treinado (Etapa 2)
│   ├── resultados_global.csv      # Métricas dos modelos globais (Etapa 3)
│   ├── resultados_clusters.csv    # Métricas dos modelos por cluster (Etapa 4)
│   ├── modelo_cluster_0.pkl       # Melhor modelo — Cluster 0 (Etapa 4)
│   ├── modelo_cluster_1.pkl       # Melhor modelo — Cluster 1 (Etapa 4)
│   ├── modelo_cluster_2.pkl       # Melhor modelo — Cluster 2 (Etapa 4)
│   ├── comparacao_modelos_clusters.png  # R² e MAE por modelo/cluster (Etapa 4)
│   └── previsto_vs_real_clusters.png    # Previsto vs Real por cluster (Etapa 4)
│
├── notebooks/
│   ├── 01_preparacao.ipynb        # ✅ Etapa 1 — concluída (Bia)
│   ├── 02_clusterizacao.ipynb     # ✅ Etapa 2 — concluída (Valéria)
│   ├── 03_regressao_global.ipynb  # ✅ Etapa 3 — concluída (Fátima)
│   └── 04_regressao_clusters.ipynb # ✅ Etapa 4 — concluída (Ian)
│
└── README.md
```

---

## ⚙️ Como rodar o projeto (Windows + VS Code)

### 1. Pré-requisitos

- Python 3.8+
- Git
- VS Code com extensões **Python** e **Jupyter** instaladas

### 2. Clonar o repositório

```bash
git clone https://github.com/SEU_USUARIO/ml-segmentacao-predicao.git
cd ml-segmentacao-predicao
```

### 3. Criar e ativar o ambiente virtual

```bash
python -m venv venv
venv\Scripts\activate
```

> 💡 Lembre de ativar o `venv` toda vez que abrir o VS Code.

### 4. Instalar as dependências

```bash
pip install pandas scikit-learn matplotlib seaborn jupyter ipykernel joblib
```

### 5. Selecionar o kernel no VS Code

Abra qualquer notebook → canto superior direito → **Select Kernel** → escolha o interpretador da `venv`.

---

## ✅ Etapa 1 — Preparação do Dataset (`01_preparacao.ipynb`)

**Responsável:** Bia | **Revisora:** Valéria

### O que foi feito

- Carregamento do `insurance.csv` com pandas
- Encoding de variáveis categóricas:
  - `sex` e `smoker` → `LabelEncoder` (variáveis binárias)
  - `region` → `get_dummies` com `drop_first=True` (4 categorias)
- Normalização das variáveis de entrada com `StandardScaler`
- Separação entre `X` (variáveis de entrada) e `y` (variável alvo: `charges`)
- Exportação de `X_scaled.csv`, `y.csv` e `scaler.pkl` para a pasta `data/`

### Colunas após o encoding

| Coluna original | Transformação | Resultado                                                  |
| --------------- | ------------- | ---------------------------------------------------------- |
| `sex`           | LabelEncoder  | `0` = female, `1` = male                                   |
| `smoker`        | LabelEncoder  | `0` = no, `1` = yes                                        |
| `region`        | get_dummies   | `region_northwest`, `region_southeast`, `region_southwest` |
| `charges`       | —             | variável alvo (`y`), removida de `X`                       |

### Arquivos gerados

| Arquivo             | Descrição                                         |
| ------------------- | ------------------------------------------------- |
| `data/X_scaled.csv` | Entrada normalizada — use para Etapas 2, 3 e 4    |
| `data/y.csv`        | Variável alvo — use para Etapas 3 e 4             |
| `data/scaler.pkl`   | Scaler treinado — use na Etapa 6 (predição final) |

---

## ✅ Etapa 2 — Clusterização (02_clusterizacao.ipynb)

**Responsável:** Valéria | **Revisora:** Fátima

### O que foi feito

- Carregamento dos dados normalizados (`X_scaled.csv`) gerados na Etapa 1
- Aplicação do algoritmo de clusterização **K-Means**
- Avaliação do número ideal de clusters utilizando:
  - Método Elbow
  - Silhouette Score
- Escolha de `K=3` por apresentar boa separação dos grupos e melhor interpretabilidade
- Aplicação do PCA (Principal Component Analysis) para redução de dimensionalidade e visualização dos clusters em 2D
- Criação da coluna `cluster` no dataset
- Interpretação estatística dos grupos encontrados
- Exportação do dataset clusterizado e do modelo treinado

### Resultado da Clusterização

#### Silhouette Score final

K=3 → 0.2305

| Cluster | Quantidade |
| ------- | ---------- |
| 0       | 649        |
| 1       | 325        |
| 2       | 364        |


Interpretação dos Clusters
Cluster	Perfil resumido	Custo médio
- 0	Não fumantes com idade e IMC variados → perfil de menor risco	Baixo a médio
- 1	Fumantes com IMC elevado → perfil de maior risco e maior custo	Alto
- 2	Não fumantes mais velhos e/ou com mais filhos → perfil intermediário	Médio


Observação: A variável smoker demonstrou ser o principal fator de separação entre os clusters, indicando forte relação entre tabagismo e aumento dos custos de seguro de saúde.

Visualizações geradas
- Gráfico Elbow
- Gráfico Silhouette Score
- Visualização dos clusters com PCA
- Distribuição dos clusters por custo (charges)

Arquivos gerados

| Arquivo                       |  Descrição                    |
| -------                       | ----------                    |
| data/insurance_clustered.csv  | Dataset com coluna de cluster |
| models/kmeans_model.pkl       | Modelo K-Means treinado       |

## ✅ Etapa 3 — Regressão Global (`03_regressao_global.ipynb`)

**Responsável:** Fátima | **Revisor:** Ian

### O que foi feito
Nesta etapa, foram treinados modelos de regressão utilizando todo o conjunto de dados de forma integrada. O objetivo foi estabelecer uma linha de base (baseline) de desempenho preditivo global, permitindo avaliar posteriormente se a segmentação por clusters trará melhorias reais nas estimativas.

Foram testados e validados três algoritmos distintos conforme o pipeline definido:
- **Regressão Linear**
- **KNN Regressor** (configurado com `n_neighbors=5`)
- **Decision Tree Regressor** (configurado com `random_state=42`)

A base de dados foi dividida utilizando a proporção padrão de 80% para treinamento e 20% para a fase de testes.

### Métricas da Regressão Global
Abaixo está o mapeamento comparativo de desempenho de cada algoritmo, ordenado pelo coeficiente de determinação ($R^2$):

| Modelo | MAE | MSE | RMSE | R² |
| :--- | :--- | :--- | :--- | :--- |
| **Regressão Linear** | 4181.1944 | 33596915.8513 | 5796.2847 | 0.7833 |
| **KNN Regressor** | 4261.2725 | 38221805.9901 | 6182.3787 | 0.7535 |
| **Decision Tree Regressor** | 3073.4981 | 43232675.2758 | 6575.1559 | 0.7212 |

### Conclusões da Etapa
* O modelo de **Regressão Linear** obteve o maior coeficiente de determinação ($R^2 = 0.7833$), sendo capaz de explicar aproximadamente 78,33% da variabilidade dos custos de seguro de forma global.
* A **Árvore de Decisão** (*Decision Tree*) registrou o menor Erro Absoluto Médio ($MAE = 3073.4981$), o que indica que, na maior parte das previsões cotidianas, ela erra por margens menores. No entanto, por apresentar um $MSE$ mais elevado, conclui-se que o modelo sofreu penalizações severas por erros mais discrepantes (outliers) em casos específicos da distribuição.

*Nota: O arquivo contendo estes resultados foi exportado com sucesso para `data/resultados_global.csv`, estando pronto para consumo na Etapa 5 (Comparação de Resultados).*
	
## ✅ Etapa 4 — Regressão por Cluster (`04_regressao_clusters.ipynb`)

**Responsável:** Ian | **Revisora:** Rebeca

### O que foi feito

- Carregamento de `X_scaled.csv`, `y.csv` e `cluster_labels.csv` gerados nas etapas anteriores
- Para cada cluster (0, 1, 2): filtragem dos dados, divisão treino/teste (80%/20%, `random_state=42`) e treinamento dos mesmos 3 modelos da Etapa 3 para comparação direta
- Avaliação com MAE, MSE, RMSE e R²
- Identificação do melhor modelo por cluster (maior R²)
- Geração de visualizações comparativas (R² e MAE por modelo/cluster; Previsto vs Real)

### Métricas por Cluster

| Cluster | Modelo | MAE | RMSE | R² |
| :--- | :--- | :--- | :--- | :--- |
| 0 | Regressão Linear | $4,278.50 | $6,063.40 | 0.6373 |
| 0 | **KNN Regressor** | **$3,540.22** | **$5,622.67** | **0.6881** |
| 0 | Decision Tree | $4,109.21 | $7,829.78 | 0.3952 |
| 1 | Regressão Linear | $4,070.04 | $6,420.88 | 0.7581 |
| 1 | **KNN Regressor** | **$3,464.89** | **$5,575.57** | **0.8176** |
| 1 | Decision Tree | $3,104.99 | $6,507.09 | 0.7516 |
| 2 | Regressão Linear | $3,286.58 | $4,589.26 | 0.8612 |
| 2 | **KNN Regressor** | **$2,627.72** | **$3,342.84** | **0.9264** |
| 2 | Decision Tree | $3,080.99 | $6,021.57 | 0.7611 |

### Melhor modelo por cluster vs. Baseline Global

| | Modelo | R² | MAE |
| :--- | :--- | :--- | :--- |
| Baseline Global | KNN Regressor | 0.8028 | $3,514.20 |
| Cluster 0 | KNN Regressor | 0.6881 | $3,540.22 |
| Cluster 1 | KNN Regressor | 0.8176 | $3,464.89 |
| Cluster 2 | KNN Regressor | 0.9264 | $2,627.72 |

**KNN Regressor foi o melhor modelo nos 3 clusters.** Cluster 2 (não-fumantes mais velhos/com filhos) apresentou o maior ganho com a segmentação (R²=0.9264 vs 0.8028 do baseline). Cluster 0 ficou abaixo do baseline global em R², o que pode indicar maior heterogeneidade interna neste grupo.

### Arquivos gerados

| Arquivo | Descrição |
| --- | --- |
| `data/resultados_clusters.csv` | Métricas de todos os modelos por cluster |
| `data/modelo_cluster_0.pkl` | KNeighborsRegressor treinado — Cluster 0 |
| `data/modelo_cluster_1.pkl` | KNeighborsRegressor treinado — Cluster 1 |
| `data/modelo_cluster_2.pkl` | KNeighborsRegressor treinado — Cluster 2 |
| `data/comparacao_modelos_clusters.png` | R² e MAE comparativos com linha de baseline |
| `data/previsto_vs_real_clusters.png` | Scatter Previsto vs Real por cluster |

----

## 🗓️ Próximas etapas

| Etapa                       | Responsável | Status                         |
| --------------------------- | ----------- | ------------------------------ |
| 1. Preparação do Dataset    | Bia         | ✅ Concluída                   |
| 2. Clusterização            | Valéria     | ✅ Concluída                   |
| 3. Regressão Global         | Fátima      | ✅ Concluída                   |
| 4. Regressão por Cluster    | Ian         | ✅ Concluída                   |
| 5. Comparação de Resultados | Rebeca      | 🔒 Aguarda Etapas 3 e 4        |
| 6. Predição Final           | Todos       | 🔒 Aguarda Etapa 5             |

---

## 👥 Integrantes

| Nome    | Etapa líder                     | Etapa revisora |
| ------- | ------------------------------- | -------------- |
| Bia     | Etapa 1 — Preparação            | Etapa 5        |
| Valéria | Etapa 2 — Clusterização         | Etapa 1        |
| Fátima  | Etapa 3 — Regressão Global      | Etapa 2        |
| Ian     | Etapa 4 — Regressão por Cluster | Etapa 3        |
| Rebeca  | Etapa 5 — Comparação            | Etapa 4        |

---

_Documento iniciado em 26/05/2026 por Bia — atualizar após cada merge._
