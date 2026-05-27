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
│   ├── insurance.csv        # Dataset original (não modificar!)
│   ├── X_scaled.csv         # Variáveis de entrada normalizadas (gerado pela Etapa 1)
│   ├── y.csv                # Variável alvo: charges (gerado pela Etapa 1)
│   └── scaler.pkl           # Objeto StandardScaler salvo (necessário para predict.py)
│
├── notebooks/
│   └── 01_preparacao.ipynb  # ✅ Etapa 1 — concluída (Bia)
│   └── 02_clusterizacao.ipynb  # ✅ Etapa 2 — concluída (Valéria)
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

```text
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

	
----

## 🗓️ Próximas etapas

| Etapa                       | Responsável | Status                         |
| --------------------------- | ----------- | ------------------------------ |
| 1. Preparação do Dataset    | Bia         | ✅ Concluída                   |
| 2. Clusterização            | Valéria     | ✅ Concluída                   |
| 3. Regressão Global         | Fátima      | ⏳ Aguardando merge da Etapa 2 |
| 4. Regressão por Cluster    | Ian         | 🔒 Aguarda Etapa 2             |
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
