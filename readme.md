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
│
├── predict.py               # Etapa 6 — a preencher (Todos)
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

## 🗓️ Próximas etapas

| Etapa                       | Responsável | Status                         |
| --------------------------- | ----------- | ------------------------------ |
| 1. Preparação do Dataset    | Bia         | ✅ Concluída                   |
| 2. Clusterização            | Valéria     | ⏳ Aguardando merge da Etapa 1 |
| 3. Regressão Global         | Fátima      | 🔒 Aguarda Etapa 2             |
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
