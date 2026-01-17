# 📊 Predição de Valor de Crédito: Análise de Regressão
> **Projeto de Inteligência Artificial** | Unidades II e III
> **Base de Dados:** Statlog (German Credit Data)
> **Equipe 01:** Foco em Métricas de Regressão da Scikit-learn

---

## 🎯 Objetivo do Projeto
Desenvolver um modelo de aprendizado de máquina capaz de prever o **Valor do Crédito (Credit Amount)** solicitado por clientes, utilizando técnicas de regressão para análise de risco financeiro.

## 🛠️ Requisitos Atendidos
De acordo com as normas da Unidade III, o projeto cumpre os seguintes itens:

- [x] **Tratamento dos dados**
- [x] **Validação cruzada**
- [x] **Otimização dos hiperparâmetros**
- [x] **Avaliação dos resultados**
- [x] **Organização do código**

---

## 🚀 Processo de Desenvolvimento

### 1. Tratamento dos Dados
* **Definição do Alvo:** A coluna 5 (Credit Amount) foi selecionada como variável dependente para transformar o problema original em uma tarefa de regressão.
* **Limpeza:** Tratamento de valores nulos via mediana com `SimpleImputer`.
* **Outliers:** Remoção de valores extremos no alvo através do método IQR para evitar distorções no erro quadrático.
* **Escalonamento:** Aplicação do `RobustScaler` para normalizar as variáveis independentes.

### 2. Otimização e Algoritmo
Foi utilizado o **RandomForestRegressor**, um algoritmo de conjunto (ensemble) compatível com a natureza não-linear dos dados.
* **GridSearchCV:** Otimização automatizada dos hiperparâmetros `n_estimators`, `max_depth` e `min_samples_split`.

### 3. Validação Cruzada (K-Fold)
Para garantir a consistência estatística, aplicamos 5-folds de validação:
* **Média $R^2$ (CV):** 0.4619
* **Desvio Padrão:** 0.0172
* *Conclusão:* O modelo apresenta estabilidade e não indica sinais de overfitting.

---

## 📈 Resultados e Métricas (Scikit-learn)
As métricas abaixo foram extraídas para cumprir o seminário da Equipe 01:

| Métrica | Valor Obtido | Descrição |
| :--- | :--- | :--- |
| **$R^2$ Score** | 0.4682 | Percentual da variância explicada pelo modelo. |
| **MAE** | 955.41 | Erro Médio Absoluto em unidades monetárias. |
| **RMSE** | 1.262,52 | Raiz do Erro Quadrático Médio (Escala real). |

### Análise Visual
O gráfico de **Resíduos** (abaixo) confirma que o modelo captura a tendência central dos dados, apresentando maior dispersão apenas em créditos de valores muito elevados.



---

## 📂 Estrutura do Repositório
* `main.py`: Script principal com a lógica modularizada.
* `german.data`: Base de dados original.
* `/results`: Gráficos gerados durante a execução.

---
> **Nota:** Este projeto atende aos critérios de "Organização de Código" através de funções modulares e tipagem de dados.
