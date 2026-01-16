# 📂 Projeto de Regressão: Previsão de Risco de Crédito

Este repositório contém um projeto de Machine Learning focado na previsão de risco de crédito utilizando o dataset **German Credit Data**. O projeto implementa técnicas de regressão, validação cruzada robusta e otimização de hiperparâmetros para prever a viabilidade de empréstimos.

---

## 📝 Descrição do Projeto

O objetivo principal é prever se um cliente de banco será classificado como "bom" ou "ruim" para a concessão de crédito. Embora o problema seja originalmente de classificação, este projeto explora a abordagem via **Regressão** (utilizando `RandomForestRegressor`) para analisar a probabilidade e a magnitude dos riscos associados a variáveis como idade, status financeiro e histórico de crédito.

---

## 🛠️ Tecnologias e Bibliotecas

O projeto foi desenvolvido em **Python 3.x** utilizando as seguintes bibliotecas:

* **Manipulação de Dados:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn` (Random Forest, GridSearchCV, Cross-Validation)
* **Balanceamento:** `imbalanced-learn` (SMOTE)
* **Visualização:** `matplotlib`, `seaborn`

---

## 📊 Metodologia

O desenvolvimento seguiu um pipeline rigoroso de Ciência de Dados:

1.  **Carregamento e Imputação:** Conversão de dados para formato numérico e tratamento de valores ausentes utilizando a estratégia de moda (valor mais frequente).
2.  **Escalonamento:** Aplicação de `StandardScaler` para garantir que todas as variáveis estivessem na mesma escala, facilitando a convergência do modelo.
3.  **Balanceamento com SMOTE:** > No dataset **German Credit Data**, o desbalanceamento entre clientes "bons" e "ruins" pode enviesar o modelo. Utilizamos o **SMOTE (Synthetic Minority Over-sampling Technique)** para criar novos exemplos sintéticos da classe minoritária através da interpolação. Isso garante que o modelo aprenda as características dos clientes de alto risco em vez de apenas memorizar a classe majoritária.



4.  **Validação Cruzada:** Implementação de 5-fold CV para validar a capacidade de generalização e reduzir o risco de overfitting.
5.  **Otimização de Hiperparâmetros:** Uso de `GridSearchCV` para encontrar a configuração ideal de profundidade e número de árvores.

---

## 📈 Resultados Finais

Após o ajuste fino, o modelo apresentou os seguintes indicadores de performance:

### Métricas de Avaliação
| Métrica | Valor |
| :--- | :---: |
| **Acurácia Média (CV)** | 0.7911 |
| **Precisão** | 0.7909 |
| **Recall** | 0.7699 |
| **F1-Score** | 0.7803 |

### Melhores Hiperparâmetros Encontrados
* `max_depth`: 10
* `n_estimators`: 50
* `max_features`: 'sqrt'
* `min_samples_leaf`: 1
* `min_samples_split`: 2

### Análise de Resíduos
Durante a execução, são gerados gráficos para validar a qualidade das previsões:
* **Resíduos vs. Valores Reais:** Verifica se os erros são aleatórios (ideal) ou se seguem um padrão (indicando falha do modelo).
* **Erro Absoluto:** Mede a magnitude média dos desvios em relação ao valor real.
