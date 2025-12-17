# Projeto de Regressão Linear com Validação Cruzada e Otimização de Hiperparâmetros

## Descrição do Projeto

Este projeto tem como objetivo implementar um modelo de **Regressão Linear** utilizando a **Validação Cruzada** e a **Otimização de Hiperparâmetros** com base em uma base de dados do **german credit dataset**. Durante o processo, o modelo será treinado, avaliado e ajustado para melhor performance.

O fluxo do trabalho no projeto abrange desde a preparação dos dados até a avaliação final do modelo, utilizando métricas como **MAE**, **MSE**, **RMSE** e **R²**.

## 1. Tratamento dos Dados

### Passo a passo:

- **Carregamento e Preparação dos Dados**: O arquivo de dados `german.data` foi lido e convertido para formato numérico.
- **Imputação de Dados Faltantes**: Para tratar dados faltantes, utilizou-se a estratégia de **imputação com o valor mais frequente** nas variáveis independentes e dependentes.
- **Normalização dos Dados**: Utilizou-se o **StandardScaler** para normalizar os dados de treino e teste, melhorando a performance do modelo de regressão linear.

### Resultados do Tratamento:
Após a imputação, todos os valores ausentes foram preenchidos com os valores mais frequentes, e a normalização foi aplicada a ambas as variáveis de treino e teste.

## 2. Validação Cruzada

### O que foi feito:

Foi implementada uma **validação cruzada com 5 folds** para avaliar o desempenho do modelo e verificar sua robustez em diferentes subconjuntos dos dados.

- A métrica utilizada para validação foi o **RMSE** (Root Mean Squared Error).
- O RMSE médio foi calculado, bem como o desvio padrão, para entender a variabilidade do modelo em diferentes splits de dados.

### Resultado:
O RMSE médio foi calculado com base nos 5 folds, dando uma visão geral de como o modelo se comporta com diferentes conjuntos de dados.

## 3. Otimização de Hiperparâmetros

### O que foi feito:

A otimização dos hiperparâmetros foi realizada utilizando o **GridSearchCV**. Isso permite buscar os melhores valores para os parâmetros do modelo e otimizar o desempenho.

- Para este projeto, otimizamos o modelo de **RandomForestRegressor**, alterando os parâmetros como:
  - `n_estimators`: Número de árvores
  - `max_depth`: Profundidade máxima das árvores
  - `min_samples_split`: Mínimo número de amostras para dividir um nó
  - `min_samples_leaf`: Mínimo número de amostras para ser uma folha
  
**Resultado**:
Os melhores hiperparâmetros encontrados foram exibidos para garantir que o modelo esteja configurado para o melhor desempenho possível.

## 4. Avaliação dos Resultados

### O que foi feito:

Foram calculadas diversas métricas para avaliar o modelo de regressão linear, incluindo:
- **MAE (Mean Absolute Error)**: Mede a média das diferenças absolutas entre os valores previstos e os reais.
- **MSE (Mean Squared Error)**: Mede a média dos quadrados das diferenças.
- **RMSE (Root Mean Squared Error)**: A raiz quadrada do MSE, tornando a métrica mais interpretável em termos da mesma unidade de medida dos dados.
- **R² (Coeficiente de Determinação)**: Mede a proporção da variabilidade dos dados que é explicada pelo modelo.

### Resultado:
As métricas finais, incluindo **MAE**, **MSE**, **RMSE** e **R²**, foram exibidas para avaliar a qualidade do modelo.

## 5. Organização do Código

O código foi estruturado de maneira modular, com funções específicas para:
- Carregar e preparar os dados
- Treinar e avaliar o modelo
- Realizar validação cruzada
- Otimizar os hiperparâmetros
- Plotar os gráficos de resíduos

Esse formato facilita a manutenção do código e o entendimento das etapas do projeto.

## Como executar o projeto

### Pré-requisitos:
- Python 3.x
- Bibliotecas necessárias:
  - `pandas`
  - `numpy`
  - `sklearn`
  - `matplotlib`

### Passos para execução:
1. Clone este repositório no seu computador:
    ```bash
    git clone https://github.com/seu-usuario/nome-do-repositorio.git
    ```
2. Navegue até o diretório do projeto:
    ```bash
    cd nome-do-repositorio
    ```
3. Crie e ative um ambiente virtual:
    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # Windows
    source venv/bin/activate  # Linux/macOS
    ```
4. Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```
5. Execute o código:
    ```bash
    python main.py
    ```

### Observações:
- O arquivo `german.data` deve estar presente no diretório correto.
- Os resultados serão mostrados no terminal e o gráfico de resíduos será exibido.

## Conclusão

Este projeto demonstrou o uso de regressão linear, validação cruzada, otimização de hiperparâmetros e avaliação dos resultados. A implementação é flexível e pode ser adaptada para outros datasets de aprendizado supervisionado.
