import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor  # Para Random Forest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ===============================
# 1. Carregamento dos dados
# ===============================
def load_and_prepare_data(data_file_path):
    """Carrega e prepara os dados."""
    df = pd.read_csv(data_file_path, header=None, sep=r"\s+")
    df = df.apply(pd.to_numeric, errors="coerce")
    
    # Imputação para variáveis independentes e dependentes
    imputer_X = SimpleImputer(strategy="most_frequent")
    imputer_y = SimpleImputer(strategy="most_frequent")
    
    X = imputer_X.fit_transform(df.iloc[:, :-1])  # variáveis independentes
    y = imputer_y.fit_transform(df.iloc[:, -1].values.reshape(-1, 1)).ravel()  # variável dependente
    return X, y

# ===============================
# 2. Modelo de Regressão (RandomForest)
# ===============================
def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Treina e avalia o modelo de regressão linear ou qualquer outro modelo."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)  # RandomForestRegressor
    model.fit(X_train, y_train)
    
    # Predições
    y_pred = model.predict(X_test)
    
    # Cálculo das métricas
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    return mae, mse, rmse, r2, y_pred

# ===============================
# 3. Validação Cruzada
# ===============================
def cross_validation(X, y):
    """Executa a validação cruzada com 5 folds."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)  # Usando RandomForestRegressor para validação cruzada
    scores = cross_val_score(model, X, y, cv=5, scoring="neg_root_mean_squared_error")
    return -scores.mean(), scores.std()  # Retorna RMSE médio e desvio padrão

# ===============================
# 4. Otimização de Hiperparâmetros
# ===============================
def optimize_hyperparameters(X_train, y_train):
    """Otimiza os hiperparâmetros do modelo usando GridSearchCV."""
    
    # Usando RandomForestRegressor para otimização
    model = RandomForestRegressor(random_state=42)
    
    # Definindo o grid de parâmetros para otimizar
    param_grid = {
        'n_estimators': [50, 100, 200],  # Número de árvores na floresta
        'max_depth': [10, 20, None],      # Profundidade máxima da árvore
        'min_samples_split': [2, 5, 10],  # Quantidade mínima de amostras para dividir um nó
        'min_samples_leaf': [1, 2, 4]     # Quantidade mínima de amostras para ser folha
    }
    
    # GridSearchCV para otimização de parâmetros
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring="neg_root_mean_squared_error")
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_params_  # Retorna os melhores parâmetros encontrados

# ===============================
# 5. Visualização (Gráfico de Resíduos)
# ===============================
def plot_residuals(y_test, y_pred):
    """Plota o gráfico de resíduos."""
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_test - y_pred, alpha=0.6)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Valores Reais")
    plt.ylabel("Resíduos")
    plt.title("Resíduos vs Valores Reais")
    plt.tight_layout()
    plt.show()

# ===============================
# 6. Função Principal
# ===============================
def main():
    # Caminho do arquivo
    data_file_path = "C:/Users/Pichau/OneDrive/Documentos/Projeto-Regressao-CreditData/statlog+german+credit+data/german.data"
    
    # Carregar e preparar dados
    X, y = load_and_prepare_data(data_file_path)
    
    # Dividir dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("\nDivisão de dados completada!")
    
    # Normalização dos dados (usando StandardScaler)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # Normalizando os dados de treino
    X_test = scaler.transform(X_test)  # Normalizando os dados de teste
    
    # Treinar e avaliar o modelo
    mae, mse, rmse, r2, y_pred = train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # Exibir as métricas
    print(f"\nMAE = {mae:.4f}")
    print(f"MSE = {mse:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"R² = {r2:.4f}")
    
    # Plotar os resíduos
    plot_residuals(y_test, y_pred)
    
    # Realizar validação cruzada
    rmse_cv, rmse_cv_std = cross_validation(X, y)
    print("\nValidação Cruzada (5-fold):")
    print("RMSE médio:", rmse_cv)
    print("RMSE desvio padrão:", rmse_cv_std)
    
    # Otimização de hiperparâmetros
    best_params = optimize_hyperparameters(X_train, y_train)
    print("\nMelhores hiperparâmetros encontrados:", best_params)

    # Salvar dados processados em arquivo CSV
    df_encoded = pd.DataFrame(X)  # Transformar X imputado novamente em DataFrame
    df_encoded["y_real"] = y      # Adicionar variável dependente
    df_encoded.to_csv("dados_processados.csv", index=False)
    print("\nArquivo 'dados_processados.csv' gerado com sucesso.")


if __name__ == "__main__":
    main()