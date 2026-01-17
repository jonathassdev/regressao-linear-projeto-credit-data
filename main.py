import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

# ===============================
# 1. Tratamento dos Dados [Item Exigido]
# ===============================
def load_and_prepare_data(data_file_path):
    """Carrega os dados e define o Valor do Crédito (coluna 5) como alvo."""
    df = pd.read_csv(data_file_path, header=None, sep=r"\s+")
    
    # Alvo: Valor do Crédito (índice 4)
    y = df.iloc[:, 4].values 
    X = df.drop(df.columns[4], axis=1)
    
    # Transformação de categorias para numérico
    X = X.apply(lambda x: pd.factorize(x)[0] if x.dtype == object else x)
    
    # Imputação pela mediana
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)
    
    return X, y

def remove_outliers(X, y):
    """Remove outliers do alvo para estabilizar o modelo."""
    q1, q3 = np.percentile(y, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    
    mask = (y >= lower_bound) & (y <= upper_bound)
    return X[mask], y[mask]

# ===============================
# 2. Otimização de Hiperparâmetros [Item Exigido]
# ===============================
def optimize_hyperparameters(X_train, y_train):
    """Busca parâmetros via GridSearchCV."""
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=42), 
        param_grid, 
        cv=3, 
        scoring='r2'
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_

# ===============================
# 3. Avaliação dos Resultados [Item Exigido]
# ===============================
def plot_results(y_test, y_pred):
    """Gera gráficos para análise de regressão."""
    plt.figure(figsize=(12, 5))
    
    # Real vs Predito
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.5, color='teal')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Valores Reais')
    plt.ylabel('Previsões')
    plt.title('Real vs. Predito')

    # Resíduos
    plt.subplot(1, 2, 2)
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5, color='orange')
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Previsões')
    plt.ylabel('Resíduo (Erro)')
    plt.title('Distribuição de Resíduos')
    
    plt.tight_layout()
    plt.show()

# ===============================
# 4. Função Principal [Organização do Código]
# ===============================
def main():
    # Caminho absoluto conforme seu diretório
    data_path = r"C:/Users/Pichau/OneDrive/Documentos/Projeto-Regressao-CreditData/statlog+german+credit+data/german.data"
    
    print("--- Executando Regressão: German Credit Data ---")
    
    X, y = load_and_prepare_data(data_path)
    X, y = remove_outliers(X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print("Otimizando modelo...")
    best_params = optimize_hyperparameters(X_train, y_train)
    
    model = RandomForestRegressor(**best_params, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Cálculo das Métricas
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse) # NOVO: Raiz do Erro Quadrático Médio
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n--- Métricas de Regressão ---")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE (Erro Médio Absoluto): {mae:.2f}")
    print(f"MSE (Erro Quadrático Médio): {mse:.2f}")
    print(f"RMSE (Raiz do Erro Quadrático): {rmse:.2f} <-- Métrica na mesma escala do Crédito")
    
    print("\nExecutando Validação Cruzada...")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"R² Médio (CV): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    plot_results(y_test, y_pred)

if __name__ == "__main__":
    main()