import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier  # Alterado para Classificador
from sklearn.linear_model import LogisticRegression  # Modelo de regressão logística
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE  # Para balanceamento de dados
import xgboost as xgb  # XGBoost
import lightgbm as lgb  # LightGBM

# ===============================
# 1. Carregamento e preparação dos dados
# ===============================
def load_and_prepare_data(data_file_path):
    """Carrega os dados do arquivo, trata valores faltantes e retorna as variáveis independentes (X) e dependentes (y)."""
    df = pd.read_csv(data_file_path, header=None, sep=r"\s+")
    df = df.apply(pd.to_numeric, errors="coerce")  # Converte todos os valores para numérico (ignora erros)
    
    # Imputação de valores faltantes: substitui valores ausentes pelo valor mais frequente
    imputer_X = SimpleImputer(strategy="most_frequent")
    imputer_y = SimpleImputer(strategy="most_frequent")
    
    # Divisão entre variáveis independentes (X) e dependentes (y)
    X = imputer_X.fit_transform(df.iloc[:, :-1])  # Variáveis independentes
    y = imputer_y.fit_transform(df.iloc[:, -1].values.reshape(-1, 1)).ravel()  # Variável dependente
    
    return X, y

# ===============================
# 2. Remoção de Outliers (detecção utilizando IQR)
# ===============================
def remove_outliers(X, y):
    """Detecta e remove outliers utilizando o método do Intervalo Interquartil (IQR)."""
    Q1 = np.percentile(X, 25, axis=0)  # Primeiro quartil
    Q3 = np.percentile(X, 75, axis=0)  # Terceiro quartil
    IQR = Q3 - Q1  # Intervalo Interquartil
    mask = (X >= (Q1 - 1.5 * IQR)) & (X <= (Q3 + 1.5 * IQR))  # Definindo limites para remoção de outliers
    mask = np.all(mask, axis=1)  # Verificando se todos os dados estão dentro dos limites
    X_clean = X[mask]
    y_clean = y[mask]  # Removendo as mesmas amostras de y para garantir consistência
    return X_clean, y_clean

# ===============================
# 3. Treinamento e Avaliação do Modelo
# ===============================
def train_and_evaluate(X_train, X_test, y_train, y_test, model):
    """Treina o modelo com os dados de treino e avalia com os dados de teste."""
    model.fit(X_train, y_train)  # Treinamento do modelo
    
    # Realiza previsões com o modelo treinado
    y_pred = model.predict(X_test)
    
    # Calcula as métricas de desempenho
    precision = precision_score(y_test, y_pred)  # Precisão
    recall = recall_score(y_test, y_pred)  # Recall
    f1 = f1_score(y_test, y_pred)  # F1-Score
    cm = confusion_matrix(y_test, y_pred)  # Matriz de confusão
    
    return precision, recall, f1, cm, y_pred

# ===============================
# 4. Validação Cruzada (5-fold)
# ===============================
def cross_validation(X, y, model):
    """Executa a validação cruzada com 5 folds para avaliar a consistência do modelo."""
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    return scores.mean(), scores.std()  # Acurácia média e desvio padrão

# ===============================
# 5. Ajuste de Hiperparâmetros com Grid Search
# ===============================
def optimize_hyperparameters(X_train, y_train, model):
    """Busca os melhores hiperparâmetros para o modelo utilizando GridSearchCV."""
    
    if isinstance(model, RandomForestClassifier):  # Para o RandomForest
        param_grid = {
            'n_estimators': [50, 100, 200, 300],  # Número de árvores na floresta
            'max_depth': [10, 20, None, 30],      # Profundidade máxima da árvore
            'min_samples_split': [2, 5, 10],  # Quantidade mínima para dividir um nó
            'min_samples_leaf': [1, 2, 4],     # Quantidade mínima para ser folha
            'max_features': ['auto', 'sqrt', 'log2']  # Critério de divisão das árvores
        }
    
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring="accuracy", verbose=2)
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_params_

# ===============================
# 6. Visualização de Resultados
# ===============================
def plot_residuals(y_test, y_pred):
    """Plota gráficos de resíduos e erro absoluto para avaliação do modelo."""
    # Gráfico de resíduos
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_test - y_pred, alpha=0.6)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Valores Reais")
    plt.ylabel("Resíduos")
    plt.title("Resíduos vs Valores Reais")
    plt.tight_layout()
    plt.show()

    # Gráfico de erro absoluto
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, np.abs(y_test - y_pred), alpha=0.6)
    plt.xlabel("Valores Reais")
    plt.ylabel("Erro Absoluto")
    plt.title("Erro Absoluto vs Valores Reais")
    plt.tight_layout()
    plt.show()

# ===============================
# 7. Função Principal
# ===============================
def main():
    data_file_path = "C:/Users/Pichau/OneDrive/Documentos/Projeto-Regressao-CreditData/statlog+german+credit+data/german.data"
    
    # Carregar e preparar dados
    X, y = load_and_prepare_data(data_file_path)
    
    # Remover outliers
    X, y = remove_outliers(X, y)
    
    # Balanceamento de dados com SMOTE (aumentando a classe minoritária)
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    print(f"\nNúmero de amostras após balanceamento: {len(y_res)}")
    
    # Divisão dos dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
    print("\nDivisão de dados completada!")
    
    # Normalização dos dados
    scaler = RobustScaler()  # Usa o RobustScaler para lidar com outliers
    X_train = scaler.fit_transform(X_train)  # Normalizando os dados de treino
    X_test = scaler.transform(X_test)  # Normalizando os dados de teste
    
    # Criando e treinando o modelo RandomForestClassifier
    model = RandomForestClassifier(random_state=42)
    
    # Otimização de hiperparâmetros
    best_params = optimize_hyperparameters(X_train, y_train, model)
    print("\nMelhores hiperparâmetros encontrados:", best_params)
    
    # Treinar e avaliar o modelo com os melhores parâmetros
    model.set_params(**best_params)
    precision, recall, f1, cm, y_pred = train_and_evaluate(X_train, X_test, y_train, y_test, model)
    
    # Exibir as métricas
    print(f"\nPrecisão: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Matriz de Confusão:\n{cm}")
    
    # Plotar os resíduos e erro absoluto
    plot_residuals(y_test, y_pred)
    
    # Realizar validação cruzada
    accuracy_cv, accuracy_cv_std = cross_validation(X_res, y_res, model)
    print("\nValidação Cruzada (5-fold):")
    print("Acurácia média:", accuracy_cv)
    print("Acurácia desvio padrão:", accuracy_cv_std)

if __name__ == "__main__":
    main()
