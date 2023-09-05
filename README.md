# ecg.ebac
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Carregar os dados do arquivo CSV em um DataFrame
file_path = 'ECGCvdata (2).csv'
df = pd.read_csv(file_path)

# Visualizar as primeiras linhas do DataFrame
print(df.head())

# Informações gerais sobre o DataFrame
print(df.info())

# Estatísticas descritivas dos dados
print(df.describe())

# Verificar a presença de valores nulos
print(df.isnull().sum())

# Gráfico de barras para contar a quantidade de valores únicos em cada coluna categórica
for col in df.select_dtypes(include='object').columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=col, order=df[col].value_counts().index)
    plt.title(f'Contagem de Valores Únicos em {col}', fontsize=16)
    plt.xlabel(col, fontsize=14)
    plt.ylabel('Contagem', fontsize=14)
    plt.xticks(rotation=45)
    plt.show()

# Gráfico de histograma para variáveis numéricas
for col in df.select_dtypes(include=['int64', 'float64']).columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=col, kde=True)
    plt.title(f'Distribuição de {col}', fontsize=16)
    plt.xlabel(col, fontsize=14)
    plt.ylabel('Contagem', fontsize=14)
    plt.show()

# Boxplot para visualizar a distribuição e detectar outliers em variáveis numéricas
for col in df.select_dtypes(include=['int64', 'float64']).columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, y=col)
    plt.title(f'Boxplot de {col}', fontsize=16)
    plt.ylabel(col, fontsize=14)
    plt.show()

# Gráfico de dispersão para analisar possíveis relações entre variáveis numéricas
sns.pairplot(df.select_dtypes(include=['int64', 'float64']))
plt.suptitle('Gráfico de Dispersão das Variáveis Numéricas', fontsize=16, y=1.02)
plt.show()

# Gráfico de correlação entre variáveis numéricas
correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Matriz de Correlação', fontsize=16)
plt.show()

# Construir um modelo de regressão linear simples (exemplo)
X = df[['feature1', 'feature2']]  # Substitua 'feature1' e 'feature2' pelas suas variáveis
y = df['target']  # Substitua 'target' pela sua variável alvo

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Avaliação do modelo
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Apresentação de visualização de dados adicionais
# Adicione aqui suas visualizações adicionais ou outras análises relevantes

# Exemplo de visualização: Gráfico de dispersão dos resultados da regressão
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.title('Gráfico de Dispersão dos Resultados da Regressão', fontsize=16)
plt.xlabel('Valor Real', fontsize=14)
plt.ylabel('Valor Previsto', fontsize=14)
plt.show()
