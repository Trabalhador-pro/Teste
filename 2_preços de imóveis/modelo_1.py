# %% 
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LinearRegression
from sklearn import metrics

from sklearn.model_selection import train_test_split


pd.set_option("display.max_columns", None)   # mostra todas as colunas
pd.set_option("display.max_rows", 200)       # mostra mais linhas
pd.set_option("display.width", 2000)         # aumenta largura
pd.set_option("display.max_colwidth", None)  # mostra texto completo
pd.set_option("display.max_info_columns", 200)  # melhora df.info()

# %%
# Importando arquivo de treino
df = pd.read_csv('data/train.csv')

df_teste_final = pd.read_csv('data/test.csv')

# Visualizando as 5 primeiras linhas
df.head()

# %%
# Análise descritiva rápida
df.describe()

# %%
# Olhando a quantidade de valores ausentes em cada variavel
df.isna().sum()

# %%
# Olhando o tipo de colunas e numeros faltantes 
df.info()

# %%
# Criando nossas variaveis preditivas
features = df.drop(columns=['SalePrice', 'Id'])

# Criando nosso alvo em log 
target = df['SalePrice']

# Separando o teste e treino
X_train, X_test, y_train, y_test = train_test_split(features, target,
                                                    test_size=0.2,
                                                    random_state=42)

# %%
# Separando automaticamente as colunas numéricas e categóricas
# para aplicar o tratamento correto em cada tipo de dado
colunas_numericas = features.select_dtypes(include=['float64', 'int64']).columns.to_list()
colunas_categoricas = features.select_dtypes(include=['object']).columns.to_list()

# Trata valores ausentes nas colunas numéricas usando a mediana
Pipeline_numerico = Pipeline([
    ('imput', SimpleImputer(strategy='median')) 
])

# Trata valores ausentes nas colunas categorias usando o valor mais frequente 'moda'
Pipeline_categorico = Pipeline([
    ('imput', SimpleImputer(strategy='most_frequent', fill_value='0')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Criando o pepiline para o tratamento final 
limpeza = ColumnTransformer([
    ('num', Pipeline_numerico, colunas_numericas),
    ('cat', Pipeline_categorico, colunas_categoricas)
])

# Criando o Pipeline final com o modelo
modelo_pipeline = Pipeline([
    ('limpeza', limpeza),
    ('modelo', LinearRegression())
])

# %%
# Treinando o modelo
modelo_pipeline.fit(X_train, y_train)

# Fazendo as previsoes do treino e teste
y_pred_train = modelo_pipeline.predict(X_train)
y_pred_test = modelo_pipeline.predict(X_test)

# Avaliando o treino
mae_train = metrics.mean_absolute_error(y_train, y_pred_train)
r2_train = metrics.r2_score(y_train, y_pred_train)

# Avalidando o teste
mae_test = metrics.mean_absolute_error(y_test, y_pred_test)
r2_test = metrics.r2_score(y_test, y_pred_test)

print(f'Erro absoluto médio, treino: {mae_train:.2f}')
print(f'coeficiente de deterincao, treino: {r2_train:.2f}')


print(f'Erro absoluto médio, teste: {mae_test:.2f}')
print(f'coeficiente de deterincao, teste: {r2_test:.2f}')


# %%

# Salvando os id para produzi depois o resultado
idfinal = df_teste_final['Id']

# Separando nossas colunas que tiramos em cima 
features = df_teste_final.drop(columns=['Id'])

# Fazendo a previsao final do teste
previsao_final = modelo_pipeline.predict(features)

# %%
# Criando nosso data frame
resultado = pd.DataFrame({
    'Id': idfinal,
    'SalePrice': previsao_final
})

resultado.head()
# %%
# Criando um arquivo
resultado.to_csv("submission.csv", index=False)

# %%
