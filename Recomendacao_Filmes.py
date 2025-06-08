#!/usr/bin/env python
# coding: utf-8

# ## Importação de Bibliotecas
# ---

import numpy as np
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# # Primeiro Dataset
# 
# Nesta primeira parte vamos tratar do nosso primeiro DataSet que é o de `filmes`, contendo cerca de 5K de filmes.
# 
# ---
# ## Visuliazação da Base de Dados de 5000 Filmes do IMDB
# ---

filmes = pd.read_csv('tmdb_5000_movies.csv')
#filmes.head()


# ## Análise Exploratória dos Dados
# ---
# 
# Nesta etapa vamos analisar como os dados estão distribuidos, quais features serão significativas, tratar os valores que forem necessários.

#filmes.info()


# ### Exclusão de **Features Desnecessárias** para nosso algoritmo
# ---
# 
# Como a **Recomendação** ira levar em consideração o conteúdo do filme, informações como *língua, duração e etc* não será util.

filmes.drop(columns = ['budget', 'homepage', 'id', 'original_title', 'original_language',
                       'popularity', 'production_countries', 'release_date',
                       'revenue', 'runtime', 'spoken_languages',
                       'status', 'vote_average', 'vote_count'],
                        axis=1, inplace=True)


# ### Visualizção de Valores `NaN` e Únicos
# ---
faltantes = pd.DataFrame({'colunas': filmes.columns, 
                      'tipo': filmes.dtypes,
                      'Qtde valores NaN': filmes.isna().sum(),
                      '% valores NaN': (filmes.isna().sum()/filmes.shape[0]*100).round(2),
                      'valores únicos por feature': filmes.nunique(),
                       'valores duplicados': filmes.duplicated().sum()  
                         })
faltantes = faltantes.reset_index()
#faltantes

# #### Preenchimento da **Feature** `tagline`
# ---
# 
# Após a análise é possível notar que temos muitos dados `Nan` na coluna `tagline`, **não podemos dropar pois iremos perder muitos filmes *(844)* cerca de 17% da nossa base**, então vamos preenche - los com um espaço vazio aqueles que são `Nan` e mais pra frente no projeto isso não fara diferença.

filmes.tagline = filmes.tagline.fillna(' ')

# #### Preenchimento da **Feature** `overview`
# ---
# 
# A **Feature** `overview`, possui apenas 3 valores `NaN` mas como estamos tentando preservar ao máximo nossa base iremos fazer a mesma coisa que fizemos com a `tagline` e no decorrer do desenvolvimento irá ficar claro o porque.


filmes.overview = filmes.overview.fillna(' ')
#filmes.info()

# ### Função para extrair infos das `features dentro de uma lista de dicionários`
# 
# ----
# 
# Podemos notar que algumas de nossas **Features** estão dentro de uma lista de dicionários e os elementos que queremos está dentro da chave `'name'`

#filmes.head()


# E os elementos que queremos está dentro da chave `'name'`, como pode - se ver nos prints abaixo.

#print(filmes['keywords'].iloc[0])
#print('-------------------------')
#print(filmes['genres'].iloc[0])
#print('-------------------------')
#print(filmes['production_companies'].iloc[0])


# #### **Função** `extrair_chave_lista`
# 
# ---
# Função para percorrer a lista de dicionários e retornar apenas os elementos da chave `name`, em uma lista, separados por `,`
# 
# ---

def extrair_chave_lista(x, chave='name'):
    """
    Extrai os valores de uma chave de uma lista de dicionários armazenada como string.

    Retorna:
    - Uma lista com os valores, ou None se vazio ou erro.
    """
    try:
        lista = ast.literal_eval(x)
        if isinstance(lista, list) and len(lista) > 0:
            valores = [item.get(chave) for item in lista if chave in item]
            return valores 
        else:
            return []
    except:
        return []

filmes['genres'] = filmes['genres'].apply(extrair_chave_lista)

filmes['keywords'] = filmes['keywords'].apply(extrair_chave_lista)

filmes['production_companies'] = filmes['production_companies'].apply(extrair_chave_lista)

#filmes.head()

# # Segundo Dataset
# 
# ---
# ## Visualização da Base de Dados `credits`, nela contém as informações sobre elenco e produção dos filmes.

credits = pd.read_csv('tmdb_5000_credits.csv')

#credits.head()


# ## Análise Exploratória dos Dados
# ---
# 
# Nesta etapa vamos analisar como os dados estão distribuidos, quais features serão significativas, tratar os valores que forem necessários.

#credits.info()

# ### Exclusão de **Features Desnecessárias** para nosso algoritmo
# ---
# 
# Como a **Recomendação** ira levar em consideração o conteúdo do filme, não será preciso ter a coluna `movie_id`

credits.drop('movie_id', axis=1, inplace=True)

# ### Função para extrair infos das `features dentro de uma lista de dicionários`
# 
# ----
# 
# Como já sabemos que algumas informações importantes para nosso algoritmo esta dentro de uma **lista de dicionários**, vamos printar um exemplo e ver o que será util.

#print(credits['cast'].iloc[0])
#print('-------------------------')
#print(credits['crew'].iloc[0])
#print('-------------------------')

# #### Função `extrair_chave_lista_n`
# ---
# Após análise é possível notar que a chave `name` da **Feature** contém os nomes dos atores qua atuaram, mas para não adicionarmos muita informção que não seja relevante vamos limitar a função para retornar apenas os 3 primeiros atores, que serão os principais de cada filme. 

def extrair_chave_lista_n(x, chave='name', n=3):
    """
    Extrai até n valores de uma chave de uma lista de dicionários armazenada como string.

    Retorna:
    - Uma lista com os valores, ou None se vazio ou erro.
    """
    try:
        lista = ast.literal_eval(x)
        if isinstance(lista, list) and len(lista) > 0:
            valores = [item[chave] for item in lista[:n] if chave in item]
            return valores 
        else:
            return []
    except:
        return []

credits['cast'] = credits['cast'].apply(extrair_chave_lista_n)

# #### Função `extrair_nome_por_job`
# ---
# Após examinar a **Feature** `crew`, é notável que ela armazena informações sobre a equipe responsável pelo desenvolvimento do filme, como Diretor, Editor e etc. Notamos que a chave responsávelpor armazenar estas funções é a `job` e assim vamos utilizar isso na nossa função para retornar o Diretor do filme.

def extrair_nome_por_job(x, job='Director'):
    """
    Extrai os nomes das pessoas com o job especificado de uma lista de dicionários armazenada como string.

    Retorna:
    - Uma lista com os nomes encontrados, ou None se não houver correspondências.
    """
    try:
        lista = ast.literal_eval(x)
        if isinstance(lista, list) and len(lista) > 0:
            nomes = [item['name'] for item in lista if item.get('job') == job]
            return nomes 
        else:
            return []
    except:
        return []


credits['director'] = credits['crew'].apply(extrair_nome_por_job)

credits.drop('crew',axis=1 ,inplace=True)

#credits

# # Terceiro Dataset
# 
# ---
# Agora que temos os dois datasets anteriores bem tratados podemos junta - los e criar um só armazenando tudo que precisamos `filmes2`.

filmes2 = filmes.merge(credits,on='title')

#filmes2.head()

# ## Função `transformar_em_lista`
# 
# ---
# 
# Após a união das tabelas foi possível notas que algumas colunas contém dados dentro de uma lista que foram aquelas colunas que trabalhamos acima, entretanto algumas colunas que contém informações relevantes como `tagline`, não está dentro de uma lista, isso dificulta nosso trabalho posteriormente para criarmos nossa principal **feature**, que é uma **sopa de palavras**. Vamos deixar todas colunas necessárias então dentro de uma lista.

def transformar_em_lista(valor):
    if isinstance(valor, list):
        return valor
    elif pd.isna(valor):  # trata NaN, None, etc.
        return []
    else:
        return [valor]

colunas = ['overview', 'tagline']

for col in colunas:
    filmes2[col] = filmes2[col].apply(transformar_em_lista)

#filmes2.head()

# ## Criação da Feature `soup_of_words`
# 
# ---
# 
# O que é a sopa de palavras?
# A sopa de palavras é uma técnica usada em sistemas de recomendação baseados em conteúdo, onde combinamos diversas informações relevantes sobre um item (como um filme) em uma única string textual. Isso facilita quando precisamos passar por uma vetorização textual, que chegaremos em breve.

filmes2['soup_of_words'] = filmes2['genres'] + filmes2['keywords'] + filmes2['overview'] + filmes2['production_companies'] + filmes2['tagline'] + filmes2['cast'] + filmes2['director']

# # Quarto Dataset
# 
# ---
# 
# Agora chegamos no nosso quarto e último DataSet após um longo tratamento dos dados podemos separar apenas o que será necessário, `title` para saber o nome do filme e `soup_of_words` feature que será utilizada para fazer as recomendações.

filmes_final = filmes2[['title', 'soup_of_words']]

#filmes_final

# ## Retirar as informações de dentro da lista
# 
# ---
# 
# Como usamos as listas afim apenas de faciltar na organização das Features e evitar erros ao unir todas Features em uma só, agora podemos aplicar uma função lambda para remover as listas.

filmes_final['soup_of_words'] = filmes_final['soup_of_words'].apply(lambda x:" ".join(x))

#filmes_final['soup_of_words'][0]

# ## Deixando todas palavras em *Minusculo*
# 
# ---
# 
# Essa parte não é tão necessária mas afim de evitar erros por letras Maiusculas ou Minusculas, vamos deixar todas em um único padrão **Minusculas**.

filmes_final['soup_of_words'] = filmes_final['soup_of_words'].apply(lambda x:x.lower())

#filmes_final['soup_of_words'][0]

# ## Vetorização de Palavras
# 
#  ---
#  
# Aqui estamos transformando o conteúdo textual da coluna `'soup_of_words'` em uma representação numérica
# usando a técnica **TF-IDF (Term Frequency - Inverse Document Frequency).**
# 
# **O TfidfVectorizer realiza duas tarefas principais:**
# 
# 1. Remove as "stop words" (palavras muito comuns em inglês como "the", "and", "is", etc.) que não agregam valor semântico.
# 
# 2. Converte cada palavra em um peso numérico que representa sua importância em relação ao texto do filme e ao conjunto total de filmes.
# 
# O resultado é uma matriz onde cada linha representa um filme e cada coluna representa uma palavra.
# Os valores nessa matriz indicam a relevância de cada palavra para cada filme, permitindo comparar filmes por similaridade de conteúdo.

vectorizer = TfidfVectorizer(stop_words='english')

tfidf = vectorizer.fit_transform(filmes_final['soup_of_words'])

# ## Cáculo da Similaridade de Cosseno
# 
# ---
# 
# Após **transformar os textos em vetores com o TF-IDF**, usamos a função `cosine_similarity` para
# medir o quão semelhantes são os filmes entre si com base no conteúdo textual.
# 
# **A similaridade do cosseno compara dois vetores e retorna um valor entre 0 e 1:**
# - **1** indica máxima similaridade (mesma direção vetorial).
# - **0** indica nenhuma similaridade (vetores ortogonais).
# 
# O resultado é uma matriz simétrica onde cada célula (i, j) representa a similaridade
# entre o filme i e o filme j.

sim = cosine_similarity(tfidf)
#sim

# ## Função de Recomendação `recomendar_filmes`
# 
# ---
# 
# Esta função recebe o nome de um filme e retorna os 10 filmes mais similares, com base na **matriz de similaridade de cosseno**. O resultado é **retornado em formato de DataFrame, incluindo o título do filme e seu grau de similaridade.**

def recomendar_filmes(filme, df, matriz_similaridade):
    idx = df[df['title'] == filme].index[0]
    similaridades = list(enumerate(matriz_similaridade[idx]))
    similares_ordenados = sorted(similaridades, key=lambda x: x[1], reverse=True)[1:11]
    resultados = [(df.iloc[i].title, score) for i, score in similares_ordenados]
    
    return pd.DataFrame(resultados, columns=['Filme Recomendado', 'Similaridade'])

# Exemplo de uso:
# recomendar_filmes("The Matrix", filmes_final, similarity)

recomendar_filmes('Thor', filmes_final, sim)
print(recomendar_filmes('Thor', filmes_final, sim))




