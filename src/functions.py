## Bibliotecas Gerais 
import sys
sys.executable
import re

## Bibliotecas de Análise de Dados
import pandas as pd 
import geopandas as gpd
import builtins as builtins
import matplotlib.pyplot as plt
import seaborn as sns 
from IPython.display import display, Image
from tabulate import tabulate
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from matplotlib.gridspec import GridSpec

# Bibliotecas de Manipulação de Tempo
from datetime import datetime, date, timedelta

## Bibliotecas de Modelagem Matemática e Estatística
import numpy as np
import scipy as sp 
import scipy.stats as stats
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import normaltest, ttest_ind, ttest_rel, mannwhitneyu, wilcoxon, kruskal, uniform, chi2_contingency
from statsmodels.stats.weightstats import ztest
from numpy import interp
import random


# Bibliotecas de Pré-Processamento e Pipeline
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate, cross_val_predict
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from category_encoders import CatBoostEncoder

# Bibliotecas de Modelos de Machine Learning
import joblib
from joblib import Parallel, delayed
import pickle
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import lightgbm as lgb
from lightgbm import LGBMRegressor, LGBMClassifier, early_stopping
from sklearn.cluster import KMeans
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import skpro 
import mapie
from skpro.regression.residual import ResidualDouble
from mapie.metrics.regression import regression_coverage_score
from mapie.regression import SplitConformalRegressor
from mapie.utils import train_conformalize_test_split
import networkx as nx
import shap

# Bibliotecas de Métricas de Machine Learning
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error, mean_absolute_percentage_error, accuracy_score, roc_auc_score, roc_curve, auc, precision_score, recall_score, precision_recall_curve, average_precision_score, f1_score, log_loss, brier_score_loss, confusion_matrix, cohen_kappa_score, silhouette_score

# Bibliotecas de Spark  

# # Spark Session
# spark = SparkSession.builder.getOrCreate()

def plota_barras(variaveis, df, titulo_base='Distribuição', rotation=0,
                 figsize=(8, 5), top_n=None, limites=None, usar_subplot=False):

    # --- Normalização de entrada ---
    if isinstance(variaveis, str):
        variaveis = [variaveis]
    if limites is None:
        limites = {}

    # ============================================================
    # FUNÇÃO AUXILIAR: ordena apenas se valores forem numéricos
    # ============================================================
    def ordenar_counts_se_numerico(counts):
        try:
            # tenta converter todos os labels para número
            pd.to_numeric(counts.index, errors='raise')
            return counts.sort_index()      # ordena se der certo
        except:
            return counts                   # mantém a ordem original

    # ============================================================
    # CASO 1: Uma variável ou subplot desativado
    # ============================================================
    if len(variaveis) == 1 or not usar_subplot:
        for var in variaveis:

            limite_var = limites.get(var, top_n)
            counts = df[var].value_counts()

            # Ordenação segura
            counts = ordenar_counts_se_numerico(counts)

            if limite_var is not None:
                counts = counts.head(limite_var)

            order = counts.index
            values = counts.values
            total = values.sum()

            plt.figure(figsize=figsize)
            ax = sns.barplot(x=order, y=values, color='#1FB3E5')
            ax.set_title(f'{titulo_base} — {var}', fontsize=14)
            ax.set_xlabel(var, fontsize=12)
            ax.set_ylabel('Quantidade', fontsize=12)

            # Percentuais acima das barras
            for i, v in enumerate(values):
                ax.text(i, v + (max(values) * 0.01), f'{(v/total)*100:.2f}%',
                        ha='center', va='bottom', fontsize=10)

            ax.set_ylim(0, max(values) * 1.15)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation, ha='right', fontsize=10)
            ax.set_yticklabels(['{:,.0f}'.format(y) for y in ax.get_yticks()], fontsize=10)

            plt.tight_layout()
            plt.show()
        return

    # ============================================================
    # CASO 2: Múltiplas variáveis com subplot
    # ============================================================
    n_vars = len(variaveis)
    n_cols = 2
    n_rows = (n_vars + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(figsize[0]*n_cols, figsize[1]*n_rows))
    axes = axes.flatten()

    for i, var in enumerate(variaveis):

        limite_var = limites.get(var, top_n)
        counts = df[var].value_counts()

        # Ordenação segura
        counts = ordenar_counts_se_numerico(counts)

        if limite_var is not None:
            counts = counts.head(limite_var)

        order = counts.index
        values = counts.values
        total = values.sum()

        ax = axes[i]
        sns.barplot(x=order, y=values, color='#1FB3E5', ax=ax)
        ax.set_title(f'{var}', fontsize=12)
        ax.set_xlabel('')
        ax.set_ylabel('Qtd')

        for j, v in enumerate(values):
            ax.text(j, v + (max(values) * 0.01), f'{(v/total)*100:.1f}%',
                    ha='center', va='bottom', fontsize=9)

        ax.set_ylim(0, max(values) * 1.15)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation, ha='right', fontsize=9)

    # Remove subplots vazios
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(titulo_base, fontsize=16)
    plt.tight_layout()
    plt.show()


def plota_histograma(lista_variaveis, df, linhas, colunas, titulo):
    if (linhas == 1) and (colunas == 1): 
        k = 0
        mediana = df[lista_variaveis[k]].median()
        media = df[lista_variaveis[k]].mean()
        plt.figure(figsize = (14, 5))
        ax = sns.histplot(x = lista_variaveis[k], data = df, color = '#1FB3E5', bins = 30)
        ax.set_title(f'{titulo}')
        ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 14)
        ax.set_ylabel(f'Frequência', fontsize = 14)
        ax.axvline(x = mediana, ymax = 0.75 ,color = '#231F20', linestyle = '-', label = f'mediana = {mediana}')
        ax.axvline(x = media, ymax = 0.75,color = '#231F20', linestyle = '--', label = f'media = {media}')
        plt.ticklabel_format(style='plain')
        plt.legend(loc = 'best')
        plt.show()
    elif linhas == 1:
        fig, axis = plt.subplots(linhas, colunas, figsize = (14, 5), sharey = True)
        fig.suptitle(f'{titulo}')
        k = 0
        for i in np.arange(linhas):
            for j in np.arange(colunas):
                mediana = df[lista_variaveis[k]].median()
                media = df[lista_variaveis[k]].mean().round()
                ax = sns.histplot(x = lista_variaveis[k], data = df, color = '#1FB3E5', ax = axis[j], bins = 30)
                ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 14)
                ax.set_ylabel(f'Frequência', fontsize = 14)
                ax.axvline(x = mediana, ymax = 0.75 ,color = '#231F20', linestyle = '-', label = f'mediana = {mediana}')
                ax.axvline(x = media, ymax = 0.75,color = '#231F20', linestyle = '--', label = f'media = {media}')
                ax.ticklabel_format(style='plain')
                ax.legend(loc = 'best')
                k = k + 1
    elif colunas == 1:
        fig, axis = plt.subplots(linhas, colunas, figsize = (14, 5), sharey = True)
        fig.suptitle(f'{titulo}')
        k = 0
        for i in np.arange(linhas):
            for j in np.arange(colunas):
                mediana = df[lista_variaveis[k]].median()
                media = df[lista_variaveis[k]].mean().round()
                ax = sns.histplot(x = lista_variaveis[k], data = df, color = '#1FB3E5', ax = axis[i], bins = 30)
                ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 14)
                ax.set_ylabel(f'Frequência', fontsize = 14)
                ax.axvline(x = mediana, ymax = 0.75 ,color = '#231F20', linestyle = '-', label = f'mediana = {mediana}')
                ax.axvline(x = media, ymax = 0.75,color = '#231F20', linestyle = '--', label = f'media = {media}')
                ax.ticklabel_format(style='plain')
                ax.legend(loc = 'best')
                k = k + 1
    else:
        fig, axis = plt.subplots(linhas, colunas, figsize = (14, 5), sharey = True)
        fig.suptitle(f'{titulo}')
        k = 0
        for i in np.arange(linhas):
            for j in np.arange(colunas):
                mediana = df[lista_variaveis[k]].median()
                media = df[lista_variaveis[k]].mean().round()
                ax = sns.histplot(x = lista_variaveis[k], data = df, color = '#1FB3E5', ax = axis[i, j], bins = 30)
                ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 14)
                ax.set_ylabel(f'Frequência', fontsize = 14)
                ax.axvline(x = mediana, ymax = 0.75 ,color = '#231F20', linestyle = '-', label = f'mediana = {mediana}')
                ax.axvline(x = media, ymax = 0.75,color = '#231F20', linestyle = '--', label = f'media = {media}')
                ax.ticklabel_format(style='plain')
                ax.legend(loc = 'best')
                k = k + 1



def plota_boxplot(df,variaveis,categorias=None,titulo_base='Boxplot',rotation=0,figsize=(8, 5),usar_subplot=False,modo='bivariado'):

    if isinstance(variaveis, str):
        variaveis = [variaveis]

    # --- Caso simples (um gráfico por variável) ---
    if len(variaveis) == 1 or not usar_subplot:
        for var in variaveis:
            plt.figure(figsize=figsize)

            if modo == 'bivariado':
                if categorias is None:
                    raise ValueError("No modo 'bivariado', o parâmetro 'categorias' é obrigatório.")
                sns.boxplot(x=categorias, y=var, data=df, palette=['green', 'yellow', 'red'])
                plt.xlabel(categorias)
            else:  # univariado
                sns.boxplot(y=var, data=df, color='#1FB3E5')

            plt.title(f'{titulo_base} — {var}', fontsize=12)
            plt.ylabel(var)
            plt.xticks(rotation=rotation)
            plt.tight_layout()
            plt.show()
        return

    # --- Caso com subplots (várias variáveis) ---
    n_vars = len(variaveis)
    n_cols = 2
    n_rows = (n_vars + 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0]*n_cols, figsize[1]*n_rows))
    axes = axes.flatten()

    for i, var in enumerate(variaveis):
        ax = axes[i]

        if modo == 'bivariado':
            if categorias is None:
                raise ValueError("No modo 'bivariado', o parâmetro 'categorias' é obrigatório.")
            sns.boxplot(x=categorias, y=var, data=df, ax=ax, palette=['green', 'yellow', 'red'])
            ax.set_xlabel(categorias)
        else:
            sns.boxplot(y=var, data=df, ax=ax, color='#1FB3E5')

        ax.set_title(f'{var}', fontsize=11)
        ax.set_ylabel(var)
        ax.tick_params(axis='x', rotation=rotation)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(titulo_base, fontsize=14)
    plt.tight_layout()
    plt.show()



def plota_grafico_linhas(df, x, y, nao_calcula_media, title):

    if nao_calcula_media:
        # Criando o gráfico de linha
        plt.figure(figsize=(20, 8))
        plt.plot(df[x], df[y], marker='o', linestyle='-', color='#1FB3E5')

        # Adicionando títulos e rótulos aos eixos
        plt.title(title)
        plt.xlabel(x)
        plt.ylabel(y)

        for i, txt in enumerate(df[y]):
            plt.annotate(f'{txt:.1f}', (df[x][i], df[y][i]), textcoords="offset points", xytext=(0,1), ha='center')

        # Exibindo o gráfico
        plt.grid(True)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()
    else:
        media = df[y].mean()
        # Criando o gráfico de linha
        plt.figure(figsize=(20, 8))
        plt.plot(df[x], df[y], marker='o', linestyle='-', color='#1FB3E5')

        # Adicionando linha da média
        plt.axhline(y=media, color='r', linestyle='--', linewidth=1, label=f'Média: {media:.2f}')
        plt.legend()

        # Adicionando títulos e rótulos aos eixos
        plt.title(title)
        plt.xlabel(x)
        plt.ylabel(y)

        for i, txt in enumerate(df[y]):
            plt.annotate(f'{txt:.1f}', (df[x][i], df[y][i]), textcoords="offset points", xytext=(0,1), ha='center')

        # Exibindo o gráfico
        plt.grid(True)
        plt.xticks(rotation=90)
        #plt.ylim(0, 50)
        plt.tight_layout()
        plt.show()

def analisa_correlacao(metodo, df):
    plt.figure(figsize=(30, 15))
    mask = np.triu(np.ones_like(df.corr(method=metodo), dtype=bool))
    heatmap = sns.heatmap(df.corr(method=metodo), vmin=-1, vmax=1, cmap='magma', annot=True, fmt='.1f', cbar_kws={"shrink": .8}, mask=mask)
    heatmap.set_title(f"Analisando Correlação de {metodo}")
    plt.grid(False)
    plt.box(False)
    plt.tight_layout()
    plt.grid(False)
    plt.show()

def calcular_psi_temporal(df, coluna_data='data_pedido', coluna_metricas='tempo_entrega', 
                          nome_metrica='Tempo de Entrega', data_base_inicio=None, 
                          data_base_fim=None, data_teste_inicio=None, data_teste_fim=None,
                          tipo_analise='mensal'):
    """
    Função para cálculo e plotagem de PSI temporal com flexibilidade de períodos
    
    Parameters:
    df: DataFrame - DataFrame com os dados
    coluna_data: str - Nome da coluna de data
    coluna_metricas: str - Nome da coluna com a métrica a analisar
    nome_metrica: str - Nome amigável da métrica para exibição
    data_base_inicio: str/datetime - Data de início do período base
    data_base_fim: str/datetime - Data de fim do período base
    data_teste_inicio: str/datetime - Data de início do período de teste
    data_teste_fim: str/datetime - Data de fim do período de teste
    tipo_analise: str - 'mensal' ou 'diária' (para formatação dos labels)
    
    Returns:
    psi_value: float - Valor do PSI calculado
    psi_df: DataFrame - DataFrame com cálculos detalhados
    """
    
    # ============================================================================
    # ETAPA 1: PREPARAÇÃO DOS DADOS TEMPORAIS
    # ============================================================================
    
    # Converter coluna de data para datetime
    df[coluna_data] = pd.to_datetime(df[coluna_data])
    df = df.sort_values(coluna_data)
    
    # Definir datas base e teste
    if data_base_inicio is None:
        data_base_inicio = df[coluna_data].min()
    else:
        data_base_inicio = pd.to_datetime(data_base_inicio)
    
    if data_base_fim is None:
        if tipo_analise == 'mensal':
            data_base_fim = data_base_inicio + pd.DateOffset(months=1)
        else:
            data_base_fim = data_base_inicio + pd.DateOffset(days=1)
    else:
        data_base_fim = pd.to_datetime(data_base_fim)
    
    if data_teste_inicio is None:
        if data_teste_fim is None:
            data_teste_inicio = df[coluna_data].max() - pd.DateOffset(months=1)
        else:
            data_teste_inicio = data_teste_fim - pd.DateOffset(months=1)
    else:
        data_teste_inicio = pd.to_datetime(data_teste_inicio)
    
    if data_teste_fim is None:
        data_teste_fim = df[coluna_data].max()
    else:
        data_teste_fim = pd.to_datetime(data_teste_fim)
    
    # Filtrar dados
    mascara_base = (df[coluna_data] >= data_base_inicio) & (df[coluna_data] < data_base_fim)
    mascara_teste = (df[coluna_data] >= data_teste_inicio) & (df[coluna_data] < data_teste_fim)
    
    dados_base = df[mascara_base].copy()
    dados_teste = df[mascara_teste].copy()
    
    # Verificar se há dados suficientes
    if len(dados_base) == 0:
        raise ValueError(f"Nenhum dado encontrado para o período base: {data_base_inicio.date()} a {data_base_fim.date()}")
    if len(dados_teste) == 0:
        raise ValueError(f"Nenhum dado encontrado para o período de teste: {data_teste_inicio.date()} a {data_teste_fim.date()}")
    
    print("📊 ANÁLISE PSI TEMPORAL")
    print(f"Período base: {data_base_inicio.date()} a {data_base_fim.date()}")
    print(f"Período teste: {data_teste_inicio.date()} a {data_teste_fim.date()}")
    print(f"Registros base: {len(dados_base):,} | Registros teste: {len(dados_teste):,}")
    
    # ============================================================================
    # ETAPA 2: CÁLCULO DO PSI
    # ============================================================================
    
    def calcular_psi(distribuicao_base, distribuicao_teste, num_buckets=10):
        """Calcula o Population Stability Index entre duas distribuições"""
        
        # Remover valores nulos
        base_limpa = np.array(distribuicao_base[~pd.isnull(distribuicao_base)])
        teste_limpa = np.array(distribuicao_teste[~pd.isnull(distribuicao_teste)])
        
        # Definir pontos de corte pelos percentis da distribuição base
        pontos_corte = np.percentile(base_limpa, [i * 100/num_buckets for i in range(num_buckets + 1)])
        pontos_corte = np.unique(pontos_corte)
        
        # Calcular frequências em cada bucket
        freq_base = np.histogram(base_limpa, pontos_corte)[0]
        freq_teste = np.histogram(teste_limpa, pontos_corte)[0]
        
        # Adicionar valor pequeno para evitar divisão por zero
        freq_base = freq_base + 0.0001
        freq_teste = freq_teste + 0.0001
        
        # Calcular proporções
        prop_base = freq_base / len(base_limpa)
        prop_teste = freq_teste / len(teste_limpa)
        
        # Calcular componentes do PSI para cada bucket
        componentes_psi = (prop_teste - prop_base) * np.log(prop_teste / prop_base)
        psi_total = np.sum(componentes_psi)
        
        # Criar DataFrame com resultados detalhados
        psi_detalhado = pd.DataFrame({
            'decil': range(1, len(pontos_corte)),
            'frequencia_base': freq_base,
            'frequencia_teste': freq_teste,
            'proporcao_base': prop_base,
            'proporcao_teste': prop_teste,
            'componente_psi': componentes_psi,
            'limite_inferior': pontos_corte[:-1],
            'limite_superior': pontos_corte[1:]
        })
        
        return psi_total, psi_detalhado
    
    # Calcular PSI
    valor_psi, df_psi = calcular_psi(dados_base[coluna_metricas], dados_teste[coluna_metricas])
    
    # ============================================================================
    # ETAPA 3: PLOTAGEM DOS GRÁFICOS
    # ============================================================================
    
    # Criar figura com dois subplots
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 1, height_ratios=[1, 2])
    
    # --- GRÁFICO SUPERIOR: EVOLUÇÃO DO PSI ---
    ax_superior = plt.subplot(gs[0])
    
    # Plotar linha do PSI acumulado
    ax_superior.plot(df_psi['decil'], df_psi['componente_psi'].cumsum(), 
                     marker='o', linewidth=2, markersize=8, color='blue', 
                     label='PSI Acumulado')
    
    # Linha do valor total do PSI
    ax_superior.axhline(y=valor_psi, color='red', linestyle='--', linewidth=2, 
                        label=f'PSI Total: {valor_psi:.4f}')
    
    # Áreas coloridas para interpretação
    ax_superior.axhspan(0, 0.1, alpha=0.3, color='green', label='PSI ≤ 0.1 (Estável)')
    ax_superior.axhspan(0.1, 0.25, alpha=0.3, color='yellow', label='0.1 < PSI ≤ 0.25 (Atenção)')
    ax_superior.axhspan(0.25, max(valor_psi, 0.5), alpha=0.3, color='red', label='PSI > 0.25 (Instável)')
    
    # Formatar título baseado no tipo de análise
    titulo_temporal = 'Mensal' if tipo_analise == 'mensal' else 'Diária'
    ax_superior.set_title(f'Análise de Estabilidade {titulo_temporal} - {nome_metrica}', 
                          fontsize=14, fontweight='bold', pad=20)
    ax_superior.set_ylabel('Valor do PSI', fontsize=12)
    ax_superior.set_xlabel('Decis', fontsize=12)
    ax_superior.set_xticks(df_psi['decil'])
    ax_superior.set_xticklabels([f'D{i}' for i in df_psi['decil']])
    ax_superior.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax_superior.grid(True, alpha=0.3)
    
    # --- GRÁFICO INFERIOR: BARRAS EMPILHADAS POR SAFRA ---
    ax_inferior = plt.subplot(gs[1])
    
    # Preparar dados para o gráfico de barras empilhadas
    num_decis = len(df_psi)
    
    # Criar cores com gradiente de azul (mais escuro para decis maiores)
    cores_decis = plt.cm.Blues(np.linspace(0.3, 0.95, num_decis))
    
    # Formatar nomes das safras
    if tipo_analise == 'diária':
        nome_safra_base = f"Safra Base\n{data_base_inicio.strftime('%d/%m/%Y')}"
        nome_safra_teste = f"Safra Teste\n{data_teste_inicio.strftime('%d/%m/%Y')}"
    else:
        nome_safra_base = f"Safra Base\n{data_base_inicio.strftime('%b/%Y')}"
        nome_safra_teste = f"Safra Teste\n{data_teste_inicio.strftime('%b/%Y')}"
    
    # Posições das barras no eixo X (safras)
    safras = ['Base', 'Teste']
    posicoes_x = np.arange(len(safras))
    largura_barra = 0.6
    
    # Preparar dados empilhados
    # Cada safra tem 10 decis empilhados que somam 100%
    proporcoes_base = df_psi['proporcao_base'].values * 100  # Em porcentagem
    proporcoes_teste = df_psi['proporcao_teste'].values * 100  # Em porcentagem
    
    # Criar barras empilhadas
    acumulado_base = 0
    acumulado_teste = 0
    
    # Plotar cada decil como uma camada empilhada
    for i in range(num_decis-1, -1, -1):  # Do decil 10 ao 1 (para empilhar corretamente)
        decil_num = i + 1
        
        # Barra da safra base para este decil
        altura_base = proporcoes_base[i]
        barra_base = ax_inferior.bar(posicoes_x[0], altura_base, 
                                    width=largura_barra,
                                    bottom=acumulado_base,
                                    color=cores_decis[i],
                                    edgecolor='white',
                                    linewidth=0.5,
                                    alpha=0.9,
                                    label=f'Decil {decil_num}' if i == num_decis-1 else "")
        
        # Barra da safra teste para este decil
        altura_teste = proporcoes_teste[i]
        barra_teste = ax_inferior.bar(posicoes_x[1], altura_teste, 
                                     width=largura_barra,
                                     bottom=acumulado_teste,
                                     color=cores_decis[i],
                                     edgecolor='white',
                                     linewidth=0.5,
                                     alpha=0.9,
                                     hatch='//' if i == num_decis-1 else "//")
        
        # Adicionar texto dentro da barra se espaço suficiente
        if altura_base > 3:
            ax_inferior.text(posicoes_x[0], acumulado_base + altura_base/2, 
                           f'D{decil_num}',
                           ha='center', va='center',
                           fontsize=8, fontweight='bold',
                           color='white')
        
        if altura_teste > 3:
            ax_inferior.text(posicoes_x[1], acumulado_teste + altura_teste/2, 
                           f'D{decil_num}',
                           ha='center', va='center',
                           fontsize=8, fontweight='bold',
                           color='white')
        
        acumulado_base += altura_base
        acumulado_teste += altura_teste
    
    # Configurar eixo X
    ax_inferior.set_xlabel('Safras Analisadas', fontsize=12, fontweight='bold')
    ax_inferior.set_ylabel('Proporção da População (%)', fontsize=12)
    
    # Título do gráfico
    titulo_grafico = f'Distribuição por Decis - Comparação entre Safras'
    ax_inferior.set_title(titulo_grafico, fontsize=13, fontweight='bold', pad=15)
    
    # Configurar ticks do eixo X
    ax_inferior.set_xticks(posicoes_x)
    ax_inferior.set_xticklabels([nome_safra_base, nome_safra_teste], 
                               fontsize=11, fontweight='bold')
    
    # Adicionar linha horizontal em 100% para referência
    ax_inferior.axhline(y=100, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Adicionar valor total no topo de cada barra
    ax_inferior.text(posicoes_x[0], 102, f'100%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')
    
    ax_inferior.text(posicoes_x[1], 102, f'100%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')
    
    # Configurar limites do eixo Y
    ax_inferior.set_ylim(0, 110)
    
    # Adicionar grade apenas no eixo Y
    ax_inferior.grid(True, alpha=0.3, axis='y')
    
    # Adicionar legenda dos decis (representativa)
    from matplotlib.patches import Patch
    
    # Criar elementos de legenda para alguns decis representativos
    legend_elements = [
        Patch(facecolor=cores_decis[-1], edgecolor='white', label='Decil 10 (Mais alto)'),
        Patch(facecolor=cores_decis[num_decis//2], edgecolor='white', label=f'Decil {num_decis//2}'),
        Patch(facecolor=cores_decis[0], edgecolor='white', label='Decil 1 (Mais baixo)'),
        Patch(facecolor='white', edgecolor='black', hatch='//', alpha=0.9,
              label='Safra Teste (Hachurado)')
    ]
    
    # ax_inferior.legend(handles=legend_elements, loc='upper left', 
    #                   bbox_to_anchor=(1, 1), title="Legenda dos Decis")
    
    # ============================================================================
    # ETAPA 4: INFORMAÇÕES E INTERPRETAÇÃO
    # ============================================================================
    
    # Determinar interpretação do PSI
    if valor_psi <= 0.1:
        interpretacao = "ESTÁVEL ✅"
        cor_interpretacao = "green"
    elif valor_psi <= 0.25:
        interpretacao = "ATENÇÃO ⚠️"
        cor_interpretacao = "orange"
    else:
        interpretacao = "INSTÁVEL 🚨"
        cor_interpretacao = "red"
    
    # Texto com estatísticas detalhadas
    texto_estatisticas = f'''
    📊 INFORMAÇÕES DAS SAFRAS:
    
    SAFRA BASE:
    • Período: {data_base_inicio.strftime('%d/%m/%Y')} a {data_base_fim.strftime('%d/%m/%Y')}
    • Registros: {len(dados_base):,}
    • Média: {dados_base[coluna_metricas].mean():.2f}
    • Mediana: {dados_base[coluna_metricas].median():.2f}
    
    SAFRA TESTE:
    • Período: {data_teste_inicio.strftime('%d/%m/%Y')} a {data_teste_fim.strftime('%d/%m/%Y')}
    • Registros: {len(dados_teste):,}
    • Média: {dados_teste[coluna_metricas].mean():.2f}
    • Mediana: {dados_teste[coluna_metricas].median():.2f}
    
    📈 RESULTADO DO PSI:
    • Valor PSI: {valor_psi:.4f}
    • Interpretação: {interpretacao}
    • Métrica: {nome_metrica}
    '''
    
    ax_inferior.text(1.02, 0.98, texto_estatisticas, transform=ax_inferior.transAxes, 
                    fontsize=9, verticalalignment='top', color=cor_interpretacao,
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Print final no console
    print(f"\n🎯 RESULTADO FINAL: PSI = {valor_psi:.4f} - {interpretacao}")
    print("=" * 60)
    
    # Adicionar informação detalhada por decil
    print("\n📋 DETALHAMENTO POR DECIL:")
    print("=" * 60)
    print(f"{'Decil':<6} {'% Base':<10} {'% Teste':<10} {'Diferença':<12} {'PSI Comp.':<10}")
    print("-" * 60)
    
    for _, row in df_psi.iterrows():
        decil = int(row['decil'])
        perc_base = row['proporcao_base'] * 100
        perc_teste = row['proporcao_teste'] * 100
        diferenca = perc_teste - perc_base
        psi_comp = row['componente_psi']
        
        print(f"{f'D{decil}':<6} {perc_base:<10.2f} {perc_teste:<10.2f} {diferenca:<12.2f} {psi_comp:<10.4f}")
    
    return valor_psi, df_psi

# Configurar estilo dos gráficos
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def plot_histograms_comparison(df_train, df_valid, df_test, df_oot, column='tempo_entrega'):
    """
    Plota 4 histogramas lado a lado para comparar as distribuições
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Distribuição de {column} nos Conjuntos de Dados', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    datasets = [
        (df_train[column], 'Treino', '#1f77b4'),
        (df_valid[column], 'Validação', '#2ca02c'),
        (df_test[column], 'Teste', '#ff7f0e'),
        (df_oot[column], 'OOT (Out of Time)', '#d62728')
    ]
    
    for idx, (data, title, color) in enumerate(datasets):
        ax = axes[idx // 2, idx % 2]
        
        # Remover valores NaN
        data_clean = data.dropna()
        
        if len(data_clean) == 0:
            ax.text(0.5, 0.5, 'Sem dados', 
                   ha='center', va='center', fontsize=12)
            ax.set_title(f'{title} (n=0)', fontsize=14, fontweight='bold')
            continue
        
        # Calcular estatísticas
        mean_val = data_clean.mean()
        median_val = data_clean.median()
        std_val = data_clean.std()
        q1 = np.percentile(data_clean, 25)
        q3 = np.percentile(data_clean, 75)
        
        # Determinar número de bins (regra de Sturges)
        n_bins = min(50, int(1 + 3.322 * np.log10(len(data_clean))))
        
        # Plotar histograma
        n, bins, patches = ax.hist(data_clean, bins=n_bins, alpha=0.7, color=color, 
                                   density=True, edgecolor='black', linewidth=0.5)
        
        # Adicionar linhas de média e mediana
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                  label=f'Média: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2, 
                  label=f'Mediana: {median_val:.2f}')
        
        # Adicionar KDE (Kernel Density Estimation)
        from scipy.stats import gaussian_kde
        try:
            kde = gaussian_kde(data_clean)
            x_range = np.linspace(data_clean.min(), data_clean.max(), 1000)
            ax.plot(x_range, kde(x_range), color='black', linewidth=2, alpha=0.8, label='KDE')
        except:
            pass  # Ignorar se não conseguir calcular KDE
        
        # Configurações do gráfico
        ax.set_title(f'{title} (n={len(data_clean):,})', fontsize=14, fontweight='bold')
        ax.set_xlabel(column, fontsize=12)
        ax.set_ylabel('Densidade', fontsize=12)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Adicionar estatísticas no canto
        stats_text = (f'Média: {mean_val:.2f}\n'
                     f'Mediana: {median_val:.2f}\n'
                     f'Std: {std_val:.2f}\n'
                     f'Q1: {q1:.2f}\n'
                     f'Q3: {q3:.2f}\n'
                     f'IQR: {q3-q1:.2f}')
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.show()
    
    # Retornar estatísticas para uso posterior
    stats = {}
    for data, title, _ in datasets:
        data_clean = data.dropna()
        if len(data_clean) > 0:
            stats[title] = {
                'n': len(data_clean),
                'mean': data_clean.mean(),
                'median': data_clean.median(),
                'std': data_clean.std(),
                'min': data_clean.min(),
                'max': data_clean.max(),
                'q1': np.percentile(data_clean, 25),
                'q3': np.percentile(data_clean, 75)
            }
    
    return stats

def plot_boxplot_comparison(df_train, df_valid, df_test, df_oot, column='tempo_entrega'):
    """
    Plot boxplot comparativo dos 4 conjuntos
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    # Preparar dados para boxplot
    train_data = df_train[column].dropna()
    valid_data = df_valid[column].dropna()
    test_data = df_test[column].dropna()
    oot_data = df_oot[column].dropna()
    
    data_to_plot = [train_data, valid_data, test_data, oot_data]
    labels = ['Treino', 'Validação', 'Teste', 'OOT']
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
    
    # BOXPLOT PRINCIPAL
    box = ax1.boxplot(data_to_plot, labels=labels, patch_artist=True, 
                     showmeans=True, meanline=True, showfliers=False,
                     medianprops=dict(color='yellow', linewidth=2),
                     meanprops=dict(color='red', linewidth=2))
    
    # Colorir as caixas
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Adicionar estatísticas como anotações
    for i, data in enumerate(data_to_plot):
        if len(data) > 0:
            median = np.median(data)
            mean = np.mean(data)
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            
            # Adicionar texto com estatísticas
            ax1.text(i + 1, median, f'Med: {median:.1f}', 
                    ha='center', va='bottom', fontweight='bold', 
                    fontsize=10, color='yellow', 
                    bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))
            ax1.text(i + 1, mean, f'Média: {mean:.1f}', 
                    ha='center', va='top', fontsize=9, color='red',
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
    
    ax1.set_title(f'Comparação de {column} entre Conjuntos de Dados', 
                 fontsize=16, fontweight='bold')
    ax1.set_ylabel(column, fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # GRÁFICO DE BARRAS COM CONTAGEM E MÉDIA
    x_pos = np.arange(len(labels))
    counts = [len(d) for d in data_to_plot]
    means = [d.mean() if len(d) > 0 else 0 for d in data_to_plot]
    
    # Barras de contagem
    bars1 = ax2.bar(x_pos - 0.2, counts, width=0.4, label='Contagem', 
                   color=colors, alpha=0.7)
    ax2.set_ylabel('Nº Amostras', color='black', fontsize=11)
    ax2.tick_params(axis='y', labelcolor='black')
    
    # Eixo secundário para médias
    ax2_secondary = ax2.twinx()
    bars2 = ax2_secondary.bar(x_pos + 0.2, means, width=0.4, label='Média', 
                             color=['#8b0000', '#006400', '#8B4513', '#800080'], 
                             alpha=0.7)
    ax2_secondary.set_ylabel('Média', color='black', fontsize=11)
    ax2_secondary.tick_params(axis='y', labelcolor='black')
    
    # Adicionar valores nas barras
    for i, (bar, count) in enumerate(zip(bars1, counts)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                f'{count:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    for i, (bar, mean_val) in enumerate(zip(bars2, means)):
        ax2_secondary.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(means)*0.01,
                          f'{mean_val:.1f}', ha='center', va='bottom', 
                          fontsize=9, fontweight='bold', color='darkred')
    
    ax2.set_xlabel('Conjunto de Dados', fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, fontsize=11)
    ax2.set_title('Contagem de Amostras e Médias por Conjunto', fontsize=13, fontweight='bold')
    
    # Combinar legendas
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_secondary.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    # Imprimir tabela resumo
    print("\n" + "="*80)
    print("RESUMO ESTATÍSTICO - DISTRIBUIÇÃO DE TEMPO DE ENTREGA")
    print("="*80)
    
    summary_data = []
    for i, (data, label) in enumerate(zip(data_to_plot, labels)):
        if len(data) > 0:
            summary_data.append({
                'Conjunto': label,
                'Amostras': f"{len(data):,}",
                'Média': f"{data.mean():.2f}",
                'Mediana': f"{np.median(data):.2f}",
                'Std': f"{data.std():.2f}",
                'Min': f"{data.min():.2f}",
                'Max': f"{data.max():.2f}",
                'Q1': f"{np.percentile(data, 25):.2f}",
                'Q3': f"{np.percentile(data, 75):.2f}",
                'IQR': f"{np.percentile(data, 75) - np.percentile(data, 25):.2f}"
            })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    print("="*80)

def plot_comparative_density(df_train, df_valid, df_test, df_oot, column='tempo_entrega'):
    """
    Plot de densidade sobreposto para fácil comparação
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Preparar dados
    datasets = [
        (df_train[column].dropna(), 'Treino', '#1f77b4'),
        (df_valid[column].dropna(), 'Validação', '#2ca02c'),
        (df_test[column].dropna(), 'Teste', '#ff7f0e'),
        (df_oot[column].dropna(), 'OOT', '#d62728')
    ]
    
    # Plotar KDE para cada conjunto
    for data, label, color in datasets:
        if len(data) > 0:
            from scipy.stats import gaussian_kde
            try:
                kde = gaussian_kde(data)
                x_range = np.linspace(
                    min([d[0].min() for d in datasets if len(d[0]) > 0]),
                    max([d[0].max() for d in datasets if len(d[0]) > 0]),
                    1000
                )
                ax.plot(x_range, kde(x_range), label=label, color=color, linewidth=2.5, alpha=0.8)
                
                # Adicionar linha vertical na média
                mean_val = data.mean()
                ax.axvline(mean_val, color=color, linestyle='--', alpha=0.5, linewidth=1)
                ax.text(mean_val, kde(mean_val)*1.05, f'{mean_val:.1f}', 
                       color=color, fontsize=9, ha='center',
                       bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
            except:
                pass
    
    ax.set_title(f'Comparação de Densidade de {column}', fontsize=16, fontweight='bold')
    ax.set_xlabel(column, fontsize=12)
    ax.set_ylabel('Densidade', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Adicionar área sombreada para IQR do Treino (referência)
    if len(datasets[0][0]) > 0:
        train_data = datasets[0][0]
        q1_train = np.percentile(train_data, 25)
        q3_train = np.percentile(train_data, 75)
        ax.axvspan(q1_train, q3_train, alpha=0.1, color='blue', label='IQR Treino')
    
    plt.tight_layout()
    plt.show()

def visualize_all_comparisons(df_train, df_valid, df_test, df_oot, column='tempo_entrega'):
    """
    Função principal que executa todas as visualizações
    """
    print(f"\n📊 ANALISANDO DISTRIBUIÇÃO DE: {column}")
    print("="*60)
    
    # 1. Histogramas individuais
    print("\n1. Gerando histogramas individuais...")
    stats = plot_histograms_comparison(df_train, df_valid, df_test, df_oot, column)
    
    # 2. Boxplot comparativo
    print("\n2. Gerando boxplot comparativo...")
    plot_boxplot_comparison(df_train, df_valid, df_test, df_oot, column)
    
    # 3. Densidade comparativa
    print("\n3. Gerando gráfico de densidade comparativo...")
    plot_comparative_density(df_train, df_valid, df_test, df_oot, column)
    
    return stats

# Versão alternativa para trabalhar com Series diretamente
def visualize_from_series(train_series, valid_series, test_series, oot_series, column_name='tempo_entrega'):
    """
    Versão para trabalhar com Series ao invés de DataFrames completos
    """
    # Converter para DataFrames temporários
    df_train_temp = pd.DataFrame({column_name: train_series})
    df_valid_temp = pd.DataFrame({column_name: valid_series})
    df_test_temp = pd.DataFrame({column_name: test_series})
    df_oot_temp = pd.DataFrame({column_name: oot_series})
    
    return visualize_all_comparisons(df_train_temp, df_valid_temp, df_test_temp, df_oot_temp, column_name)

def separa_feature_target(target, dados):
        x = dados.drop(target, axis=1)
        y = dados[target]
        return x, y


def cat_encoder(df = None, categoricas = None, target = None, salvar=False):

    if salvar:
        catboost_encoder = CatBoostEncoder(
            cols=categoricas,
            random_state=42,
            handle_unknown="value",
            handle_missing="value"
        )

        catboost_encoder.fit(df[categoricas],df[target])

        # Salvando o encoder treinado
        joblib.dump(catboost_encoder,"../Modelo_Delivery/models/catboost_encoder.joblib")

    else:
        catboost_encoder = joblib.load("../Modelo_Delivery/models/catboost_encoder.joblib")

        return catboost_encoder
    
def carrega_salva_modelo(opcao, modelo = None):
    # Treina e Salva o Modelo
    if opcao == 'salvar':
        joblib.dump(modelo, "../Modelo_Delivery/models/lgbm_hyperopt.joblib")

        return print('Modelo de risk_transaction Treinado e Salvo com Sucesso!')

    else:
        # Carrega o Classificador e Escora para as bases de Treino, Validação, Teste e OOT
        lgbm_hyperopt = joblib.load("../Modelo_Delivery/models/lgbm_hyperopt.joblib")
        return lgbm_hyperopt
    
def separa_feature_target(target, dados):
    x = dados.drop(target, axis = 1)
    y = dados[[target]]

    return x, y

def aplica_feature_selection_feature_importance(df, target, binarias, categoricas, quantitativas):
    def remove_features_feature_importance(target, df, threshold):
        x, y = separa_feature_target(target, df)
        model = LGBMRegressor(
            random_state=42,
            n_estimators=300,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="regression"
        )
        model.fit(x, y)
        feature_importances = model.feature_importances_
        feature_importance_df = (
            pd.DataFrame({
                "feature": x.columns,
                "importance": feature_importances
            })
            .query("importance > @threshold")
            .sort_values("importance", ascending=False)
        )
        # Normaliza para %
        feature_importance_df["importance"] = (feature_importance_df["importance"]/ feature_importance_df["importance"].sum()* 100)
        return feature_importance_df

    def remove_features_altamente_correlacionadas_quantitativas(df,variaveis_importantes_df,quantitativas,threshold_correlacao=0.9):
        features_quantitativas_importantes = [f for f in variaveis_importantes_df["feature"]if f in quantitativas]

        if len(features_quantitativas_importantes) <= 1:
            print("Nenhuma variável quantitativa removida por correlação.")
            return features_quantitativas_importantes

        df_reduzido = df[features_quantitativas_importantes]
        correlacoes = df_reduzido.corr(method="spearman").abs()

        features_para_remover = set()

        for i in range(len(correlacoes.columns)):
            for j in range(i):
                if correlacoes.iloc[i, j] > threshold_correlacao:
                    col_i = correlacoes.columns[i]
                    col_j = correlacoes.columns[j]

                    imp_i = variaveis_importantes_df.loc[variaveis_importantes_df["feature"] == col_i,"importance"].values[0]
                    imp_j = variaveis_importantes_df.loc[variaveis_importantes_df["feature"] == col_j,"importance"].values[0]

                    # Remove a de menor importância
                    features_para_remover.add(col_i if imp_i < imp_j else col_j)

        if features_para_remover:
            print(f"Variáveis removidas por alta correlação (Spearman > {threshold_correlacao}):")
            for f in sorted(features_para_remover):
                print(f" - {f}")
        else:
            print("Nenhuma variável quantitativa removida por correlação.")

        return [f for f in features_quantitativas_importantes if f not in features_para_remover]

    # 1. Feature importance global (LGBM)
    feature_importances = remove_features_feature_importance(target, df, threshold=0)
    # 2. Correlação apenas nas quantitativas
    quantitativas_filtradas = remove_features_altamente_correlacionadas_quantitativas(df,feature_importances,quantitativas)
    # 3. Binárias e categóricas passam direto
    outras_features = [f for f in feature_importances["feature"]if f not in quantitativas]
    features_finais = set(quantitativas_filtradas + outras_features)
    feature_importances_final = feature_importances[feature_importances["feature"].isin(features_finais)]

    return feature_importances_final

def aplica_feature_selection_shap(df, target, binarias, categoricas, quantitativas):

    def remove_features_shap(target, df, threshold):
        x, y = separa_feature_target(target, df)

        model = LGBMRegressor(
            device='gpu',                         # Usa GPU (se disponível) - substitui tree_method='gpu_hist'
            verbosity = -1,                       # Nível de verbosidade (-1: silencioso, 0: erros, 1: avisos, 2: informações)                
            random_state=42,                      # Semente aleatória para reproducibilidade dos resultados
            boosting_type='gbdt',                 # Tipo de boosting 'gbdt' (Gradient Boosting Decision Tree), 'dart' (Dropouts meet Multiple Additive Regression Trees) ou 'goss' (Gradient-based One-Side Sampling)
            objective='regression',               # Objetivo 'binary' (Classificação Binária) 'regression' (Regressão)
            metric='rmse',                        # Métrica de avaliação 'binary_logloss' (Classificação Binária) 'rmse' (Regressão)
            importance_type='gain',               # Método escolhido para calcular o Feature Importance, podendo ser Gain (ganho médio de informação ao utilizar a Feature), Weight (número de vezes que a Feature foi utilizada) ou Cover (número de amostras impactadas pela Feature)
            #class_weight={0:1, 1:class_weight},  # Pesos para classes (ou 'balanced')
            n_estimators=300,                     # Número de árvores no modelo
            max_depth=10,                         # Profundidade máxima
            learning_rate=0.05,                   # Taxa de aprendizado
            max_bin=255,                          # quantidade de bins que as variáveis numéricas serão divididas
            colsample_bytree=0.5,                 # Fração de features por árvore
            subsample=0.5,                        # Fração de amostras por árvore
            reg_alpha=5,                          # Regularização L1
            reg_lambda=5,                         # Regularização L2
            min_split_gain=5,                     # Controle de poda da árvore, maior gamma leva a menos crescimento da árvore
            num_leaves=30,                        # número máximo de folhas por árvore (controle essencial para evitar overfitting no crescimento leaf-wise)
            min_data_in_leaf=300,                 # quantidade de amostras necessárias para que uma Folha seja válida
            min_sum_hessian_in_leaf=0.001,        # A soma das Hessianas em uma folha mede o “peso estatístico” daquela folha, portanto, representa o mínimo na soma das Hessianas em uma folha
            min_child_weight = 0.001,             # A soma das Hessianas em uma folha mede o “peso estatístico” daquela folha, portanto, representa o mínimo na soma das Hessianas em uma folha
            path_smooth = 10                      # Parâmetro de suavização para evitar grandes variações na predição entre nós pai e filho
        )

        model.fit(x, y)

        # SHAP (tree-based)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x)

        # Mean Absolute SHAP
        shap_importance = np.abs(shap_values).mean(axis=0)

        shap_importance_df = (
            pd.DataFrame({
                "feature": x.columns,
                "importance": shap_importance
            })
            .query("importance > @threshold")
            .sort_values("importance", ascending=False)
        )

        shap_importance_df["importance"] = (shap_importance_df["importance"]/ shap_importance_df["importance"].sum()* 100)

        return {
            "importance_df": shap_importance_df,
            "model": model,
            "X": x,
            "shap_values": shap_values
        }

    def remove_features_altamente_correlacionadas_quantitativas(df,variaveis_importantes_df,quantitativas,threshold_correlacao=0.9):
        features_quantitativas_importantes = [f for f in variaveis_importantes_df["feature"]if f in quantitativas]

        if len(features_quantitativas_importantes) <= 1:
            print("Nenhuma variável quantitativa removida por correlação.")
            return features_quantitativas_importantes

        df_reduzido = df[features_quantitativas_importantes]
        correlacoes = df_reduzido.corr(method="spearman").abs()

        features_para_remover = set()

        for i in range(len(correlacoes.columns)):
            for j in range(i):
                if correlacoes.iloc[i, j] > threshold_correlacao:
                    col_i = correlacoes.columns[i]
                    col_j = correlacoes.columns[j]

                    imp_i = variaveis_importantes_df.loc[variaveis_importantes_df["feature"] == col_i,"importance"].values[0]
                    imp_j = variaveis_importantes_df.loc[variaveis_importantes_df["feature"] == col_j,"importance"].values[0]

                    features_para_remover.add(col_i if imp_i < imp_j else col_j)

        return [f for f in features_quantitativas_importantes if f not in features_para_remover]

    # ======================================================
    # 1. SHAP global
    # ======================================================
    shap_output = remove_features_shap(target, df, threshold=0)

    shap_importances = shap_output["importance_df"]

    # ======================================================
    # 2. Correlação apenas nas quantitativas
    # ======================================================
    quantitativas_filtradas = remove_features_altamente_correlacionadas_quantitativas(df,shap_importances,quantitativas)

    # ======================================================
    # 3. Binárias e categóricas passam direto
    # ======================================================
    outras_features = [f for f in shap_importances["feature"]if f not in quantitativas]
    features_finais = set(quantitativas_filtradas + outras_features)
    shap_importances_final = shap_importances[shap_importances["feature"].isin(features_finais)]

    return {
        "feature_importance": shap_importances_final,
        "model": shap_output["model"],
        "X": shap_output["X"],
        "shap_values": shap_output["shap_values"]
    }


def metricas_regressao(model_name, y_train, y_pred_train, y_test, y_pred_test, etapa_1='treino', etapa_2='teste', por_faixa=False):

    # --------------------------
    # Funções auxiliares
    # --------------------------
    def var20(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=np.float64).flatten()
        y_pred = np.asarray(y_pred, dtype=np.float64).flatten()
        y_true_safe = np.where(y_true == 0, 1e-10, y_true)
        relative_error = np.abs(y_pred - y_true) / y_true_safe
        within_20_percent = relative_error <= 0.20
        return np.mean(within_20_percent)

    def under20(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=np.float64).flatten()
        y_pred = np.asarray(y_pred, dtype=np.float64).flatten()
        y_true_safe = np.where(y_true == 0, 1e-10, y_true)
        relative_error = (y_true - y_pred) / y_true_safe
        return np.mean(relative_error > 0.20)

    def over20(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=np.float64).flatten()
        y_pred = np.asarray(y_pred, dtype=np.float64).flatten()
        y_true_safe = np.where(y_true == 0, 1e-10, y_true)
        relative_error = (y_pred - y_true) / y_true_safe
        return np.mean(relative_error > 0.20)

    def rmsle(y_true, y_pred):
        y_true = np.maximum(np.asarray(y_true, dtype=np.float64).flatten(), 0)
        y_pred = np.maximum(np.asarray(y_pred, dtype=np.float64).flatten(), 0)
        return np.sqrt(mean_squared_log_error(y_true, y_pred))

    def cohen_kappa_deciles(y_true, y_pred, n_deciles=10):
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        if len(np.unique(y_true)) < n_deciles:
            n_deciles = len(np.unique(y_true))
        if n_deciles < 2:
            return 0.0
        bins = np.percentile(y_true, np.linspace(0, 100, n_deciles+1))
        y_true_cat = np.digitize(y_true, bins, right=True) - 1
        y_pred_cat = np.digitize(y_pred, bins, right=True) - 1
        return cohen_kappa_score(y_true_cat, y_pred_cat)

    def calcular_mape(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=np.float64).flatten()
        y_pred = np.asarray(y_pred, dtype=np.float64).flatten()
        y_true_safe = np.where(y_true == 0, 1e-10, y_true)
        return np.mean(np.abs(y_pred - y_true) / y_true_safe) * 100

    # --------------------------
    # Função para calcular métricas
    # --------------------------
    def calcular_metricas(y_true, y_pred, etapa, pct_amostras=100):
        data = {
            'MAE': [mean_absolute_error(y_true, y_pred)],
            'RMSE': [np.sqrt(mean_squared_error(y_true, y_pred))],
            'RMSLE': [rmsle(y_true, y_pred)],
            'MAPE (%)': [calcular_mape(y_true, y_pred)],
            'Var20 (%)': [var20(y_true, y_pred) * 100],
            'Subestimação (%)': [under20(y_true, y_pred) * 100],
            'Superestimação (%)': [over20(y_true, y_pred) * 100],
            "CohenKappa": [cohen_kappa_deciles(y_true, y_pred)],
            'Etapa': [etapa],
            'Modelo': [model_name]
        }
        if pct_amostras is not None:
            data['Pct_amostras (%)'] = [pct_amostras]
        return pd.DataFrame(data)

    # Métricas globais
    metricas_treino = calcular_metricas(y_train, y_pred_train, etapa_1)
    metricas_teste = calcular_metricas(y_test, y_pred_test, etapa_2)

    # --------------------------
    # Métricas por faixa de tempo (opcional)
    # --------------------------
    if por_faixa:
        bins = [0, 30, 45, 60, 75, 90, 105, np.inf]
        labels = ['Até 30min','Até 45min','Até 60min','Até 75min','Até 90min','Até 105min','Mais que 105min']

        metricas_treino_faixa = []
        metricas_teste_faixa = []

        y_train_arr = np.asarray(y_train).flatten()
        y_pred_train_arr = np.asarray(y_pred_train).flatten()
        y_test_arr = np.asarray(y_test).flatten()
        y_pred_test_arr = np.asarray(y_pred_test).flatten()

        total_train = len(y_train_arr)
        total_test = len(y_test_arr)

        # Treino por faixa
        for i in range(len(bins)-1):
            mask = (y_train_arr > bins[i]) & (y_train_arr <= bins[i+1])
            if np.any(mask):
                pct = np.sum(mask) / total_train * 100
                df_faixa = calcular_metricas(
                    y_train_arr[mask], 
                    y_pred_train_arr[mask], 
                    f"{etapa_1} ({labels[i]})",
                    pct_amostras=pct
                )
                metricas_treino_faixa.append(df_faixa)

        # Teste por faixa
        for i in range(len(bins)-1):
            mask = (y_test_arr > bins[i]) & (y_test_arr <= bins[i+1])
            if np.any(mask):
                pct = np.sum(mask) / total_test * 100
                df_faixa = calcular_metricas(
                    y_test_arr[mask], 
                    y_pred_test_arr[mask], 
                    f"{etapa_2} ({labels[i]})",
                    pct_amostras=pct
                )
                metricas_teste_faixa.append(df_faixa)

        metricas_treino = pd.concat([metricas_treino] + metricas_treino_faixa).reset_index(drop=True)
        metricas_teste = pd.concat([metricas_teste] + metricas_teste_faixa).reset_index(drop=True)

    return pd.concat([metricas_treino, metricas_teste]).reset_index(drop=True)


def metricas_modelos_juntos_regressao(lista_modelos):
    if len(lista_modelos) > 0:
        metricas_modelos = pd.concat(lista_modelos)
    else:
        return pd.DataFrame()

    # Redefinir índice e arredondar
    df = metricas_modelos.reset_index(drop=True)
    df = df.round(2)

    metricas_cols = ['MAE', 'RMSE', 'RMSLE', 'MAPE (%)', 
                     'Var20 (%)', 'Subestimação (%)', 'Superestimação (%)', 
                     'CohenKappa', 'Pct_amostras (%)']

    # Função para colorir por etapa
    def color_etapa(val):
        val = str(val).lower()
        color = 'black'
        if 'treino' in val:
            color = 'blue'
        elif 'teste' in val or 'validacao' in val:
            color = 'red'
        return f'color: {color}; font-weight: bold;'

    # Função para criar borda inferior quando o modelo muda
    def separador_modelos(df):
        estilos = pd.DataFrame('', index=df.index, columns=df.columns)
        modelos = df['Modelo']
        for i in range(len(modelos)-1):
            if modelos[i] != modelos[i+1]:
                estilos.loc[i, :] = 'border-bottom: 3px solid black;'
        return estilos

    # Estilizando o DataFrame
    styled_df = df.style \
        .format({col: "{:.2f}" for col in metricas_cols}) \
        .applymap(lambda x: 'color: black; font-weight: bold; background-color: white; font-size: 14px', subset=pd.IndexSlice[:, :]) \
        .applymap(color_etapa, subset=pd.IndexSlice[:, ['Etapa']]) \
        .applymap(lambda x: 'color: black; font-weight: bold; background-color: white; font-size: 14px', subset=pd.IndexSlice[:, metricas_cols]) \
        .apply(separador_modelos, axis=None) \
        .set_table_styles([
            {'selector': 'thead', 'props': [('color', 'black'), ('font-weight', 'bold'), ('background-color', 'lightgray')]}
        ])
    
    return styled_df

def metricas_regressao_diarias(df,coluna_data,y_true_col,y_pred_col):
    # --------------------------
    # Funções auxiliares
    # --------------------------
    def var20(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        y_true_safe = np.where(y_true == 0, 1e-10, y_true)
        return np.mean(np.abs(y_pred - y_true) / y_true_safe <= 0.20) * 100

    def rmsle(y_true, y_pred):
        y_true = np.maximum(np.asarray(y_true), 0)
        y_pred = np.maximum(np.asarray(y_pred), 0)
        return np.sqrt(mean_squared_log_error(y_true, y_pred)) * 100

    # --------------------------
    # Preparação
    # --------------------------
    df = df.copy()
    df[coluna_data] = pd.to_datetime(df[coluna_data])
    df = df.sort_values(coluna_data)

    resultados = []

    # --------------------------
    # Loop diário
    # --------------------------
    for data, grupo in df.groupby(df[coluna_data].dt.date):

        y_true = grupo[y_true_col].values
        y_pred = grupo[y_pred_col].values

        resultados.append({
            "data_pedido": pd.to_datetime(data),
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "RMSLE": rmsle(y_true, y_pred),
            "Var20 (%)": var20(y_true, y_pred),
            "Qtd_registros": len(grupo)
        })

    return pd.DataFrame(resultados)


def Regressor(loss_function, x_train, y_train, x_test, y_test):
    
    cols = list(x_train.columns)
    x_train = x_train[cols]
    x_test = x_test[cols]

    base_params = dict(
        device='gpu',                         # Usa GPU (se disponível) - substitui tree_method='gpu_hist'
        verbosity = -1,                       # Nível de verbosidade (-1: silencioso, 0: erros, 1: avisos, 2: informações)                
        random_state=42,                      # Semente aleatória para reproducibilidade dos resultados
        boosting_type='gbdt',                 # Tipo de boosting 'gbdt' (Gradient Boosting Decision Tree), 'dart' (Dropouts meet Multiple Additive Regression Trees) ou 'goss' (Gradient-based One-Side Sampling)
        #objective='regression',               # Função de custo 'binary' (Classificação Binária) 'regression' (Regressão)
        #metric='mae',                         # Métrica de avaliação durante as Logs de Treinamento
        importance_type='gain',               # Método escolhido para calcular o Feature Importance, podendo ser Gain (ganho médio de informação ao utilizar a Feature), Weight (número de vezes que a Feature foi utilizada) ou Cover (número de amostras impactadas pela Feature)
        n_estimators=300,                     # Número de árvores no modelo
        max_depth=7,                         # Profundidade máxima
        learning_rate=0.05,                   # Taxa de aprendizado
        max_bin=255,                          # quantidade de bins que as variáveis numéricas serão divididas
        # colsample_bytree=0.5,                 # Fração de features por árvore
        # subsample=0.5,                        # Fração de amostras por árvore
        # reg_alpha=5,                          # Regularização L1
        # reg_lambda=5,                         # Regularização L2
        # min_split_gain=5,                     # Controle de poda da árvore, maior gamma leva a menos crescimento da árvore
        # num_leaves=30,                        # número máximo de folhas por árvore (controle essencial para evitar overfitting no crescimento leaf-wise)
        # min_data_in_leaf=300,                 # quantidade de amostras necessárias para que uma Folha seja válida
        # min_sum_hessian_in_leaf=0.001,        # A soma das Hessianas em uma folha mede o “peso estatístico” daquela folha, portanto, representa o mínimo na soma das Hessianas em uma folha
        # min_child_weight = 0.001,             # A soma das Hessianas em uma folha mede o “peso estatístico” daquela folha, portanto, representa o mínimo na soma das Hessianas em uma folha
        # path_smooth = 10                      # Parâmetro de suavização para evitar grandes variações na predição entre nós pai e filho
        #class_weight={0:1, 1:class_weight},   # Pesos para classes (ou 'balanced')
    )

    models = {
        "MAE": LGBMRegressor(
            objective="regression_l1",
            metric="l1",
            **base_params
        ),

        "RMSE": LGBMRegressor(
            objective="regression",
            metric="l2",
            **base_params
        ),

        "Huber": LGBMRegressor(
            objective="huber",
            metric="l1",
            huber_delta=1.0,
            **base_params
        ),

        "RMSLE": LGBMRegressor(
            objective="regression",
            metric="l2",
            **base_params
        ),

        "Gamma": LGBMRegressor(
            objective="gamma",
            metric="gamma",
            **base_params
        )
    }

    if loss_function not in models:
        raise ValueError(f"Loss function '{loss_function}' não suportada.")

    model = models[loss_function]
    
    # Tratamento especial para RMSLE: aplicar log1p no target e depois expm1 nas predições
    if loss_function == "RMSLE":
        # Aplicar log1p no target (ln(1+y))
        y_train_transformed = np.log1p(y_train)
        
        # Treinar o modelo no target transformado
        model.fit(x_train, y_train_transformed)
        
        # Fazer predições no espaço transformado
        y_pred_train_transformed = model.predict(x_train)
        y_pred_test_transformed = model.predict(x_test)
        
        # Aplicar expm1 para voltar ao espaço original (e^pred - 1)
        y_pred_train = np.expm1(y_pred_train_transformed)
        y_pred_test = np.expm1(y_pred_test_transformed)
        
        # IMPORTANTE: ajustar para garantir valores não-negativos
        y_pred_train = np.maximum(y_pred_train, 0)
        y_pred_test = np.maximum(y_pred_test, 0)
        
    else:
        # Para outras funções de perda, treinar normalmente
        model.fit(x_train, y_train)
        y_pred_train = model.predict(x_train)
        y_pred_test = model.predict(x_test)

    return model, y_pred_train, y_pred_test

    
def otimizacao_hyperopt_regression(x_train, y_train, x_test, y_test, max_evals):

    # Espaço de busca dos hiperparâmetros
    search_space = {
        'n_estimators': hp.choice('n_estimators', [700, 800, 900, 1000]),
        'max_depth': hp.choice('max_depth', [10, 11, 12]),
        'learning_rate': hp.uniform('learning_rate', 0.05, 0.1),
        'max_bin': hp.choice('max_bin', [64, 128, 255]),
        'reg_alpha': hp.uniform('reg_alpha', 0, 1),
        'reg_lambda': hp.uniform('reg_lambda', 0, 1),
        'min_split_gain': hp.uniform('min_split_gain', 0, 10),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
        'subsample': hp.uniform('subsample', 0.5, 1),
        'num_leaves': hp.choice('num_leaves', [30, 35, 40, 45, 50]),
        'min_data_in_leaf': hp.choice('min_data_in_leaf', [300, 400, 500]),
        'min_sum_hessian_in_leaf': hp.uniform('min_sum_hessian_in_leaf', 0.001, 0.005),
        #'path_smooth': hp.uniform('path_smooth', 0, 20)
    }


    # Função de custo do Hyperopt
    def objective(params):
        # Split interno para validação
        X_tr, X_val, y_tr, y_val = train_test_split(
            x_train, y_train, test_size=0.2, random_state=42
        )

        model = LGBMRegressor(
            device='gpu',                         # Usa GPU (se disponível) - substitui tree_method='gpu_hist'
            verbosity = -1,                       # Nível de verbosidade (-1: silencioso, 0: erros, 1: avisos, 2: informações)                
            random_state=42,                      # Semente aleatória para reproducibilidade dos resultados
            boosting_type='gbdt',                 # Tipo de boosting 'gbdt' (Gradient Boosting Decision Tree), 'dart' (Dropouts meet Multiple Additive Regression Trees) ou 'goss' (Gradient-based One-Side Sampling)
            importance_type='gain',               # Método escolhido para calcular o Feature Importance, podendo ser Gain (ganho médio de informação ao utilizar a Feature), Weight (número de vezes que a Feature foi utilizada) ou Cover (número de amostras impactadas pela Feature)
            objective='gamma',  # Função de Custo
            metric='rmse',  # usado apenas para log/monitoramento
            **params
        )

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=[early_stopping(stopping_rounds=20, verbose=False)]
        )


        # Previsões na validação
        preds = model.predict(X_val)
        preds = np.maximum(preds, 1e-6)

        # RMSLE real
        score = np.sqrt(mean_squared_log_error(y_val, preds))

        return {'loss': score, 'status': STATUS_OK}

    # Rodando a otimização
    trials = Trials()
    best = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        rstate=np.random.default_rng(42)
    )

    # Reconstruir hiperparâmetros escolhidos
    best['n_estimators'] = [700, 800, 900, 1000][best['n_estimators']]
    best['max_depth'] = [10, 11, 12][best['max_depth']]
    best['max_bin'] = [64, 128, 255][best['max_bin']]
    best['num_leaves'] = [30, 35, 40, 45, 50][best['num_leaves']]
    best['min_data_in_leaf'] = [300, 400, 500][best['min_data_in_leaf']]


    # Treinar modelo final com os melhores hiperparâmetros
    final_model = LGBMRegressor(
        device='gpu',                         # Usa GPU (se disponível) - substitui tree_method='gpu_hist'
        verbosity = -1,                       # Nível de verbosidade (-1: silencioso, 0: erros, 1: avisos, 2: informações)                
        random_state=42,                      # Semente aleatória para reproducibilidade dos resultados
        boosting_type='gbdt',                 # Tipo de boosting 'gbdt' (Gradient Boosting Decision Tree), 'dart' (Dropouts meet Multiple Additive Regression Trees) ou 'goss' (Gradient-based One-Side Sampling)
        importance_type='gain',               # Método escolhido para calcular o Feature Importance, podendo ser Gain (ganho médio de informação ao utilizar a Feature), Weight (número de vezes que a Feature foi utilizada) ou Cover (número de amostras impactadas pela Feature)
        objective='gamma',  # Função de Custo
        metric='rmse',  # usado apenas para log/monitoramento
        **best
    )

    final_model.fit(
        x_train, y_train,
        eval_set=[(x_test, y_test)],
        eval_metric='rmse',
        callbacks=[early_stopping(stopping_rounds=20, verbose=False)]
    )

    # Previsões finais
    y_pred_train = final_model.predict(x_train)
    y_pred_test = final_model.predict(x_test)

    # Organiza melhores hiperparâmetros em DataFrame
    hiperparametros = pd.DataFrame([best])

    return final_model, y_pred_train, y_pred_test, hiperparametros, trials


