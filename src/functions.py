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
from datetime import datetime, date

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

# Bibliotecas de Modelos de Machine Learning
import joblib
from joblib import Parallel, delayed
import pickle
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
import shap

# Bibliotecas de Métricas de Machine Learning
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, accuracy_score, roc_auc_score, roc_curve, auc, precision_score, recall_score, precision_recall_curve, average_precision_score, f1_score, log_loss, brier_score_loss, confusion_matrix, silhouette_score

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

def calcular_psi_temporal(df, coluna_data='data_pedido', coluna_metricas='tempo_entrega', nome_metrica='Tempo de Entrega'):
    """
    Função única para cálculo e plotagem de PSI temporal
    Analisa a estabilidade da distribuição de uma métrica ao longo do tempo
    
    Parameters:
    df: DataFrame - DataFrame com os dados
    coluna_data: str - Nome da coluna de data
    coluna_metricas: str - Nome da coluna com a métrica a analisar
    nome_metrica: str - Nome amigável da métrica para exibição
    
    Returns:
    psi_value: float - Valor do PSI calculado
    psi_df: DataFrame - DataFrame com cálculos detalhados
    """
    
    # ============================================================================
    # ETAPA 1: PREPARAÇÃO DOS DADOS TEMPORAIS
    # ============================================================================
    
    # Converter coluna de data para datetime
    df[coluna_data] = pd.to_datetime(df[coluna_data])
    
    # Ordenar por data
    df = df.sort_values(coluna_data)
    
    # Dividir dados: primeiro mês (base) vs último mês (atual)
    data_minima = df[coluna_data].min()
    data_maxima = df[coluna_data].max()
    
    dados_base = df[df[coluna_data] <= data_minima + pd.DateOffset(months=1)]
    dados_atual = df[df[coluna_data] >= data_maxima - pd.DateOffset(months=1)]
    
    print("📊 ANÁLISE PSI TEMPORAL")
    print(f"Período base: {dados_base[coluna_data].min().date()} a {dados_base[coluna_data].max().date()}")
    print(f"Período atual: {dados_atual[coluna_data].min().date()} a {dados_atual[coluna_data].max().date()}")
    print(f"Registros base: {len(dados_base):,} | Registros atual: {len(dados_atual):,}")
    
    # ============================================================================
    # ETAPA 2: CÁLCULO DO PSI
    # ============================================================================
    
    def calcular_psi(distribuicao_base, distribuicao_atual, num_buckets=10):
        """Calcula o Population Stability Index entre duas distribuições"""
        
        # Remover valores nulos
        base_limpa = np.array(distribuicao_base[~pd.isnull(distribuicao_base)])
        atual_limpa = np.array(distribuicao_atual[~pd.isnull(distribuicao_atual)])
        
        # Definir pontos de corte pelos percentis
        pontos_corte = np.percentile(base_limpa, [i * 100/num_buckets for i in range(num_buckets + 1)])
        pontos_corte = np.unique(pontos_corte)  # Garantir pontos únicos
        
        # Calcular frequências em cada bucket
        freq_base = np.histogram(base_limpa, pontos_corte)[0]
        freq_atual = np.histogram(atual_limpa, pontos_corte)[0]
        
        # Adicionar valor pequeno para evitar divisão por zero
        freq_base = freq_base + 0.0001
        freq_atual = freq_atual + 0.0001
        
        # Calcular proporções
        prop_base = freq_base / len(base_limpa)
        prop_atual = freq_atual / len(atual_limpa)
        
        # Calcular componentes do PSI para cada bucket
        componentes_psi = (prop_atual - prop_base) * np.log(prop_atual / prop_base)
        psi_total = np.sum(componentes_psi)
        
        # Criar DataFrame com resultados detalhados
        psi_detalhado = pd.DataFrame({
            'bucket': range(1, len(pontos_corte)),
            'frequencia_base': freq_base,
            'frequencia_atual': freq_atual,
            'proporcao_base': prop_base,
            'proporcao_atual': prop_atual,
            'componente_psi': componentes_psi
        })
        
        return psi_total, psi_detalhado
    
    # Calcular PSI
    valor_psi, df_psi = calcular_psi(dados_base[coluna_metricas], dados_atual[coluna_metricas])
    
    # ============================================================================
    # ETAPA 3: PLOTAGEM DOS GRÁFICOS
    # ============================================================================
    
    # Criar figura com dois subplots
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 1, height_ratios=[1, 2])
    
    # --- GRÁFICO SUPERIOR: EVOLUÇÃO DO PSI ---
    ax_superior = plt.subplot(gs[0])
    
    # Plotar linha do PSI acumulado
    ax_superior.plot(df_psi['bucket'], df_psi['componente_psi'].cumsum(), 
                     marker='o', linewidth=2, markersize=8, color='blue', 
                     label='PSI Acumulado')
    
    # Linha do valor total do PSI
    ax_superior.axhline(y=valor_psi, color='red', linestyle='--', linewidth=2, 
                        label=f'PSI Total: {valor_psi:.4f}')
    
    # Áreas coloridas para interpretação
    ax_superior.axhspan(0, 0.1, alpha=0.3, color='green', label='PSI ≤ 0.1 (Estável)')
    ax_superior.axhspan(0.1, 0.25, alpha=0.3, color='yellow', label='0.1 < PSI ≤ 0.25 (Atenção)')
    ax_superior.axhspan(0.25, max(valor_psi, 0.5), alpha=0.3, color='red', label='PSI > 0.25 (Instável)')
    
    ax_superior.set_title(f'Análise de Estabilidade Temporal - {nome_metrica}', 
                          fontsize=14, fontweight='bold', pad=20)
    ax_superior.set_ylabel('Valor do PSI', fontsize=12)
    ax_superior.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax_superior.grid(True, alpha=0.3)
    
    # --- GRÁFICO INFERIOR: DISTRIBUIÇÕES COMPARADAS ---
    ax_inferior = plt.subplot(gs[1])
    
    # Preparar dados para barras
    buckets = df_psi['bucket']
    posicoes_x = np.arange(len(buckets))
    largura_barra = 0.35
    
    # Plotar barras das distribuições
    barras_base = ax_inferior.bar(posicoes_x - largura_barra/2, df_psi['proporcao_base'] * 100, 
                                  largura_barra, label='Distribuição Base (%)', 
                                  alpha=0.7, color='blue')
    
    barras_atual = ax_inferior.bar(posicoes_x + largura_barra/2, df_psi['proporcao_atual'] * 100, 
                                  largura_barra, label='Distribuição Atual (%)', 
                                  alpha=0.7, color='orange')
    
    # Adicionar valores nas barras (apenas se altura > 5%)
    for barra in [barras_base, barras_atual]:
        for retangulo in barra:
            altura = retangulo.get_height()
            if altura > 5:
                ax_inferior.text(retangulo.get_x() + retangulo.get_width()/2., altura + 0.5,
                               f'{altura:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax_inferior.set_xlabel('Percentis (P10 a P90)', fontsize=12)
    ax_inferior.set_ylabel('Proporção da População (%)', fontsize=12)
    ax_inferior.set_title('Comparação das Distribuições por Percentil', fontsize=13, fontweight='bold')
    ax_inferior.set_xticks(posicoes_x)
    ax_inferior.set_xticklabels([f'P{(i+1)*10}' for i in range(len(buckets))])
    ax_inferior.legend()
    ax_inferior.grid(True, alpha=0.3)
    
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
    
    # Texto com estatísticas
    texto_estatisticas = f'''
    📈 RESULTADO DO PSI:
    • Valor PSI: {valor_psi:.4f}
    • Interpretação: {interpretacao}
    • Período Base: {len(dados_base):,} registros
    • Período Atual: {len(dados_atual):,} registros
    • Métrica: {nome_metrica}
    '''
    
    ax_inferior.text(1.02, 0.98, texto_estatisticas, transform=ax_inferior.transAxes, 
                    fontsize=11, verticalalignment='top', color=cor_interpretacao,
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Print final no console
    print(f"\n🎯 RESULTADO FINAL: PSI = {valor_psi:.4f} - {interpretacao}")
    print("=" * 60)
    
    return valor_psi, df_psi