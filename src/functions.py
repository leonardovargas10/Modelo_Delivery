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

# Bibliotecas de Modelos de Machine Learning
import joblib
from joblib import Parallel, delayed
import pickle
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.cluster import KMeans
import networkx as nx
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
