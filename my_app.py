import pandas as pd
from pandas_datareader import data as pdr
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta, date
from dash import Dash, html,Input, Output, dcc, State
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template

df = pd.read_csv(r'assets/kmeans-empresas-b3.csv', sep=';', index_col='Unnamed: 0')

linkedin_btn = html.A(
    dbc.Button(
        html.Img(src='https://www.svgrepo.com/show/922/linkedin.svg',style={'width': '30px', 'height': '30px'}),
        color='link',
        class_name='mr-2'),
    href='https://www.linkedin.com/in/renanldbr',
    target='_brank'
)

##### Botão Github IO #####  
git_hub_io = html.A(
    dbc.Button(
        html.Img(src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png',style={'width': '30px', 'height': '30px'}),
        color='link',
        class_name='mr-2'),
    href='https://renanldbr.github.io/',
    target='_brank'
)

##### Botão Github #####  
git_hub = html.A(
    dbc.Button(
        html.Img(src='data:image/png;base64,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',style={'width': '30px', 'height': '30px'}),
        color='link',
        class_name='mr-2'),
    href='https://github.com/renanldbr',
    target='_brank'
)

disclaimer = """
Aviso Legal: Este dashboard é fornecido apenas para fins didáticos e não deve ser interpretado como uma recomendação de investimento. O criador do dashboard não se responsabiliza por quaisquer operações realizadas pelo usuário com base nas informações apresentadas. É importante ressaltar que investimentos envolvem riscos e podem resultar em perdas financeiras. Recomendamos que os usuários busquem aconselhamento financeiro profissional antes de tomar qualquer decisão de investimento.

O criador do dashboard não garante a precisão, confiabilidade ou integridade das informações apresentadas e não se responsabiliza por quaisquer erros ou omissões. As informações contidas neste dashboard são fornecidas "como estão" e "conforme disponíveis" sem qualquer garantia expressa ou implícita.

Este aviso legal segue as legislações brasileiras mais atuais."""

modelo = """
Neste dashboard, aproveitamos a classificação das principais empresas brasileiras negociadas na B3, realizada por meio da técnica de K-Means, com base em seus indicadores econômicos. Essa análise nos permite agrupar as empresas de forma inteligente e eficiente, facilitando a tomada de decisões de investimento.

Com o poder do nosso dashboard, os usuários têm a oportunidade de simular a montagem de uma carteira de investimentos utilizando a renomada Teoria da Fronteira Eficiente de Markowitz. Essa teoria é uma estratégia consagrada para otimizar carteiras de investimento, buscando o melhor equilíbrio entre risco e retorno.

Ao combinar os insights fornecidos pelo K-Means e a Teoria da Fronteira Eficiente de Markowitz, nosso dashboard capacita os investidores a construírem carteiras diversificadas e bem ajustadas ao perfil de risco individual, permitindo que alcancem seus objetivos financeiros de forma mais informada e assertiva. Com esta poderosa ferramenta, exploramos novos horizontes no mercado financeiro, trazendo a ciência e a lucratividade para o alcance de todos."""

#app
app = Dash(__name__, external_stylesheets=[dbc.themes.JOURNAL])
load_figure_template('minty')

data_f = datetime.today().strftime('%Y-%m-%d')
data_i = datetime.today()-timedelta(days=2)
data_i = data_i.strftime('%Y-%m-%d')

app.layout = html.Div([
    #Row 1
    dbc.Row([
        html.H2('Algoritmos em Ação: Desbravando o Mercado acionário brasileiro com K-Means e Markowitz',
                style={'textAlign':'center'})
    ], align='center'),

    #Row 2
    dbc.Row([
        #Coluna 1
        dbc.Col([
            html.P(modelo, style={'textAlign':'justify'}),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        html.H1('RLdBr Labs', style={'font-style':'oblique 30deg', 'margin-top':'5px','textAlign':'center'}),
                        html.Hr(),
                        html.H6('Wanna See More?', style={'textAlign':'center'}),
                        dbc.Row([
                            dbc.Col(linkedin_btn, lg=3, sm=10),
                            dbc.Col(git_hub_io, lg=3, sm=10),
                            dbc.Col(git_hub, lg=3, sm=10)
                        ], justify='center')
                    ])
                ])
            ])
        ], style={'height':'60vh', 'margin-left':'10px'}, md=2),
                        
        #Coluna 2
        dbc.Col([
            dbc.Row([
                html.H6('Cluster 1: Opção para investidores mais conservadores, pois essas empresas têm um desempenho financeiro estável e menor risco de efeitos incomuns nos impostos.'),
                html.H6('Cluster 2: Essas empresas podem ser mais arriscadas devido ao desempenho financeiro abaixo da média e alta taxa de imposto para cálculos. Investidores com maior apetite a riscos podem considerar esse cluster.'),
                html.H6('Cluster 3: Empresas com desempenho financeiro extremamente forte podem ser atraentes para investidores que buscam alto retorno e têm tolerância a riscos mais elevada.'),
                html.H6('Cluster 4: Empresas com desempenho fraco ou prejuízo e que podem ser arriscadas para investimento, a menos que haja uma estratégia específica para recuperar a performance.'),
                html.H6('Cluster 5: Não há classificação de empresas. Você pode escolher qualquer empresa entre as disponíveis'),
                dcc.RadioItems(['Cluster 1', 'Cluster 2','Cluster 3', 'Cluster 4', 'Cluster 5'], 'Cluster 2', inline=True, labelStyle={'margin-right':'26px'}, id='radio1')
            ], style={'textAlign':'justify'}),      

            dbc.Row([
                dbc.Col([
                    html.H6('Data Inicial', style={'textAlign':'jutify'}),
                    dcc.DatePickerSingle(
                        date='2020-01-01',
                        display_format='DD-MM-Y',
                        id='data_inicial',
                    )
                ], width={'size':3, 'order':1}),
                dbc.Col([
                    html.H6('Data Final', style={'textAlign':'justify'}),
                    dcc.DatePickerSingle(
                        date=datetime.today().strftime('%Y-%m-%d'),
                        display_format='DD-MM-Y',
                        max_date_allowed=datetime.now(),
                        id='data_final',
                    )
                ], width={'size':3, 'order':2}),
            ], justify='center'),

            html.P(),

            dbc.Row([
                dcc.Dropdown(
                    multi=True,
                    id='drop_empresas1',
                ),
            
            dbc.Row([dcc.Graph(id='grafico0')])
        ])
        ]),

        #coluna 3
        dbc.Col([
            dbc.Row([dcc.Graph(id='grafico1')], justify='center'),
            dbc.Row([
                html.H3('Índice de Sharpe'),
                html.P('O Índice de Sharpe é uma métrica que avalia o desempenho ajustado ao risco de um portfólio de investimentos. Ele calcula a diferença entre o retorno do portfólio e o retorno livre de risco (geralmente uma taxa de juros de curto prazo) e, em seguida, divide esse valor pelo desvio padrão (volatilidade) do portfólio. O Índice de Sharpe permite aos investidores comparar e selecionar portfólios que oferecem o melhor retorno por unidade de risco assumido.',
                       style={'textAlign':'justify'})
            ]),
            dbc.Row([
                html.H3('A fronteira Eficiente: O Portfólio de Baixa Volatilidade e Máximo Retorno'),
                html.P('A fronteira eficiente é representada graficamente como uma curva que mostra todas as combinações de ativos que oferecem o máximo retorno esperado para cada nível de risco assumido. Os pontos ao longo dessa curva representam portfólios diversificados que podem ser considerados "ótimo", pois não é possível obter um retorno maior sem aumentar o risco ou reduzir o risco sem diminuir o retorno.', 
                       style={'textAlign':'justify'})
            ]),            
            ]),

        #coluna 4
        dbc.Col([
            dbc.Row([
                dbc.Row([
                    dcc.Graph(id='grafico2'),                    
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Row([
                            html.H6('Portfólio de Baixa Vol. e Máx. Retorno'),
                        ]),
                        dbc.Row([], id='table0'),
                    ]),
                    dbc.Col([
                        dbc.Row([
                            html.H6('Portfólio segundo Sharpe⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀'),
                        ]),
                        dbc.Row([], id='table1')
                    ]),
                ]),
                
            ]),
            dbc.Row([]),
        ]),
    ]),
    
    #Row 3
    dbc.Row([
        html.H1('Disclaimer'),
        html.P(html.I(disclaimer), style={'textAlign':'justify'})
    ])
    ])

#callback

@app.callback(
    Output(component_id='drop_empresas1', component_property='options'),
    Output(component_id='drop_empresas1', component_property='value'),
    Input(component_id='radio1', component_property='value'),
)
def att_valores_clusters(radio1):
    idx = int(radio1[-1])
    if idx in (1, 2, 3, 4):
        idx = idx - 1
        df_filtered = df[df['cluster'] == idx]
        options = [{'label': ticker, 'value': ticker} for ticker in df_filtered['ticker']]
        first_value = df_filtered['ticker'].iloc[0:2].to_list()
        return options, first_value
        
    else:
        options = [{'label': ticker, 'value': ticker} for ticker in df['ticker']]       
        first_value = df['ticker'].iloc[0:2].to_list()
        return options, first_value

@app.callback(
    Output(component_id='grafico0', component_property='figure'),
    Output(component_id='grafico1', component_property='figure'),
    Output(component_id='grafico2', component_property='figure'),
    Output(component_id='table0', component_property='children'),
    Output(component_id='table1', component_property='children'),

    
    Input(component_id='data_inicial', component_property='date'),
    Input(component_id='data_final', component_property='date'),
    Input(component_id='drop_empresas1', component_property='value')
)   
def att_grafico(data_inicial, data_final, empresas_selecionadas):
    yf.pdr_override()
    frames = []
    
    for ativo in empresas_selecionadas:
        try:
            data = pdr.get_data_yahoo(ativo, start=data_inicial, end=data_final)['Adj Close']
            if not data.empty:
                frames.append({ativo:data.iloc[:].to_dict()})
        except:
            continue
    
    data = {}
    for item in frames:
        key = list(item.keys())[0]
        sub_dict = item[key]
        for sub_key, value in sub_dict.items():
            data.setdefault(sub_key,{})[key] = value
    ativos = pd.DataFrame(data)
    ativos = ativos.T
    norm = (ativos/ativos.iloc[0])*100-100
    df1 = norm.sort_index()
       
    grafico0 = go.Figure()
    correl = df1.corr()
    grafico0.add_trace(go.Heatmap(z=correl, x=correl.columns, y=correl.columns))
    for i in range(len(correl.columns)):
        for j in range(len(correl.columns)):
            grafico0.add_annotation(
                x=i,
                y=j,
                text=str(round(correl.iloc[i,j],2)),
                showarrow=False,
                font=dict(color='rgb(0,0,0)', size=15))        
    grafico0.update_layout(title_text='Correlação entre os ativos')
        
    grafico1 = go.Figure()
    for ativo in df1.columns:
        text = [f'{ativo}: {y:.2f}' for y in df1[ativo]]
        grafico1.add_trace(go.Scatter(x=df1.index, y=df1[ativo], name=ativo, text=text, hoverinfo='text'))

    grafico1.update_layout(title_text='Normalização dos Ativos')
    grafico1.update_layout(yaxis=dict(tickformat=',.2f', ticksuffix='%'))
    grafico1.update_xaxes(title_text='Período')
    grafico1.update_yaxes(title_text='Performance')
    grafico1.update_layout(width=520, height=400)

    grafico2 = go.Figure()
    taxa = pdr.get_data_fred('IRSTCI01BRM156N')['IRSTCI01BRM156N'][-1]
    retorno_ativos = np.log(ativos/ativos.shift(1))
    pfolio_retorno = []
    pfolio_volatilidade = []
    sharpe_ratio = []
    all_weights = []

    for i in range(1000):
        weights = np.random.random(len(ativos.columns))
        weights /= np.sum(weights)
        all_weights.append(weights)
        retorno_portfolio = np.sum(weights * retorno_ativos.mean()) * 252
        pfolio_retorno.append(retorno_portfolio)
        volatilidade_portifolio = np.sqrt(np.dot(weights.T, np.dot(retorno_ativos.cov() * 252, weights)))
        pfolio_volatilidade.append(volatilidade_portifolio)
        sharpe_ratio.append(retorno_portfolio - (taxa/ 100) / volatilidade_portifolio)

    pfolio_retorno = np.array(pfolio_retorno)
    pfolio_volatilidade = np.array(pfolio_volatilidade)
    sharpe_ratio = np.array(sharpe_ratio)

    portfolio = {'Retorno': pfolio_retorno,
                'Volatilidade': pfolio_volatilidade,
                'Sharpe Ratio': sharpe_ratio}

    for contar, ticker in enumerate(ativos.columns):
        portfolio[ticker + ' Peso'] = [Peso[contar] for Peso in all_weights]

    df2 = pd.DataFrame(portfolio)

    colunas = ['Retorno', 'Volatilidade', 'Sharpe Ratio'] + [ticker + ' Peso' for ticker in ativos.columns]
    df2 = df2[colunas]

    grafico2.add_trace(go.Scatter(x=df2['Volatilidade'], y=df2['Retorno'], mode='markers',
                            marker=dict(
            color=df2['Sharpe Ratio'],
            colorscale='RdYlGn',
            line=dict(
                color='black',
                width=1), showscale=True)))
    grafico2.update_xaxes(title_text = 'Volatilidade')
    grafico2.update_yaxes(title_text = 'Retorno')
    grafico2.update_layout(title_text = 'Fronteira Eficiente Markowitz')
    grafico2.update_layout(autosize=False, width=500, height=400)
      
    menor_vol = df2['Volatilidade'].min()
    maior_shp = df2['Sharpe Ratio'].max()
    
    portfolio_sharp = df2[df2['Sharpe Ratio'] == maior_shp]
    portfolio_vol = df2[df2['Volatilidade'] == menor_vol]

    table0 = portfolio_vol.T
    table0.reset_index(inplace=True)
    table0[table0.columns[1]] = (table0[table0.columns[1]]*100).round(2)
    table0[table0.columns[1]] = table0[table0.columns[1]].apply(lambda x: f'{x:.2f}%')
    table0.rename(columns={table0.columns[0]:'Ticker', table0.columns[1]:'Info'}, inplace=True)
    table0 = dbc.Table.from_dataframe(table0, bordered=True)

    table1 = portfolio_sharp.T
    table1.reset_index(inplace=True)
    table1[table1.columns[1]] = (table1[table1.columns[1]]*100).round(2)
    table1[table1.columns[1]] = table1[table1.columns[1]].apply(lambda x: f'{x:.2f}%')
    table1.rename(columns={table1.columns[0]:'Ticker', table1.columns[1]:'Info'}, inplace=True)
    table1 = dbc.Table.from_dataframe(table1, bordered=True)

    return [grafico0, grafico1, grafico2, table0, table1]

   
#server
if __name__ == '__main__':
    app.run_server(debug=True)