# import os, glob, time, json
# from itertools import product, chain
# import numpy as np
# import pandas as pd
# import geopandas as gpd
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import matplotlib
# import matplotlib.pyplot as plt
# import matplotlib.colors as colors
# from matplotlib import cm

# def PlotImportanceMap(importance, geojson, country_name, footnote):
#     eoname = importance['eoname'].unique()
#     n_eoname = len(eoname)
#     lead_month = importance[['lead','month']].drop_duplicates()
#     lead_month['monthL'] = lead_month['month'].apply(lambda x: pd.to_datetime('2000-%02d-01' % x).strftime('%b'))
#     months = lead_month['monthL']
#     leadmat = lead_month['lead'].values
#     # Feature Importance Maps ----------------------- #
#     rows, cols, scale = n_eoname, 5, 1.2
#     if country_name == 'Kenya': width, height, lims = 190*cols*scale, 220*rows*scale, np.array([33.8,41.9,-4.7,5.5]); cb_len=0.7
#     elif country_name == 'Somalia': width, height, lims = 180*cols*scale, 230*rows*scale, np.array([40.5,51,-2,12]); cb_len=0.7
#     elif country_name == 'Malawi': width, height, lims = 130*cols*scale, 250*rows*scale, np.array([32.6,35.92,-17.2,-9.3]); cb_len=0.6
#     elif country_name == 'Burkina Faso': width, height, lims = 210*cols*scale, 150*rows*scale, np.array([-5.52,2.41,9.4,15.1]); cb_len=0.8
#     else: raise ValueError('country_name is not correct.')
#     loc = list(product(range(1,rows+1),range(1,cols+1)))
#     fig = make_subplots(
#         rows=rows, cols=cols,
#         specs = [[{'type': 'choropleth'} for c in np.arange(cols)] for r in np.arange(rows)],
#         horizontal_spacing=0.005,
#         vertical_spacing=0.005,
#         subplot_titles = list(months.values),
#     )

#     for i, (name, lead) in enumerate(product(eoname, leadmat)):
#         sub = importance[(importance['eoname'] == name) & (importance['lead']==lead)]
#         htemp = 'FNID: %{location}<br>Admin1: %{customdata}<br>Admin2: %{text}<br>FI: %{z:.3f}<extra></extra>'
#         fig.add_trace(go.Choropleth(
#             locations=sub.fnid,
#             z = sub.value,
#             geojson=geojson,
#             featureidkey='properties.FNID',
#             marker_line_width=1,
#             marker_line_color='grey',
#             coloraxis = 'coloraxis1',
#             customdata=sub['ADMIN1'].tolist(),
#             text=sub['ADMIN2'].tolist(),
#             hovertemplate=htemp
#         ), row=loc[i][0], col=loc[i][1])
#     fig.update_geos(visible=False, resolution=50,
#                     showcountries=True, countrycolor="grey",
#                     lonaxis_range=lims[[0,1]],
#                     lataxis_range=lims[[2,3]],
#                     showframe=False,
#                    )
#     fig.update_layout(
#         width=width, height=height,
#         font = {'family':'arial','size':16, 'color':'black'},
#         margin={"r":0,"t":25,"l":25,"b":20},
#         coloraxis1=dict(
#             colorscale=px.colors.diverging.PiYG,
#             cmin=0,
#             cmax=0.5,
#             colorbar = dict(
#                 x=1,
#                 y=0.5,
#                 len=0.3,
#                 thickness=15,
#                 outlinewidth=1,
#                 title='Feature importance',
#                 title_side='right',
#             )
#         ),
#         dragmode=False
#     )
#     for annotation in fig.layout.annotations:
#         annotation.update(y=annotation.y-0.005, font_size=20)
#     for i, name in enumerate(eoname):
#         fig.add_annotation(text=name.upper(), xref="paper", yref="paper", x=-0.025, y=1-(i*0.25+0.12), yanchor="middle",showarrow=False, font_size=20, textangle=-90)
#     fig.add_annotation(
#         xref="paper", yref="paper",
#         x=0, y=-.02,
#         text=footnote,
#         align="right",
#         showarrow=False,
#         font = {'family':'arial','size':15, 'color':'dimgrey'},
#     )
#     return fig


# def PlotImportanceHeatmap(importance, footnote):
#     fnid_adminx = importance[['fnid','ADMINX']].drop_duplicates().set_index('fnid')['ADMINX']
#     eoname = importance['eoname'].unique()
#     lead_month = importance[['lead','month']].drop_duplicates()
#     lead_month['monthL'] = lead_month['month'].apply(lambda x: pd.to_datetime('2000-%02d-01' % x).strftime('%b'))
#     months = lead_month['monthL']
#     # Feature Importance Heatmap ----------------------- #
#     rows, cols = 1, len(eoname)
#     fig = make_subplots(
#         rows=rows, cols=cols,
#         shared_yaxes=True,
#         specs = [[{'type': 'Heatmap'} for c in np.arange(cols)] for r in np.arange(rows)],
#         horizontal_spacing=0.01,
#         subplot_titles = [t.upper() for t in eoname],
#     )
#     # Adding subplots
#     for i, name in enumerate(eoname):
#         temp = importance[importance['eoname']==name].pivot_table(index='fnid',columns='lead',values='value')
#         fnid_name = np.tile(temp.index.values[:,None],temp.shape[1])
#         fig.add_trace(go.Heatmap(z=temp,coloraxis='coloraxis1',customdata=fnid_name,
#                                  hovertemplate='FNID: %{customdata}<br>Name: %{y}<br>Lead: %{x}<br>Importance: %{z:.3f}<extra></extra>'),row=1,col=i+1)
#     # Layout updates
#     fig.update_layout(
#         plot_bgcolor='white',
#         hovermode='closest',
#         font = {'family':'arial','size':15, 'color':'black'},
#         margin={"r":0,"t":20,"l":0,"b":20},
#         height=600, width=850, 
#         yaxis=dict(
#             title='',
#             autorange='reversed',
#             dtick=1,
#             tickmode = 'array',
#             tickvals = np.arange(temp.shape[0]),
#             ticktext = [fnid_adminx[fnid] for fnid in temp.index],
#             tickfont = {'family':'arial','size':14, 'color':'black'},
#         ),
#         coloraxis1=dict(
#             colorscale='PiYG',
#             # colorscale='hot_r',
#             cmin=0,
#             cmax=0.5,
#             colorbar = dict(
#                 x=1.01,
#                 y=0.50,
#                 len=0.60,
#                 thickness=15,
#                 outlinewidth=1,
#                 title='Feature importance',
#                 title_side='right'
#             )
#         ),
#     )
#     fig.update_xaxes(
#         title='',
#         dtick=1,
#         autorange='reversed',
#         tickmode = 'array', 
#         tickvals = np.arange(0,7),
#         ticktext=list(chain(*[[m[0]] for m in months]))[::-1]
#     )
#     fig.add_annotation(
#             xref="paper", yref="paper",
#             x=0, y=-.08,
#             text=footnote,
#             align="right",
#             showarrow=False,
#             font = {'family':'arial','size':15, 'color':'dimgrey'},
#         )
#     return fig



# def PlotScoreMap(score, geojson, country_name, footnote):
#     lead_month = score[['lead','month']].drop_duplicates()
#     lead_month['monthL'] = lead_month['month'].apply(lambda x: pd.to_datetime('2000-%02d-01' % x).strftime('%b'))
#     months = list(lead_month['monthL'].values)
#     # Skill score maps ----------------------- #
#     rows, cols, scale = 2, 5, 1.2
#     if country_name == 'Kenya': width, height, lims = 180*cols*scale, 220*rows*scale, np.array([33.8,41.9,-4.7,5.5]); cb_len=0.7
#     elif country_name == 'Somalia': width, height, lims = 180*cols*scale, 235*rows*scale, np.array([40.5,51,-2,12]); cb_len=0.7
#     elif country_name == 'Malawi': width, height, lims = 115*cols*scale, 240*rows*scale, np.array([32.6,35.92,-17.2,-9.3]); cb_len=0.6
#     elif country_name == 'Burkina Faso': width, height, lims = 240*cols*scale, 180*rows*scale, np.array([-5.52,2.41,9.4,15.1]); cb_len=0.8
#     else: raise ValueError('country_name is not correct.')
#     loc = list(product(range(1,rows+1),range(1,cols+1)))
#     fig = make_subplots(
#         rows=rows, cols=cols,
#         specs = [[{'type': 'choropleth'} for c in np.arange(cols)] for r in np.arange(rows)],
#         horizontal_spacing=0.005,
#         vertical_spacing=0.005,
#         subplot_titles = months,
#     )
#     comb = product(['yield_nse', 'yield_mape'],[5,4,3,2,1])
#     for i, (name,lead) in enumerate(comb):
#         sub = score[(score['variable'] == name) & (score['lead']==lead)]
#         if name == 'yield_nse':
#             caxis = 'coloraxis1'
#             htemp = 'FNID: %{location}<br>Admin1: %{customdata}<br>Admin2: %{text}<br>NSE: %{z:.3f}<extra></extra>'
#         elif name == 'yield_mape':
#             caxis = 'coloraxis2'
#             htemp = 'FNID: %{location}<br>Admin1: %{customdata}<br>Admin2: %{text}<br>MAPE: %{z:.0f}%<extra></extra>'
#         fig.add_trace(go.Choropleth(
#             locations=sub.fnid,
#             z = sub.value,
#             geojson=geojson,
#             featureidkey='properties.FNID',
#             marker_line_width=1,
#             marker_line_color='grey',
#             coloraxis = caxis,
#             customdata=sub['ADMIN1'].tolist(),
#             text=sub['ADMIN2'].tolist(),
#             hovertemplate=htemp
#         ), row=loc[i][0], col=loc[i][1])

#     fig.update_geos(visible=False, resolution=50,
#                     showcountries=True, countrycolor="grey",
#                     lonaxis_range=lims[[0,1]],
#                     lataxis_range=lims[[2,3]],
#                     showframe=False,
#                    )

#     fig.update_layout(
#         width=width, height=height,
#         margin={"r":0,"t":20,"l":0,"b":20},
#         font = {'family':'arial','size':16, 'color':'black'},
#         coloraxis1=dict(
#             colorscale=px.colors.sequential.Cividis,
#             reversescale=True,
#             cmin=0,
#             cmax=1.0,
#             colorbar = dict(
#                 x=1,
#                 y=0.75,
#                 len=0.5,
#                 thickness=15,
#                 outlinewidth=1,
#                 title='NSE',
#                 title_side='right',
#             )
#         ),
#         coloraxis2=dict(
#             colorscale=px.colors.sequential.Viridis,
#             cmin=0,
#             cmax=50,
#             colorbar = dict(
#                 x=1,
#                 y=0.25,
#                 len=0.5,
#                 thickness=15,
#                 outlinewidth=1,
#                 title='MAPE',
#                 title_side='right',
#                 ticksuffix='%'
#             )
#         ),
#         dragmode=False
#     )
#     fig.add_annotation(
#         xref="paper", yref="paper",
#         x=0, y=-.047,
#         text=footnote,
#         align="right",
#         showarrow=False,
#         font = {'family':'arial','size':15, 'color':'dimgrey'},
#     )
#     return fig

    
# def PlotScoreHeatmap(score, footnote):
#     fnid_adminx = score[['fnid','ADMINX']].drop_duplicates().set_index('fnid')['ADMINX']
#     lead_month = score[['lead','month']].drop_duplicates()
#     lead_month['monthL'] = lead_month['month'].apply(lambda x: pd.to_datetime('2000-%02d-01' % x).strftime('%b'))
#     months = lead_month['monthL']
#     # Skill Score Heatmap ------------- #
#     nse = score[score['variable']=='yield_nse'].pivot_table(index='fnid',columns='lead',values='value')
#     mape = score[score['variable']=='yield_mape'].pivot_table(index='fnid',columns='lead',values='value')
#     fnid_name = np.tile(nse.index.values[:,None],nse.shape[1])
#     rows, cols = 1, 2
#     fig = make_subplots(
#         rows=rows, cols=cols,
#         shared_yaxes=True,
#         specs = [[{'type': 'Heatmap'} for c in np.arange(cols)] for r in np.arange(rows)],
#         horizontal_spacing=0.01,
#         subplot_titles = ['NSE','MAPE'],
#     )
#     for i in fig['layout']['annotations']: i['font'] = dict(family='arial', size=16, color='black')
#     fig.add_trace(go.Heatmap(z=nse,coloraxis='coloraxis1',customdata=fnid_name,
#                              hovertemplate='FNID: %{customdata}<br>Name: %{y}<br>Lead: %{x}<br>NSE: %{z:.3f}<extra></extra>'),row=1,col=1)
#     fig.add_trace(go.Heatmap(z=mape,coloraxis='coloraxis2',customdata=fnid_name,
#                              hovertemplate='FNID: %{customdata}<br>Name: %{y}<br>Lead: %{x}<br>MAPE: %{z:.0f}%<extra></extra>'),row=1,col=2)
#     fig.update_layout(
#         plot_bgcolor='white',
#         hovermode='closest',
#         font = {'family':'arial','size':15, 'color':'black'},
#         title_x=0.02, title_y=0.99,
#         margin={"r":0,"t":20,"l":0,"b":20},
#         height=700, width=800, 
#         yaxis=dict(
#             title='',
#             autorange='reversed',
#             dtick=1,
#             tickmode = 'array',
#             tickvals = np.arange(nse.shape[0]),
#             ticktext = [fnid_adminx[fnid] for fnid in nse.index],
#             tickfont = {'family':'arial','size':14, 'color':'black'},
#         ),
#         coloraxis1=dict(
#             colorscale='Cividis_r',
#             cmin=0,
#             cmax=1.0,
#             colorbar = dict(
#                 x=1.01,
#                 y=0.75,
#                 len=0.5,
#                 thickness=15,
#                 outlinewidth=1,
#                 title='NSE',
#                 title_font = {'family':'arial','size':16, 'color':'black'},
#                 title_side='right',
#             )
#         ),
#         coloraxis2=dict(
#             colorscale='Viridis',
#             cmin=0,
#             cmax=50,
#             colorbar = dict(
#                 x=1.01,
#                 y=0.25,
#                 len=0.5,
#                 thickness=15,
#                 outlinewidth=1,
#                 title='MAPE',
#                 title_font = {'family':'arial','size':16, 'color':'black'},
#                 title_side='right',
#                 ticksuffix='%',
#             )
#         )
#     )
#     fig.update_xaxes(
#         title='',
#         dtick=1,
#         autorange='reversed',
#         tickmode = 'array',
#         tickvals = np.arange(0,5),
#         ticktext = months[::-1],
#     )
#     fig.add_annotation(
#         xref="paper", yref="paper",
#         x=0, y=-.065,
#         text=footnote,
#         align="right",
#         showarrow=False,
#         font = {'family':'arial','size':15, 'color':'dimgrey'},
#     )
#     return fig


# def PlotForecastMap(df, geojson, country_name, footnote, ftype='forecast'):
#     scale = 2
#     rgb = matplotlib_to_plotly('bwr', 255)
#     # Forecast maps ----------------------- #
#     if country_name == 'Kenya': width, height, lims = 210*scale, 220*scale, np.array([33.8,41.9,-4.7,5.5]); cb_len=0.7
#     elif country_name == 'Somalia': width, height, lims = 210*scale, 235*scale, np.array([40.5,51,-2,12]); cb_len=0.7
#     elif country_name == 'Malawi': width, height, lims = 140*scale, 240*scale, np.array([32.6,35.92,-17.2,-9.3]); cb_len=0.6
#     elif country_name == 'Burkina Faso': width, height, lims = 270*scale, 180*scale, np.array([-5.52,2.41,9.4,15.1]); cb_len=0.8
#     else: raise ValueError('country_name is not correct.')
#     if ftype == 'forecast':
#         cmin, cmax = 40, 160
#     elif ftype == 'hindcast':
#         cmin, cmax = -60, 60
#     else:
#         raise ValueError("Invalid ftype value.")
    
#     htemp = 'FNID: %{location}<br>Admin1: %{customdata}<br>Admin2: %{text}<br>Percent: %{z:.0f}%<extra></extra>'
#     fig = go.Figure(go.Choropleth(
#         locations=df.fnid,
#         z = df.value,
#         geojson=geojson,
#         featureidkey='properties.FNID',
#         marker_line_width=1,
#         marker_line_color='grey',
#         coloraxis = 'coloraxis1',
#         customdata=df['ADMIN1'].tolist(),
#         text=df['ADMIN2'].tolist(),
#         hovertemplate=htemp
#     ))
#     fig.update_geos(visible=False, resolution=50,
#                     showcountries=True, countrycolor="grey",
#                     lonaxis_range=lims[[0,1]],
#                     lataxis_range=lims[[2,3]],
#                     showframe=False,
#                    )
#     fig.update_layout(
#         width=width, height=height,
#         margin={"r":0,"t":0,"l":0,"b":20},
#         font_size=20,
#         coloraxis1=dict(
#             colorscale = rgb,
#             reversescale=True,
#             cmin=cmin,
#             cmax=cmax,
#             colorbar = dict(
#                 x=1,
#                 y=0.50,
#                 len=cb_len,
#                 thickness=15,
#                 outlinewidth=1,
#                 title='',
#                 title_side='right',
#                 ticksuffix='%',
#                 tickfont = {'family':'arial','size':18, 'color':'black'},
#             )
#         ),
#         dragmode=False
#     )
#     for annotation in fig.layout.annotations:
#         annotation.update(y=annotation.y-0.01)
#     fig.add_annotation(
#         x=0, y=-0.05,
#         text=footnote,
#         align="right",
#         showarrow=False,
#         font = {'family':'arial','size':15, 'color':'dimgrey'},
#     )
#     # fig.show()
#     return fig


# def matplotlib_to_plotly(name, pl_entries):
#     # Revised from https://plotly.com/python/v3/matplotlib-colorscales/
#     cmap = matplotlib.cm.get_cmap(name)
#     rgb = []
#     norm = matplotlib.colors.Normalize(vmin=0, vmax=255)
#     for i in range(0, 255):
#         k = matplotlib.colors.colorConverter.to_rgb(cmap(norm(i)))
#         rgb.append(k)
#     h = 1.0/(pl_entries-1)
#     pl_colorscale = []
#     for k in range(pl_entries):
#         C = list(map(np.uint8, np.array(cmap(k*h)[:3])*255))
#         pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])
#     return pl_colorscale



import os, glob, time, json
from itertools import product, chain
import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm

def PlotImportanceMap(importance, geojson, country_name, footnote):
    eoname = importance['eoname'].unique()
    n_eoname = len(eoname)
    lead_month = importance[['lead','month']].drop_duplicates()
    lead_month['monthL'] = lead_month['month'].apply(lambda x: pd.to_datetime('2000-%02d-01' % x).strftime('%b'))
    months = lead_month['monthL']
    leadmat = lead_month['lead'].values
    # Feature Importance Maps ----------------------- #
    rows, cols, scale = n_eoname, 5, 1.2
    if country_name == 'Kenya': width, height, lims = 190*cols*scale, 220*rows*scale, np.array([33.8,41.9,-4.7,5.5]); cb_len=0.7
    elif country_name == 'Somalia': width, height, lims = 180*cols*scale, 230*rows*scale, np.array([40.5,51,-2,12]); cb_len=0.7
    elif country_name == 'Malawi': width, height, lims = 130*cols*scale, 250*rows*scale, np.array([32.6,35.92,-17.2,-9.3]); cb_len=0.6
    elif country_name == 'Burkina Faso': width, height, lims = 210*cols*scale, 150*rows*scale, np.array([-5.52,2.41,9.4,15.1]); cb_len=0.8
    else: raise ValueError('country_name is not correct.')
    loc = list(product(range(1,rows+1),range(1,cols+1)))
    fig = make_subplots(
        rows=rows, cols=cols,
        specs = [[{'type': 'choropleth'} for c in np.arange(cols)] for r in np.arange(rows)],
        horizontal_spacing=0.005,
        vertical_spacing=0.005,
        subplot_titles = list(months.values),
    )

    for i, (name, lead) in enumerate(product(eoname, leadmat)):
        sub = importance[(importance['eoname'] == name) & (importance['lead']==lead)]
        htemp = 'FNID: %{location}<br>Admin1: %{customdata}<br>Admin2: %{text}<br>FI: %{z:.3f}<extra></extra>'
        fig.add_trace(go.Choropleth(
            locations=sub.fnid,
            z = sub.value,
            geojson=geojson,
            featureidkey='properties.FNID',
            marker_line_width=1,
            marker_line_color='grey',
            coloraxis = 'coloraxis1',
            customdata=sub['ADMIN1'].tolist(),
            text=sub['ADMIN2'].tolist(),
            hovertemplate=htemp
        ), row=loc[i][0], col=loc[i][1])
    fig.update_geos(visible=False, resolution=50,
                    showcountries=True, countrycolor="grey",
                    lonaxis_range=lims[[0,1]],
                    lataxis_range=lims[[2,3]],
                    showframe=False,
                   )
    fig.update_layout(
        width=width, height=height,
        font = {'family':'arial','size':16, 'color':'black'},
        margin={"r":0,"t":25,"l":25,"b":20},
        coloraxis1=dict(
            colorscale=px.colors.diverging.PiYG,
            cmin=0,
            cmax=0.5,
            colorbar = dict(
                x=1,
                y=0.5,
                len=0.3,
                thickness=15,
                outlinewidth=1,
                title='Feature importance',
                title_side='right',
            )
        ),
        dragmode=False
    )
    for annotation in fig.layout.annotations:
        annotation.update(y=annotation.y-0.005, font_size=20)
    for i, name in enumerate(eoname):
        fig.add_annotation(text=name.upper(), xref="paper", yref="paper", x=-0.025, y=1-(i*0.25+0.12), yanchor="middle",showarrow=False, font_size=20, textangle=-90)
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0, y=-.02,
        text=footnote,
        align="right",
        showarrow=False,
        font = {'family':'arial','size':15, 'color':'dimgrey'},
    )
    return fig


def PlotImportanceHeatmap(importance, footnote):
    fnid_adminx = importance[['fnid','ADMINX']].drop_duplicates().set_index('fnid')['ADMINX']
    eoname = importance['eoname'].unique()
    lead_month = importance[['lead','month']].drop_duplicates()
    lead_month['monthL'] = lead_month['month'].apply(lambda x: pd.to_datetime('2000-%02d-01' % x).strftime('%b'))
    months = lead_month['monthL']
    # Feature Importance Heatmap ----------------------- #
    rows, cols = 1, len(eoname)
    fig = make_subplots(
        rows=rows, cols=cols,
        shared_yaxes=True,
        specs = [[{'type': 'Heatmap'} for c in np.arange(cols)] for r in np.arange(rows)],
        horizontal_spacing=0.01,
        subplot_titles = [t.upper() for t in eoname],
    )
    # Adding subplots
    for i, name in enumerate(eoname):
        temp = importance[importance['eoname']==name].pivot_table(index='fnid',columns='lead',values='value')
        fnid_name = np.tile(temp.index.values[:,None],temp.shape[1])
        fig.add_trace(go.Heatmap(z=temp,coloraxis='coloraxis1',customdata=fnid_name,
                                 hovertemplate='FNID: %{customdata}<br>Name: %{y}<br>Lead: %{x}<br>Importance: %{z:.3f}<extra></extra>'),row=1,col=i+1)
    # Layout updates
    fig.update_layout(
        plot_bgcolor='white',
        hovermode='closest',
        font = {'family':'arial','size':15, 'color':'black'},
        margin={"r":0,"t":20,"l":0,"b":20},
        height=600, width=850, 
        yaxis=dict(
            title='',
            autorange='reversed',
            dtick=1,
            tickmode = 'array',
            tickvals = np.arange(temp.shape[0]),
            ticktext = [fnid_adminx[fnid] for fnid in temp.index],
            tickfont = {'family':'arial','size':14, 'color':'black'},
        ),
        coloraxis1=dict(
            colorscale='PiYG',
            # colorscale='hot_r',
            cmin=0,
            cmax=0.5,
            colorbar = dict(
                x=1.01,
                y=0.50,
                len=0.60,
                thickness=15,
                outlinewidth=1,
                title='Feature importance',
                title_side='right'
            )
        ),
    )
    fig.update_xaxes(
        title='',
        dtick=1,
        autorange='reversed',
        tickmode = 'array', 
        tickvals = np.arange(0,7),
        ticktext=list(chain(*[[m[0]] for m in months]))[::-1]
    )
    fig.update_yaxes(
        title='',
        autorange='reversed',
        dtick=1,
        tickmode = 'array',
        tickvals = np.arange(temp.shape[0]),
        ticktext = [fnid_adminx[fnid] for fnid in temp.index],
        tickfont = {'family':'arial','size':14, 'color':'black'},
    )
    fig.add_annotation(
            xref="paper", yref="paper",
            x=0, y=-.08,
            text=footnote,
            align="right",
            showarrow=False,
            font = {'family':'arial','size':15, 'color':'dimgrey'},
        )
    return fig



def PlotScoreMap(score, geojson, country_name, footnote):
    lead_month = score[['lead','month']].drop_duplicates()
    lead_month['monthL'] = lead_month['month'].apply(lambda x: pd.to_datetime('2000-%02d-01' % x).strftime('%b'))
    months = list(lead_month['monthL'].values)
    # Colormap (with negative) -------- #
    cs = px.colors.sequential.Cividis_r
    ccs=[
        (0,'#ffedfe'),(0.000001,'#ffedfe'),
        (0.000001,cs[0]),
        (0.1/0.9,cs[1]),(0.2/0.9,cs[2]),(0.3/0.9,cs[3]),
        (0.4/0.9,cs[4]),(0.5/0.9,cs[5]),(0.6/0.9,cs[6]),
        (0.7/0.9,cs[7]),(0.8/0.9,cs[8]),(0.9/0.9,cs[9])
    ]
    # Skill score maps ----------------------- #
    rows, cols, scale = 2, 5, 1.2
    if country_name == 'Kenya': width, height, lims = 180*cols*scale, 220*rows*scale, np.array([33.8,41.9,-4.7,5.5]); cb_len=0.7
    elif country_name == 'Somalia': width, height, lims = 180*cols*scale, 235*rows*scale, np.array([40.5,51,-2,12]); cb_len=0.7
    elif country_name == 'Malawi': width, height, lims = 115*cols*scale, 240*rows*scale, np.array([32.6,35.92,-17.2,-9.3]); cb_len=0.6
    elif country_name == 'Burkina Faso': width, height, lims = 240*cols*scale, 180*rows*scale, np.array([-5.52,2.41,9.4,15.1]); cb_len=0.8
    else: raise ValueError('country_name is not correct.')
    loc = list(product(range(1,rows+1),range(1,cols+1)))
    fig = make_subplots(
        rows=rows, cols=cols,
        specs = [[{'type': 'choropleth'} for c in np.arange(cols)] for r in np.arange(rows)],
        horizontal_spacing=0.005,
        vertical_spacing=0.005,
        subplot_titles = months,
    )
    comb = product(['yield_nse', 'yield_mape'],[5,4,3,2,1])
    for i, (name,lead) in enumerate(comb):
        sub = score[(score['variable'] == name) & (score['lead']==lead)]
        if name == 'yield_nse':
            caxis = 'coloraxis1'
            htemp = 'FNID: %{location}<br>Admin1: %{customdata}<br>Admin2: %{text}<br>NSE: %{z:.3f}<extra></extra>'
        elif name == 'yield_mape':
            caxis = 'coloraxis2'
            htemp = 'FNID: %{location}<br>Admin1: %{customdata}<br>Admin2: %{text}<br>MAPE: %{z:.0f}%<extra></extra>'
        fig.add_trace(go.Choropleth(
            locations=sub.fnid,
            z = sub.value,
            geojson=geojson,
            featureidkey='properties.FNID',
            marker_line_width=1,
            marker_line_color='grey',
            coloraxis = caxis,
            customdata=sub['ADMIN1'].tolist(),
            text=sub['ADMIN2'].tolist(),
            hovertemplate=htemp
        ), row=loc[i][0], col=loc[i][1])

    fig.update_geos(visible=False, resolution=50,
                    showcountries=True, countrycolor="grey",
                    lonaxis_range=lims[[0,1]],
                    lataxis_range=lims[[2,3]],
                    showframe=False,
                   )

    fig.update_layout(
        width=width, height=height,
        margin={"r":0,"t":20,"l":0,"b":20},
        font = {'family':'arial','size':16, 'color':'black'},
        coloraxis1=dict(
            # colorscale=px.colors.sequential.Cividis_r,
            colorscale=ccs,
            reversescale=False,
            cmin=0,
            cmax=1.0,
            colorbar = dict(
                x=1,
                y=0.75,
                len=0.5,
                thickness=15,
                outlinewidth=1,
                title='NSE',
                title_side='right',
            )
        ),
        coloraxis2=dict(
            colorscale=px.colors.sequential.Viridis,
            cmin=0,
            cmax=50,
            colorbar = dict(
                x=1,
                y=0.25,
                len=0.5,
                thickness=15,
                outlinewidth=1,
                title='MAPE',
                title_side='right',
                ticksuffix='%'
            )
        ),
        dragmode=False
    )
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0, y=-.047,
        text=footnote,
        align="right",
        showarrow=False,
        font = {'family':'arial','size':15, 'color':'dimgrey'},
    )
    fig.add_annotation(
        xref="paper", yref="paper",
        x=1., y=-.04,
        text='(Light pink denotes negative NSE values)',
        align="right",
        showarrow=False,
        font = {'family':'arial','size':15, 'color':'dimgrey'},
    )
    return fig

    
def PlotScoreHeatmap(score, footnote):
    fnid_adminx = score[['fnid','ADMINX']].drop_duplicates().set_index('fnid')['ADMINX']
    lead_month = score[['lead','month']].drop_duplicates()
    lead_month['monthL'] = lead_month['month'].apply(lambda x: pd.to_datetime('2000-%02d-01' % x).strftime('%b'))
    months = lead_month['monthL']
    # Colormap (with negative) -------- #
    cs = px.colors.sequential.Cividis_r
    ccs=[
        (0,'#ffedfe'),(0.000001,'#ffedfe'),
        (0.000001,cs[0]),
        (0.1/0.9,cs[1]),(0.2/0.9,cs[2]),(0.3/0.9,cs[3]),
        (0.4/0.9,cs[4]),(0.5/0.9,cs[5]),(0.6/0.9,cs[6]),
        (0.7/0.9,cs[7]),(0.8/0.9,cs[8]),(0.9/0.9,cs[9])
    ]
    # --------------------------------- # 
    # Skill Score Heatmap ------------- #
    nse = score[score['variable']=='yield_nse'].pivot_table(index='fnid',columns='lead',values='value')
    mape = score[score['variable']=='yield_mape'].pivot_table(index='fnid',columns='lead',values='value')
    fnid_name = np.tile(nse.index.values[:,None],nse.shape[1])
    rows, cols = 1, 2
    fig = make_subplots(
        rows=rows, cols=cols,
        shared_yaxes=True,
        specs = [[{'type': 'Heatmap'} for c in np.arange(cols)] for r in np.arange(rows)],
        horizontal_spacing=0.01,
        subplot_titles = ['NSE','MAPE'],
    )
    for i in fig['layout']['annotations']: i['font'] = dict(family='arial', size=16, color='black')
    fig.add_trace(go.Heatmap(z=nse,coloraxis='coloraxis1',customdata=fnid_name,
                             hovertemplate='FNID: %{customdata}<br>Name: %{y}<br>Lead: %{x}<br>NSE: %{z:.3f}<extra></extra>'),row=1,col=1)
    fig.add_trace(go.Heatmap(z=mape,coloraxis='coloraxis2',customdata=fnid_name,
                             hovertemplate='FNID: %{customdata}<br>Name: %{y}<br>Lead: %{x}<br>MAPE: %{z:.0f}%<extra></extra>'),row=1,col=2)
    fig.update_layout(
        plot_bgcolor='white',
        hovermode='closest',
        font = {'family':'arial','size':15, 'color':'black'},
        title_x=0.02, title_y=0.99,
        margin={"r":0,"t":20,"l":0,"b":20},
        height=700, width=800, 
        yaxis=dict(
            title='',
            autorange='reversed',
            dtick=1,
            tickmode = 'array',
            tickvals = np.arange(nse.shape[0]),
            ticktext = [fnid_adminx[fnid] for fnid in nse.index],
            tickfont = {'family':'arial','size':14, 'color':'black'},
        ),
        coloraxis1=dict(
            colorscale=ccs,
            cmin=0,
            cmax=1.0,
            colorbar = dict(
                x=1.01,
                y=0.75,
                len=0.5,
                thickness=15,
                outlinewidth=1,
                title='NSE',
                title_font = {'family':'arial','size':16, 'color':'black'},
                title_side='right',
            )
        ),
        coloraxis2=dict(
            colorscale='Viridis',
            cmin=0,
            cmax=50,
            colorbar = dict(
                x=1.01,
                y=0.25,
                len=0.5,
                thickness=15,
                outlinewidth=1,
                title='MAPE',
                title_font = {'family':'arial','size':16, 'color':'black'},
                title_side='right',
                ticksuffix='%',
            )
        )
    )
    fig.update_xaxes(
        title='',
        dtick=1,
        autorange='reversed',
        tickmode = 'array',
        tickvals = np.arange(0,5),
        ticktext = months[::-1],
    )
    fig.update_yaxes(
        title='',
        autorange='reversed',
        dtick=1,
        tickmode = 'array',
        tickvals = np.arange(nse.shape[0]),
        ticktext = [fnid_adminx[fnid] for fnid in nse.index],
        tickfont = {'family':'arial','size':14, 'color':'black'},
    )
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0, y=-.065,
        text=footnote,
        align="right",
        showarrow=False,
        font = {'family':'arial','size':15, 'color':'dimgrey'},
    )
    fig.add_annotation(
        xref="paper", yref="paper",
        x=1., y=-.065,
        text='(Light pink denotes negative NSE values)',
        align="right",
        showarrow=False,
        font = {'family':'arial','size':15, 'color':'dimgrey'},
    )
    return fig


def PlotForecastMap(df, geojson, country_name, footnote, ftype='forecast'):
    scale = 2
    rgb = matplotlib_to_plotly('bwr', 255)
    # Forecast maps ----------------------- #
    if country_name == 'Kenya': width, height, lims = 210*scale, 220*scale, np.array([33.8,41.9,-4.7,5.5]); cb_len=0.7
    elif country_name == 'Somalia': width, height, lims = 210*scale, 235*scale, np.array([40.5,51,-2,12]); cb_len=0.7
    elif country_name == 'Malawi': width, height, lims = 140*scale, 240*scale, np.array([32.6,35.92,-17.2,-9.3]); cb_len=0.6
    elif country_name == 'Burkina Faso': width, height, lims = 270*scale, 180*scale, np.array([-5.52,2.41,9.4,15.1]); cb_len=0.8
    else: raise ValueError('country_name is not correct.')
    if ftype == 'forecast':
        cmin, cmax = 40, 160
    elif ftype == 'hindcast':
        cmin, cmax = -60, 60
    else:
        raise ValueError("Invalid ftype value.")
    
    htemp = 'FNID: %{location}<br>Admin1: %{customdata}<br>Admin2: %{text}<br>Percent: %{z:.0f}%<extra></extra>'
    fig = go.Figure(go.Choropleth(
        locations=df.fnid,
        z = df.value,
        geojson=geojson,
        featureidkey='properties.FNID',
        marker_line_width=1,
        marker_line_color='grey',
        coloraxis = 'coloraxis1',
        customdata=df['ADMIN1'].tolist(),
        text=df['ADMIN2'].tolist(),
        hovertemplate=htemp
    ))
    fig.update_geos(visible=False, resolution=50,
                    showcountries=True, countrycolor="grey",
                    lonaxis_range=lims[[0,1]],
                    lataxis_range=lims[[2,3]],
                    showframe=False,
                   )
    fig.update_layout(
        width=width, height=height,
        margin={"r":0,"t":0,"l":0,"b":20},
        font_size=20,
        coloraxis1=dict(
            colorscale = rgb,
            reversescale=True,
            cmin=cmin,
            cmax=cmax,
            colorbar = dict(
                x=1,
                y=0.50,
                len=cb_len,
                thickness=15,
                outlinewidth=1,
                title='',
                title_side='right',
                ticksuffix='%',
                tickfont = {'family':'arial','size':18, 'color':'black'},
            )
        ),
        dragmode=False
    )
    for annotation in fig.layout.annotations:
        annotation.update(y=annotation.y-0.01)
    fig.add_annotation(
        x=0, y=-0.05,
        text=footnote,
        align="right",
        showarrow=False,
        font = {'family':'arial','size':15, 'color':'dimgrey'},
    )
    # fig.show()
    return fig


def matplotlib_to_plotly(name, pl_entries):
    # Revised from https://plotly.com/python/v3/matplotlib-colorscales/
    cmap = matplotlib.cm.get_cmap(name)
    rgb = []
    norm = matplotlib.colors.Normalize(vmin=0, vmax=255)
    for i in range(0, 255):
        k = matplotlib.colors.colorConverter.to_rgb(cmap(norm(i)))
        rgb.append(k)
    h = 1.0/(pl_entries-1)
    pl_colorscale = []
    for k in range(pl_entries):
        C = list(map(np.uint8, np.array(cmap(k*h)[:3])*255))
        pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])
    return pl_colorscale