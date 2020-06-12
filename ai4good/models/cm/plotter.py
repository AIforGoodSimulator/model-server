import numpy as np
from math import ceil, floor
import plotly.graph_objects as go
from ai4good.models.model import ModelResult


########################################################################################################################
def population_format(num,dp=0):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    if dp==0 and not (num/100<1 and magnitude>=1): # unless force to be dp=1... return 0 dp if (K or M or B and number >10)
        return '%.0f%s' % (num, ['', 'K', 'M', 'B'][magnitude])
    else:
        return '%.1f%s' % (num, ['', 'K', 'M', 'B'][magnitude])


# fill_cols = ['rgba(50,50,50,0.2)','rgba(50,50,50,0.2)','rgba(50,50,50,0.2)','rgba(50,0,0,0.4)']
########################################################################################################################
def figure_generator(sols, params, cats_to_plot):
    population_plot = params.population
    categories = params.categories

    if len(cats_to_plot)==0:
        cats_to_plot=['I']

    font_size = 13

    lines_to_plot = []

    xx = sols[0]['t']

    for sol in sols:
        for name in categories.keys():
            if name in cats_to_plot:
                y_plot = np.asarray(100*sol['y_plot'][categories[name]['index']])
                # y_plot = np.asarray(100*sol['y_plot'][categories['A']['index']]) - np.asarray(100*sol['y_plot'][categories['I']['index']])


                
                line =  {'x': xx, 'y': y_plot,
                        'hovertemplate': '%{y:.2f}%, %{text}',
                        'text': [population_format(i*population_plot/100) for i in y_plot],
                        'line': {'color': str(categories[name]['colour'])},
                        'legendgroup': name,
                        'name': categories[name]['longname']}
                lines_to_plot.append(line)
         
         
         

    ymax = 0
    for line in lines_to_plot:
        ymax = max(ymax,max(line['y']))


    yax = dict(range= [0,min(1.1*ymax,100)])
    ##

    lines_to_plot.append(
    dict(
        type='scatter',
        x = [0,xx[-1]],
        y = [ 0, population_plot],
        yaxis="y2",
        opacity=0,
        hoverinfo = 'skip',
        showlegend=False
    ))


    yy2 = [0]
    for i in range(8):
        yy2.append(10**(i-5))
        yy2.append(2*10**(i-5))
        yy2.append(5*10**(i-5))

    yy = [i for i in yy2]

    pop_vec_lin = np.linspace(0,yy2[1],11)

    for i in range(len(yy)-1):
        if yax['range'][1]>yy[i] and yax['range'][1] <= yy[i+1]:
            pop_vec_lin = np.linspace(0,yy2[i+1],11)

    vec = [i*(population_plot) for i in pop_vec_lin]

    log_bottom = -8
    log_range = [log_bottom,np.log10(yax['range'][1])]

    pop_vec_log_intermediate = np.linspace(log_range[0],ceil(np.log10(pop_vec_lin[-1])), 1+ ceil(np.log10(pop_vec_lin[-1])-log_range[0]) )

    pop_log_vec = [10**(i) for i in pop_vec_log_intermediate]
    vec2 = [i*(population_plot) for i in pop_log_vec]

    shapes=[]
    annots=[]

    # if control_time[0]!=control_time[1] and not no_control:
    #     shapes.append(dict(
    #             # filled Blue Control Rectangle
    #             type="rect",
    #             x0= control_time[0],
    #             y0=0,
    #             x1= control_time[1],
    #             y1= yax['range'][1],
    #             line=dict(
    #                 color="LightSkyBlue",
    #                 width=0,
    #             ),
    #             fillcolor="LightSkyBlue",
    #             opacity= 0.15
    #         ))

    #     annots.append(dict(
    #             x  = 0.5*(control_time[0] + control_time[1]),
    #             y  = 0.5,
    #             text="<b>Control<br>" + "<b> In <br>" + "<b> Place",
    #             textangle=0,
    #             font=dict(
    #                 size= font_size*(30/24),
    #                 color="blue"
    #             ),
    #             showarrow=False,
    #             opacity=0.4,
    #             xshift= 0,
    #             xref = 'x',
    #             yref = 'paper',
    #     ))


    layout = go.Layout(
                    template="simple_white",
                    shapes=shapes,
                    annotations=annots,
                    font = dict(size= font_size), #'12em'),
                    margin=dict(t=5, b=5, l=10, r=10,pad=15),
                    hovermode='x',
                    xaxis= dict(
                            title='Days',
                            
                            automargin=True,
                            hoverformat='.0f',
                    ),
                    yaxis= dict(mirror= True,
                            title='Percentage of Total Population',
                            range= yax['range'],
                            
                            automargin=True,
                            type = 'linear'
                    ),
                        updatemenus = [dict(
                                                buttons=list([
                                                dict(
                                                    args=[{"yaxis": {'title': 'Percentage of Total Population', 'type': 'linear', 'range': yax['range'], 'automargin': True},
                                                    "yaxis2": {'title': 'Population','type': 'linear', 'overlaying': 'y1', 'range': yax['range'], 'ticktext': [population_format(0.01*vec[i]) for i in range(len(pop_vec_lin))], 'tickvals': [i for i in  pop_vec_lin],'automargin': True,'side':'right'}
                                                    }], # tickformat
                                                    label="Linear",
                                                    method="relayout"
                                                ),
                                                dict(
                                                    args=[{"yaxis": {'title': 'Percentage of Total Population', 'type': 'log', 'range': log_range,'automargin': True},
                                                    "yaxis2": {'title': 'Population','type': 'log', 'overlaying': 'y1', 'range': log_range, 'ticktext': [population_format(0.01*vec2[i]) for i in range(len(pop_log_vec))], 'tickvals': [i for i in  pop_log_vec],'automargin': True,'side':'right'}
                                                    }], # 'tickformat': yax_form_log,
                                                    label="Logarithmic",
                                                    method="relayout"
                                                )
                                        ]),
                                        x= 0.5,
                                        xanchor="right",
                                        pad={"r": 5, "t": 30, "b": 10, "l": 5},
                                        active=0,
                                        y=-0.13,
                                        showactive=True,
                                        direction='up',
                                        yanchor="top"
                                        )],
                                        legend = dict(
                                                        font=dict(size=font_size*(20/24)),
                                                        x = 0.5,
                                                        y = 1.03,
                                                        xanchor= 'center',
                                                        yanchor= 'bottom'
                                                    ),
                                        legend_orientation  = 'h',
                                        legend_title        = '<b> Key </b>',
                                        yaxis2 = dict(
                                                        title = 'Population',
                                                        overlaying='y1',
                                                        
                                                        range = yax['range'],
                                                        side='right',
                                                        ticktext = [population_format(0.01*vec[i]) for i in range(len(pop_vec_lin))],
                                                        tickvals = [i for i in  pop_vec_lin],
                                                        automargin=True
                                                    )

                            )



    return {'data': lines_to_plot, 'layout': layout}



################################################################################################################################################################
def uncertainty_plot(sols, params, cats_to_plot, confidence_range=None):

    population_plot = params.population
    categories = params.categories

    if len(cats_to_plot)==0:
        cats_to_plot=['I']

    font_size = 13

    lines_to_plot = []

    
    percentiles = ['97.5','75','25','2.5']
    labels = ['1','75-97.5 percentile','25-75 percentile','2.5-25 percentile']
    showledge = [False,True,True,True]

    xx = sols[0]['t']

    for name in categories.keys():
        if name == cats_to_plot:
            ii = 0
            
            for yy in confidence_range[:-1]:
                # print(yy.shape)
                if ii == 0:
                    fill = None
                else:
                    fill = 'tonexty'

                if ii==2:
                    opac = '0.5)'
                else:
                    opac = '0.2)'

                ii = ii+1

                yy = np.asarray(yy)
                y_plot = 100*yy[categories[name]['index'],:]



                
                line =  {'x': xx, 'y': y_plot,
                        'hovertemplate': '%{y:.2f}%, %{text}, ' + percentiles[ii-1] + ' percentile<extra></extra>',
                                        # 'Time: %{x:.1f} days<extra></extra>',
                        'text': [population_format(i*population_plot/100) for i in y_plot],
                        'line': {'width': 0, 'color': categories[name]['colour']},
                        'fillcolor': categories[name]['fill_colour'][:-4] + opac,
                        'legendgroup': name + 'fill',
                        'showlegend': showledge[ii-1],
                        'mode': 'lines',
                        # 'opacity': 0.1,
                        'fill': fill,
                        'name': labels[ii-1]
                        }
                lines_to_plot.append(line)

    for name in categories.keys():
            if name == cats_to_plot:
                y_plot = 100*confidence_range[-1][categories[name]['index'],:]
                
                line =  {'x': xx, 'y': y_plot,
                        'hovertemplate': '%{y:.2f}%, %{text}',
                        'text': [population_format(i*population_plot/100) for i in y_plot],
                        'line': {'color': str(categories[name]['colour'])},
                        'legendgroup': name,
                        'name': categories[name]['longname'] + '; median'}
                lines_to_plot.append(line)


    ymax = 0
    for line in lines_to_plot:
        ymax = max(ymax,max(line['y']))


    yax = dict(range= [0,min(1.1*ymax,100)])
    ##

    lines_to_plot.append(
    dict(
        type='scatter',
        x = [0,xx[-1]],
        y = [ 0, population_plot],
        yaxis="y2",
        opacity=0,
        hoverinfo = 'skip',
        showlegend=False
    ))


    yy2 = [0]
    for i in range(8):
        yy2.append(10**(i-5))
        yy2.append(2*10**(i-5))
        yy2.append(5*10**(i-5))

    yy = [i for i in yy2]

    pop_vec_lin = np.linspace(0,yy2[1],11)

    for i in range(len(yy)-1):
        if yax['range'][1]>yy[i] and yax['range'][1] <= yy[i+1]:
            pop_vec_lin = np.linspace(0,yy2[i+1],11)

    vec = [i*(population_plot) for i in pop_vec_lin]

    log_bottom = -8
    log_range = [log_bottom,np.log10(yax['range'][1])]

    pop_vec_log_intermediate = np.linspace(log_range[0],ceil(np.log10(pop_vec_lin[-1])), 1+ ceil(np.log10(pop_vec_lin[-1])-log_range[0]) )

    pop_log_vec = [10**(i) for i in pop_vec_log_intermediate]
    vec2 = [i*(population_plot) for i in pop_log_vec]

    shapes=[]
    annots=[]

    # if control_time[0]!=control_time[1] and not no_control:
    #     shapes.append(dict(
    #             # filled Blue Control Rectangle
    #             type="rect",
    #             x0= control_time[0],
    #             y0=0,
    #             x1= control_time[1],
    #             y1= yax['range'][1],
    #             line=dict(
    #                 color="LightSkyBlue",
    #                 width=0,
    #             ),
    #             fillcolor="LightSkyBlue",
    #             opacity= 0.15
    #         ))

    #     annots.append(dict(
    #             x  = 0.5*(control_time[0] + control_time[1]),
    #             y  = 0.5,
    #             text="<b>Control<br>" + "<b> In <br>" + "<b> Place",
    #             textangle=0,
    #             font=dict(
    #                 size= font_size*(30/24),
    #                 color="blue"
    #             ),
    #             showarrow=False,
    #             opacity=0.4,
    #             xshift= 0,
    #             xref = 'x',
    #             yref = 'paper',
    #     ))


    layout = go.Layout(
                    template="simple_white",
                    shapes=shapes,
                    annotations=annots,
                    font = dict(size= font_size), #'12em'),
                   margin=dict(t=5, b=5, l=10, r=10,pad=15),
                   hovermode='x',
                   xaxis= dict(
                        title='Days',
                        
                        automargin=True,
                        hoverformat='.0f',
                   ),
                   yaxis= dict(mirror= True,
                        title='Percentage of Total Population',
                        range= yax['range'],
                        
                        automargin=True,
                        type = 'linear'
                   ),
                    updatemenus = [dict(
                                            buttons=list([
                                                dict(
                                                    args=[{"yaxis": {'title': 'Percentage of Total Population', 'type': 'linear', 'range': yax['range'], 'automargin': True},
                                                    "yaxis2": {'title': 'Population','type': 'linear', 'overlaying': 'y1', 'range': yax['range'], 'ticktext': [population_format(0.01*vec[i]) for i in range(len(pop_vec_lin))], 'tickvals': [i for i in  pop_vec_lin],'automargin': True,'side':'right'}
                                                    }], # tickformat
                                                    label="Linear",
                                                    method="relayout"
                                                ),
                                                dict(
                                                    args=[{"yaxis": {'title': 'Percentage of Total Population', 'type': 'log', 'range': log_range,'automargin': True},
                                                    "yaxis2": {'title': 'Population','type': 'log', 'overlaying': 'y1', 'range': log_range, 'ticktext': [population_format(0.01*vec2[i]) for i in range(len(pop_log_vec))], 'tickvals': [i for i in  pop_log_vec],'automargin': True,'side':'right'}
                                                    }], # 'tickformat': yax_form_log,
                                                    label="Logarithmic",
                                                    method="relayout"
                                                )
                                        ]),
                                        x= 0.5,
                                        xanchor="right",
                                        pad={"r": 5, "t": 30, "b": 10, "l": 5},
                                        active=0,
                                        y=-0.13,
                                        showactive=True,
                                        direction='up',
                                        yanchor="top"
                                        )],
                                        legend = dict(
                                                        font=dict(size=font_size*(20/24)),
                                                        x = 0.5,
                                                        y = 1.03,
                                                        xanchor= 'center',
                                                        yanchor= 'bottom'
                                                    ),
                                        legend_orientation  = 'h',
                                        legend_title        = '<b> Key </b>',
                                        yaxis2 = dict(
                                                        title = 'Population',
                                                        overlaying='y1',
                                                        
                                                        range = yax['range'],
                                                        side='right',
                                                        ticktext = [population_format(0.01*vec[i]) for i in range(len(pop_vec_lin))],
                                                        tickvals = [i for i in  pop_vec_lin],
                                                        automargin=True
                                                    )

                            )



    return {'data': lines_to_plot, 'layout': layout}




########################################################################################################################
def age_structure_plot(sols, params, cats_to_plot): # ,confidence_range=None

    population_plot = params.population
    categories = params.categories
    population_frame = params.population_frame

    font_size = 13

    lines_to_plot = []

    ii = -1
    for sol in sols:
        ii += 1
        for name in categories.keys():
            if name == cats_to_plot:
                sol['y'] = np.asarray(sol['y'])
                
                xx = sol['t']
                for i in range(population_frame.shape[0]): # # age_categories
                    y_plot = 100*sol['y'][categories[name]['index']+ i*params.number_compartments,:]

                    legend_name = categories[name]['longname'] + ': ' + np.asarray(population_frame.Age)[i] # first one says e.g. infected
                    
                    line =  {'x': xx, 'y': y_plot,

                            'hovertemplate': '%{y:.2f}%, ' + '%{text} <br>',# +
                                            # 'Time: %{x:.1f} days<extra></extra>',
                            'text': [population_format(i*population_plot/100) for i in y_plot],

                            'opacity': 0.5,
                            'name': legend_name}
                    lines_to_plot.append(line)



    ymax = 0
    for line in lines_to_plot:
        ymax = max(ymax,max(line['y']))


    yax = dict(range= [0,min(1.1*ymax,100)])
    ##

    lines_to_plot.append(
    dict(
        type='scatter',
        x = [0,sol['t'][-1]],
        y = [ 0, population_plot],
        yaxis="y2",
        opacity=0,
        hoverinfo = 'skip',
        showlegend=False
    ))

    shapes=[]
    annots=[]
    # if control_time[0]!=control_time[1] and not no_control:
    #     shapes.append(dict(
    #             # filled Blue Control Rectangle
    #             type="rect",
    #             x0= control_time[0],
    #             y0=0,
    #             x1= control_time[1],
    #             y1= yax['range'][1],
    #             line=dict(
    #                 color="LightSkyBlue",
    #                 width=0,
    #             ),
    #             fillcolor="LightSkyBlue",
    #             opacity= 0.15
    #         ))

    #     annots.append(dict(
    #             x  = 0.5*(control_time[0] + control_time[1]),
    #             y  = 0.5,
    #             text="<b>Control<br>" + "<b> In <br>" + "<b> Place",
    #             textangle=0,
    #             font=dict(
    #                 size= font_size*(30/24),
    #                 color="blue"
    #             ),
    #             showarrow=False,
    #             opacity=0.4,
    #             xshift= 0,
    #             xref = 'x',
    #             yref = 'paper',
    #     ))

    yy2 = [0]
    for i in range(8):
        yy2.append(10**(i-5))
        yy2.append(2*10**(i-5))
        yy2.append(5*10**(i-5))

    yy = [i for i in yy2]

    pop_vec_lin = np.linspace(0,yy2[1],11)

    for i in range(len(yy)-1):
        if yax['range'][1]>yy[i] and yax['range'][1] <= yy[i+1]:
            pop_vec_lin = np.linspace(0,yy2[i+1],11)

    vec = [i*(population_plot) for i in pop_vec_lin]

    log_bottom = -8
    log_range = [log_bottom,np.log10(yax['range'][1])]

    pop_vec_log_intermediate = np.linspace(log_range[0],ceil(np.log10(pop_vec_lin[-1])), 1+ ceil(np.log10(pop_vec_lin[-1])-log_range[0]) )

    pop_log_vec = [10**(i) for i in pop_vec_log_intermediate]
    vec2 = [i*(population_plot) for i in pop_log_vec]





    layout = go.Layout(
                    template="simple_white",
                    shapes=shapes,
                    annotations=annots,
                    font = dict(size= font_size), #'12em'),
                   margin=dict(t=5, b=5, l=10, r=10,pad=15),
                   hovermode='x',
                    xaxis= dict(
                        title='Days',
                        
                        automargin=True,
                        hoverformat='.0f',
                   ),
                   yaxis= dict(mirror= True,
                        title='Percentage of Total Population',
                        range= yax['range'],
                        
                        automargin=True,
                        type = 'linear'
                   ),
                    updatemenus = [dict(
                                            buttons=list([
                                                dict(
                                                    args=[{"yaxis": {'title': 'Percentage of Total Population', 'type': 'linear', 'range': yax['range'], 'automargin': True},
                                                    "yaxis2": {'title': 'Population','type': 'linear', 'overlaying': 'y1', 'range': yax['range'], 'ticktext': [population_format(0.01*vec[i]) for i in range(len(pop_vec_lin))], 'tickvals': [i for i in  pop_vec_lin],'automargin': True,'side':'right'}
                                                    }], # tickformat
                                                    label="Linear",
                                                    method="relayout"
                                                ),
                                                dict(
                                                    args=[{"yaxis": {'title': 'Percentage of Total Population', 'type': 'log', 'range': log_range,'automargin': True},
                                                    "yaxis2": {'title': 'Population','type': 'log', 'overlaying': 'y1', 'range': log_range, 'ticktext': [population_format(0.01*vec2[i]) for i in range(len(pop_log_vec))], 'tickvals': [i for i in  pop_log_vec],'automargin': True,'side':'right'}
                                                    }], # 'tickformat': yax_form_log,
                                                    label="Logarithmic",
                                                    method="relayout"
                                                )
                                        ]),
                                        x= 0.5,
                                        xanchor="right",
                                        pad={"r": 5, "t": 30, "b": 10, "l": 5},
                                        active=0,
                                        y=-0.13,
                                        showactive=True,
                                        direction='up',
                                        yanchor="top"
                                        )],
                                        legend = dict(
                                                        font=dict(size=font_size*(20/24)),
                                                        x = 0.5,
                                                        y = 1.03,
                                                        xanchor= 'center',
                                                        yanchor= 'bottom'
                                                    ),
                                        legend_orientation  = 'h',
                                        legend_title        = '<b> Key </b>',
                                        yaxis2 = dict(
                                                        title = 'Population',
                                                        overlaying='y1',
                                                        
                                                        range = yax['range'],
                                                        side='right',
                                                        ticktext = [population_format(0.01*vec[i]) for i in range(len(pop_vec_lin))],
                                                        tickvals = [i for i in  pop_vec_lin],
                                                        automargin=True
                                                    )

                            )



    return {'data': lines_to_plot, 'layout': layout}


########################################################################################################################
def stacked_bar_plot(sols, params, cats_to_plot):

    population_plot = params.population
    categories = params.categories
    population_frame = params.population_frame

    font_size = 13
    lines_to_plot = []

    ii = -1
    for sol in sols:
        ii += 1
        for name in categories.keys():
            if name == cats_to_plot:
                sol['y'] = np.asarray(sol['y'])
                
                xx = sol['t']
                y_sum = np.zeros(len(xx))
                
                xx = [xx[i] for i in range(1,len(xx),2)]
                
                for i in range(population_frame.shape[0]): # age_cats
                    y_plot = 100*sol['y'][categories[name]['index']+ i*params.number_compartments,:]
                    y_sum  = y_sum + y_plot
                    legend_name = categories[name]['longname'] + ': ' + np.asarray(population_frame.Age)[i] # first one says e.g. infected

                    y_plot = [y_plot[i] for i in range(1,len(y_plot),2)]
                    
                    line =  {'x': xx, 'y': y_plot,

                            'hovertemplate': '%{y:.2f}%, ' + '%{text} <br>',# +
                                            # 'Time: %{x:.1f} days<extra></extra>',
                            'text': [population_format(i*population_plot/100) for i in y_plot],
                            # 'marker_line_width': 0,
                            # 'marker_line_color': 'black',
                            'type': 'bar',
                            'name': legend_name}
                    lines_to_plot.append(line)



    ymax = max(y_sum)
    # ymax = 0
    # for line in lines_to_plot:
    #     ymax = max(ymax,max(line['y']))


    yax = dict(range= [0,min(1.1*ymax,100)])
    ##

    lines_to_plot.append(
    dict(
        type='scatter',
        x = [0,sol['t'][-1]],
        y = [ 0, population_plot],
        yaxis="y2",
        opacity=0,
        hoverinfo = 'skip',
        showlegend=False
    ))


    yy2 = [0]
    for i in range(8):
        yy2.append(10**(i-5))
        yy2.append(2*10**(i-5))
        yy2.append(5*10**(i-5))

    yy = [i for i in yy2]

    pop_vec_lin = np.linspace(0,yy2[1],11)

    for i in range(len(yy)-1):
        if yax['range'][1]>yy[i] and yax['range'][1] <= yy[i+1]:
            pop_vec_lin = np.linspace(0,yy2[i+1],11)

    vec = [i*(population_plot) for i in pop_vec_lin]

    log_bottom = -8
    log_range = [log_bottom,np.log10(yax['range'][1])]

    pop_vec_log_intermediate = np.linspace(log_range[0],ceil(np.log10(pop_vec_lin[-1])), 1+ ceil(np.log10(pop_vec_lin[-1])-log_range[0]) )

    pop_log_vec = [10**(i) for i in pop_vec_log_intermediate]
    vec2 = [i*(population_plot) for i in pop_log_vec]





    layout = go.Layout(
                    template="simple_white",
                    font = dict(size= font_size), #'12em'),
                   margin=dict(t=5, b=5, l=10, r=10,pad=15),
                   hovermode='x',
                   xaxis= dict(
                        title='Days',
                        
                        automargin=True,
                        hoverformat='.0f',
                   ),
                   yaxis= dict(mirror= True,
                        title='Percentage of Total Population',
                        range= yax['range'],
                        
                        automargin=True,
                        type = 'linear'
                   ),
                    barmode = 'stack',

                    updatemenus = [dict(
                                            buttons=list([
                                                dict(
                                                    args=[{"yaxis": {'title': 'Percentage of Total Population', 'type': 'linear', 'range': yax['range'], 'automargin': True},
                                                    "yaxis2": {'title': 'Population','type': 'linear', 'overlaying': 'y1', 'range': yax['range'], 'ticktext': [population_format(0.01*vec[i]) for i in range(len(pop_vec_lin))], 'tickvals': [i for i in  pop_vec_lin],'automargin': True,'side':'right'}
                                                    }], # tickformat
                                                    label="Linear",
                                                    method="relayout"
                                                ),
                                                dict(
                                                    args=[{"yaxis": {'title': 'Percentage of Total Population', 'type': 'log', 'range': log_range,'automargin': True},
                                                    "yaxis2": {'title': 'Population','type': 'log', 'overlaying': 'y1', 'range': log_range, 'ticktext': [population_format(0.01*vec2[i]) for i in range(len(pop_log_vec))], 'tickvals': [i for i in  pop_log_vec],'automargin': True,'side':'right'}
                                                    }], # 'tickformat': yax_form_log,
                                                    label="Logarithmic",
                                                    method="relayout"
                                                )
                                        ]),
                                        x= 0.5,
                                        xanchor="right",
                                        pad={"r": 5, "t": 30, "b": 10, "l": 5},
                                        active=0,
                                        y=-0.13,
                                        showactive=True,
                                        direction='up',
                                        yanchor="top"
                                        )],
                                        legend = dict(
                                                        font=dict(size=font_size*(20/24)),
                                                        x = 0.5,
                                                        y = 1.03,
                                                        xanchor= 'center',
                                                        yanchor= 'bottom'
                                                    ),
                                        legend_orientation  = 'h',
                                        legend_title        = '<b> Key </b>',
                                        yaxis2 = dict(
                                                        title = 'Population',
                                                        overlaying='y1',
                                                        
                                                        range = yax['range'],
                                                        side='right',
                                                        ticktext = [population_format(0.01*vec[i]) for i in range(len(pop_vec_lin))],
                                                        tickvals = [i for i in  pop_vec_lin],
                                                        automargin=True
                                                    )

                            )

    return {'data': lines_to_plot, 'layout': layout}







