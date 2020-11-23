import dash_bootstrap_components as dbc
import dash_html_components as html
import ai4good.webapp.common_elements as common_elements


text_introduction = 'The Simulator is a web tool for NGOs and local authorities to model COVID-19 outbreak inside refugee camps and prepare timely and proportionate response measures needed to flatten the curve and reduce the number of fatalities. This tool helps to predict the possble outbreak scenarios and their potential outcomes and help NGOs design an optimal intervention strategy. '

layout = html.Div(
    [
        common_elements.nav_bar(),
        
        html.Div([
            dbc.Container([
                dbc.Row(
                    dbc.Col(
                        dbc.Card([
                            html.H4('AI for Good Simulator', className='card-title', style={'margin-bottom':'25px'}),
                            html.P(html.Center(text_introduction), className='card-text', style={'margin-bottom':'35px'}),
                            dbc.CardFooter(
                                html.Div([
                                    html.Center(dbc.Button('Get Started', id='landing-button', color='secondary', href='/sim/input_page_1')), 
                                ]), 
                            ), 
                            html.Div(id='landing-alert')
                            ], body=True
                        ), width=6
                    ), justify='center'
                )
            ])
        ], style={'padding':'100px'})
    ]
)
