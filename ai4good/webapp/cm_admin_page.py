import dash_html_components as html
from ai4good.webapp.apps import dash_app, facade
from ai4good.models.cm.cm_model import CompartmentalModel
from dash.dependencies import Input, Output
import dash


def layout():
    return html.Div(
        [
            html.H3('Admin page'),
            html.Div([
                html.B("CM Model Cache"),
                html.Pre(f'{facade.rs.list(CompartmentalModel.ID)}', id='cache_contents'),
                html.Button('Clear', id='clear_button'),
            ])

        ], style={'margin': 10}
    )


@dash_app.callback(
    Output("cache_contents", "children"),
    [Input("clear_button", "n_clicks")],
)
def update_output(n_clicks):
    if n_clicks and n_clicks > 0:
        facade.rs.remove_all(CompartmentalModel.ID)
        return f'{facade.rs.list(CompartmentalModel.ID)}'
    else:
        return dash.no_update
