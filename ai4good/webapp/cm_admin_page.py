import dash_html_components as html
from ai4good.webapp.apps import dash_app, facade, _redis, cache, local_cache
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
            ]),
            html.Div([
                html.Button('Clear redis', id='clear_redis_button'),
                html.Div(id='notification_div1')
            ]),
            html.Div([
                html.Button('Clear cache', id='clear_cache_button'),
                html.Div(id='notification_div2'),
            ]),
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


@dash_app.callback(
    Output("notification_div1", "children"),
    [Input("clear_redis_button", "n_clicks")],
)
def handle_clear_redis(n_clicks):
    if n_clicks and n_clicks > 0:
        _redis.flushdb()
        return 'async clear called'
    else:
        return dash.no_update


@dash_app.callback(
    Output("notification_div2", "children"),
    [Input("clear_cache_button", "n_clicks")],
)
def handle_clear_cache(n_clicks):
    if n_clicks and n_clicks > 0:
        cache.clear()
        local_cache.clear()
        return 'cache cleared'
    else:
        return dash.no_update
