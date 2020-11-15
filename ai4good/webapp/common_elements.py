import dash_bootstrap_components as dbc


def nav_bar():
    return dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Home", href="https://www.aiforgood.co.uk/", target="_blank")),
            dbc.NavItem(dbc.NavLink("What is it?", href="https://www.aiforgoodsimulator.com/", target="_blank")),
            dbc.NavItem(dbc.NavLink("About us", href="https://www.aiforgood.co.uk/about-us", target="_blank")),
        ],
        brand="AI for Good Simulator",
        brand_href="#",
        color="primary",
        dark=True,
    )