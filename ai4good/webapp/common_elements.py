import dash_bootstrap_components as dbc


def nav_bar(internal=None):
    if (not internal) or (internal.lower() == 'log out'):
        internal_children = 'Log out'
        internal_href = '/logout/'
    elif internal.lower() == 'register':
        internal_children = 'Register'
        internal_href = '/auth/register/'
    elif internal.lower() == 'login':
        internal_children = 'Login'
        internal_href = '/auth/login/'
    elif internal.lower() in ['landing', 'start']:
        internal_children = 'Start'
        internal_href = '/sim/'
    else:
        internal_children = ''
        internal_href = '#'
    
    return dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink('Home', href='https://www.aiforgood.co.uk/', target='_blank')),
            dbc.NavItem(dbc.NavLink('What is it?', href='https://www.aiforgoodsimulator.com/', target='_blank')),
            dbc.NavItem(dbc.NavLink('About us', href='https://www.aiforgood.co.uk/about-us/', target='_blank')),
            dbc.NavItem(dbc.NavLink(internal_children, href=internal_href, target='_blank', id='navlink-internal')),
            dbc.NavItem(dbc.NavLink('', href='#', target='_blank', id='navlink-user')),
            dbc.Tooltip('', id='navlink-tooltip-user', target='navlink-user'),
        ],
        brand='AI for Good Simulator',
        brand_href='#',
        color='primary',
        dark=True,
    )
