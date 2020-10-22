from ai4good.runner.facade import Facade
from ai4good.models.abm.np_impl.moria import *
from ai4good.models.model_registry import create_params


def get_params():
    _model = 'agent-based-model'
    # Possible values: "baseline", "small", ...
    _profile = 'small'
    camp = 'Moria'
    overrides = '{"numberOfIterations": 1, "nProcesses": 1}'
    facade = Facade.simple()
    params = create_params(facade.ps, _model, _profile, camp, overrides)
    return params


def test_unique_eth_per_hh():
    # Test that each household in the camp has a unique ethnicity

    # Initially all agents are inside their households
    moria = Moria(get_params())


    # Find agents who share households
    pos = moria.agents[:, [A_X, A_Y]]
    sharing_hh = OptimizedOps.distance_matrix(pos) < SMALL_ERROR
    sharing_hh_ind = np.argwhere(sharing_hh)

    # We now have pairs of agents i and j where i and j share households.
    # Get ethnicity for all such i and j and check if for any pair (i,j) ethnicity don't match.
    i = sharing_hh_ind[:, 0]
    j = sharing_hh_ind[:, 1]
    eth_i = moria.agents[i, A_ETHNICITY]
    eth_j = moria.agents[j, A_ETHNICITY]

    eth_mismatch_count = np.count_nonzero(eth_i != eth_j)/2

    print(eth_mismatch_count)
    assert eth_mismatch_count == 0, "Some households contain people from multiple ethnicities"
