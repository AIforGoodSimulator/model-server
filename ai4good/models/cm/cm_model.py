from abc import ABCMeta, abstractmethod

from typeguard import typechecked

from ai4good.models.cm.seirsde import SEIRSDESolver
from ai4good.models.model import Model, ModelResult
from ai4good.params.param_store import ParamStore
from ai4good.models.cm.initialise_parameters import Parameters
from ai4good.models.cm.simulator import Simulator, generate_csv
from ai4good.utils.logger_util import get_logger
from ai4good.webapp.cm_model_report_utils import normalize_report, prevalence_age_table, prevalence_all_table, \
    cumulative_all_table, cumulative_age_table

logger = get_logger(__name__)


class AbstractCompartmentalModel(Model):
    def __init__(self, ps: ParamStore):
        Model.__init__(self, ps)

    @abstractmethod
    def simulate(self, p):
        pass

    def run(self, p: Parameters) -> ModelResult:
        # config_dict, percentiles, sols_raw, standard_sol = self.simulate(p)
        config_dict, sols_raw = self.simulate(p)  # we are currently using sols_raw in results generating
        # Precompute some reports
        logger.info("Generating main report")
        if config_dict is None:
            report_raw = generate_csv(sols_raw, p, input_type='stochastic')
        else:
            report_raw = generate_csv(sols_raw, p, input_type='raw')

        # report = normalize_report(report_raw, p)
        #
        # logger.info("Computing prevalence_age_table")
        # prevalence_age = prevalence_age_table(report)
        # logger.info("Computing prevalence_all_table")
        # prevalence_all = prevalence_all_table(report)
        # logger.info("Computing cumulative_all_table")
        # cumulative_all = cumulative_all_table(report, p.population, p.camp_params)
        # logger.info("Computing cumulative_age_table")
        # cumulative_age = cumulative_age_table(report, p.camp_params)

        logger.info("Model result ready")
        return ModelResult(self.result_id(p), {
            # 'percentiles': percentiles,
            'config_dict': config_dict,
            'params': p,
            'report': report_raw,
            # 'prevalence_age': prevalence_age,
            # 'prevalence_all': prevalence_all,
            # 'cumulative_all': cumulative_all,
            # 'cumulative_age': cumulative_age
        })


@typechecked
class CompartmentalModel(AbstractCompartmentalModel):

    ID = 'compartmental-model'
    def __init__(self, ps: ParamStore):
        AbstractCompartmentalModel.__init__(self, ps)

    def id(self) -> str:
        return self.ID

    def result_id(self, p: Parameters) -> str:
        return p.sha1_hash()

    def simulate(self, p):
        sim = Simulator(p)
        sols_raw, config_dict = sim.simulate_over_parameter_range_parallel(
            p.control_dict['numberOfIterations'], p.control_dict['t_sim'], p.control_dict['nProcesses'], p.generated_disease_vectors)
        return config_dict, sols_raw,

@typechecked
class CompartmentalModelStochastic(AbstractCompartmentalModel):

    ID = 'compartmental-model-stochastic'
    def __init__(self, ps: ParamStore):
        AbstractCompartmentalModel.__init__(self, ps)

    def id(self) -> str:
        return self.ID

    def result_id(self, p: Parameters) -> str:
        return p.sha1_hash()

    def simulate(self, p):
        sim = SEIRSDESolver(p)
        sols_raw, config_dict = sim.simulate_over_parameter_range_parallel(
            p.control_dict['numberOfIterations'], p.control_dict['t_sim'],  p.control_dict['nProcesses'], p.control_dict['random_seed'])
        return config_dict, sols_raw

