# This file specifies an interface that the generative model must implement
class ModelConfig:
    model_name: str =  None
    state_shape = None
    index_set: list = []
    deets = {
        
    }
    def to_dict(self):
        return {
            "model_name": self.model_name,
            "state_shape": self.state_shape,
            "index_set": self.index_set,
            "deets": self.deets,
        }


class Core: # Abstract class defining interaction protocol for a diffusion core
    def __init__(self, config):
        self.state_shape = None
        self.model_config = config
        self.t = None
        # self.X # current state for the model, has shape self.state_shape
        # self.tentative_states # list of proposed states from last call to propose function, has shape [M, self.state_shape] for some M
        # self.model_details = ModelConfig()

    def load_from_config(self, config_path):
        pass

    def propose(self, S = 1): # generates proposal or list thereof, returns np.array([S, self.state_shape])
        pass

    def propose_external(self, init_state, index): # propose state from init_state, index
        pass

    def commit(self, state): # accepts state of shape self.state_shape and saves it as self.state
        pass
        # self.state

    def reset(self):
        pass

    def collect_run(self):
        pass

    def save(self, states, data_dir: str, run_id: str) -> dict:
        # This must be defined in subclasses because the saving process depends on the datatype of the state
        pass
