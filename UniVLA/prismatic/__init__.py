# The full model/RLDS stack (dlimp, tensorflow_datasets, ...) is optional: the
# rc365 RoboCasa scripts only use `prismatic.extern.hf.*`,
# `prismatic.models.policy.transformer_utils` and the prompt builders, none of
# which need it. Keep the import soft so a lean env still works.
try:
    from .models import available_model_names, available_models, get_model_description, load
except ImportError:
    pass



__version__ = "0.0.1"
__project__ = "OmniEmbodiment"
__author__ = "Qingwen Bu"
__license__ = "Apache License 2.0"
__email__ = "qwbu01@sjtu.edu.cn"