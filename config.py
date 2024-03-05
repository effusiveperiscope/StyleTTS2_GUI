from omegaconf import OmegaConf
import os

CONFIG_PATH = 'config.yaml'
def load_config():
    if os.path.exists(CONFIG_PATH):
        config = OmegaConf.load(CONFIG_PATH)
    else:
        config = OmegaConf.create({
            'device': 'cuda',
            'n_infer': 3,
            'models_dir': 'Models',
            'output_dir': 'results',
            'dark_mode': False,
        })
    return config

def save_config(config):
    with open(CONFIG_PATH,'w') as fp:
        OmegaConf.save(config=config, f=fp)