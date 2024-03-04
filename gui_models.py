import os
import glob
import json
from log import logger

def find_models(config):
    models_dir = config['models_dir']
    models_dir = os.path.abspath(models_dir)

    model_specs = {}
    for model_dir in os.listdir(models_dir):
        if not os.path.isdir(os.path.join(models_dir,model_dir)):
            continue
        model_spec = {}
        config_path = glob.glob(
            os.path.join(models_dir,model_dir,'*.yml'))
        if not len(config_path):
            logger.warn(f'No config found for model dir {model_dir}')
            continue
        model_spec['config'] = config_path[0]
        ckpt_path = glob.glob(os.path.join(models_dir,model_dir,'*.pth'))
        if not len(ckpt_path):
            logger.warn(f'No checkpoint found for model dir {model_dir}')
            continue
        model_spec['ckpt'] = ckpt_path[0]
        style_index_path = glob.glob(
            os.path.join(models_dir,model_dir,'*.json'))
        if not len(style_index_path):
            style_index_path = None
            model_spec['style_index'] = None
        else:
            model_spec['style_index'] = os.path.join(
                models_dir,model_dir,style_index_path[0])
            with open(model_spec['style_index'], encoding='utf-8') as f:
                j = json.load(f)
                model_spec['style_index_data'] = j
        model_specs[model_dir] = model_spec
    return model_specs