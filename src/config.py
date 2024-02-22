import os
import datetime
import json
from __init__ import volta_logger
from src.utils import replace_words


def init_config_options(config_file_path = './configs/thermal_optimized_energy.json',
                        output_folder_needed = False,
                        logger = volta_logger):

    logger.info("Loading all configuration parameters")
    config_file = open(config_file_path)
    conf = json.load(config_file)
    config_file.close()

    save_dir = conf['save_dir']
    if "work_dir" in conf:
        work_dir = conf['work_dir']
    else:
        work_dir = os.getcwd()
        conf['work_dir'] = work_dir

    logger.info('Current working directory is : %s', work_dir)
    logger.info('Saving results in directory : %s', save_dir)

    if output_folder_needed:

        timestamp = datetime.datetime.today().strftime('%Y-%m-%d_%H%M%S')
        output_folder_path = conf['output_folder_path']
        output_folder_path = replace_words(output_folder_path,
                                           {'%timestamp%': timestamp})
        logger.info('Saving simulation folders in directory : %s', output_folder_path)
        return conf, work_dir, save_dir, output_folder_path

    return conf, work_dir, save_dir
