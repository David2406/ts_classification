import logging

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    level=logging.INFO,
                    datefmt='%Y/%m/%d %H:%M:%S')
volta_logger = logging.getLogger('VOLTA logger')
volta_logger.info('VOLTA logger initialized')
