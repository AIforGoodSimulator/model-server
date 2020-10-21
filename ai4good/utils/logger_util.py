import logging


def get_logger(name: str, level: str = 'INFO'):

    logger = logging.getLogger(name)

    if level == 'DEBUG':
        logger.setLevel(logging.DEBUG)
    elif level == 'WARNING':
        logger.setLevel(logging.WARNING)
    elif level == 'ERROR':
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    logger.addHandler(ch)

    return logger


if __name__ == "__main__":
    log = get_logger('hello', 'ERROR')
    log.debug('debug')
    log.info('info')
    log.warning('warn')
    log.error('error')
