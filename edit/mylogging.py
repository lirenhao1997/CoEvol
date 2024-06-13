import logging
import time
import os


def get_logger():
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_date = rq[:8]
    log_path = os.getcwd() + '/logs/{}/'.format(log_date)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_name = log_path + rq + '.log'
    logfile = log_name

    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(logfile, encoding='UTF-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.DEBUG)
    console_handler.setLevel(logging.ERROR)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger, logfile


my_log, save_log_path = get_logger()
