import logging
import os
import sys
import time
from datetime import datetime

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)
filename = f"logs/ppo.{datetime.utcnow():%Y-%m-%dT%H:%MZ}.log"

sh = logging.StreamHandler(sys.stdout)
fh = logging.FileHandler(filename, mode="w", delay=True)

FORMAT = "%(asctime)sZ %(levelname)-7s %(name)-11s %(message)s"
DATEFORMAT = "%Y-%m-%dT%H:%M:%S"

formatter = logging.Formatter(fmt=FORMAT, datefmt=DATEFORMAT)
formatter.converter = time.gmtime
sh.setFormatter(formatter)
fh.setFormatter(formatter)

log_main = logging.getLogger("ppo.main")
log_collect = logging.getLogger("ppo.collect")
log_train = logging.getLogger("ppo.train")
log_eval = logging.getLogger("ppo.eval")
log_test = logging.getLogger("ppo.test")

for logger in [log_main, log_collect, log_train, log_eval, log_test]:
    logger.setLevel(logging.INFO)
    logger.addHandler(sh)
    logger.addHandler(fh)


def no_logfile():
    for logger in [log_main, log_collect, log_train, log_eval, log_test]:
        logger.removeHandler(fh)


def use_logfile(filename):
    new_fh = logging.FileHandler(filename, mode="w", delay=True)
    new_fh.setFormatter(formatter)
    for logger in [log_main, log_collect, log_train, log_eval, log_test]:
        logger.removeHandler(fh)
        logger.addHandler(new_fh)


# collect_log.info("what")
# train_log.warning("how")
# eval_log.debug("why")
# eval_log.error("why")
# test_log.critical("!!!")
