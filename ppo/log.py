import logging
import os
import sys
import time
from datetime import datetime

os.makedirs("logs", exist_ok=True)
filename = f"logs/ppo.{datetime.utcnow():%Y-%m-%dT%HZ}.log"

fh = logging.FileHandler(filename, mode="w")
sh = logging.StreamHandler(sys.stdout)

FORMAT = "%(asctime)sZ  %(levelname)-7s  %(name)-11s  %(message)s"

formatter = logging.Formatter(fmt=FORMAT)
formatter.converter = time.gmtime
fh.setFormatter(formatter)
sh.setFormatter(formatter)

log_main = logging.getLogger("ppo.main")
log_collect = logging.getLogger("ppo.collect")
log_train = logging.getLogger("ppo.train")
log_eval = logging.getLogger("ppo.eval")
log_test = logging.getLogger("ppo.test")

for logger in [log_main, log_collect, log_train, log_eval, log_test]:
    logger.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)

# collect_log.info("what")
# train_log.warning("how")
# eval_log.debug("why")
# eval_log.error("why")
# test_log.critical("!!!")
