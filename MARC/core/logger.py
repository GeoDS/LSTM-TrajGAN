import sys
from datetime import datetime

class Logger(object):

    LOG_LINE = None
    INFO        = '[    INFO    ]'
    WARNING     = '[  WARNING   ]'
    ERROR       = '[   ERROR    ]'
    CONFIG      = '[   CONFIG   ]'
    RUNNING     = '[  RUNNING   ]'
    QUESTION    = '[  QUESTION  ]'

    def log(self, type, message):
        if Logger.LOG_LINE:
            sys.stdout.write("\n")
            sys.stdout.flush()
            Logger.LOG_LINE = None

        sys.stdout.write(str(type) + " " + self.cur_date_time() + " :: " + message + "\n")
        sys.stdout.flush()

    def log_dyn(self, type, message):
        line = str(type) + " " + self.cur_date_time() + " :: " + message
        sys.stdout.write("\r\x1b[K" + line.__str__())
        sys.stdout.flush()
        Logger.LOG_LINE = line

    def get_answer(self, message):
        return input(Logger.QUESTION + " " + self.cur_date_time() + " :: " + message)

    def cur_date_time(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")