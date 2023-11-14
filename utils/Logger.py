# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2021/10/14
    Desc :
    Note : 
'''
import os
import logging
import sys
from LogDict import log_dict


def logcode(code,info=None):
    # log_dict = {}
    str_info = 'Code: [{}]'.format(str(code))
    value = log_dict.get(code, None)
    if info is None:
        return str_info if value is None else value
    else:
        return str_info if value is None else value.format(info)


# 日志输出
class Logger():
    # 日志级别关系映射
    level_relations = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL
    }

    def __init__(self, level="info", name=None):
        if getattr(sys, 'frozen', False): 
            BASE_PATH  = "C:\\"
        else: 
            BASE_PATH = os.path.join(os.path.pardir, sys.path[0])  # 生产环境
        
        logFilename = os.path.join(BASE_PATH, 'invoice_running.log')
        fmt = "%(asctime)s - %(name)s[line:%(lineno)d] - %""(levelname)s: %(message)s"
        logging.basicConfig(
            level=self.level_relations[level],
            format=fmt,
            handlers=[logging.FileHandler(logFilename, mode="a", encoding="utf-8")])
        self.logger = logging.getLogger(name)

        console = logging.StreamHandler()
        console.setLevel(self.level_relations[level])
        formatter = logging.Formatter(fmt)
        console.setFormatter(formatter)
        self.logger.addHandler(console)
        self.logger.debug('开始初始化Logger,Log Name:[{}],Log Path :[{}]'.format(name, logFilename))

    def getlog(self):
        return self.logger


if __name__ == '__main__':
    # Logger使用方法
    # from Logger import Logger
    logging = Logger(level="info", name=__name__).getlog()
    logging.info('test')
