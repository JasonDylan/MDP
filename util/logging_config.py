# logging_config.py
import inspect
import logging
import logging.handlers
import os


def setup_logging(caller_dir_name=None):
    # 创建日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 创建格式化器,包含文件名和行号信息
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(processName)s - %(threadName)s - %(message)s - %(filename)s:%(lineno)d - [PID: %(process)d]"
    )

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    # 获取调用函数的名称
    if not caller_dir_name:
        caller_frame = inspect.stack()[1]
        caller_function = caller_frame.function

    # 创建按时间切割的文件处理器,使用调用函数名称作为日志文件名的一部分

    dir_name = os.path.splitext(os.path.basename(caller_dir_name))[0]
    dir_name = dir_name if dir_name else caller_function
    log_folder = f"./log/{dir_name}"  # 日志文件夹
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    log_filename = f"a.log"
    log_file_path = os.path.join(log_folder, log_filename)

    # 使用 TimedRotatingFileHandler，每小时切割一次日志文件，保留最近 7 天的日志文件
    file_handler = logging.handlers.TimedRotatingFileHandler(
        log_file_path,
        when="H",
        interval=1,
        backupCount=168,  # 24 hours * 7 days = 168 hours
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # 将处理器添加到日志记录器
    # logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logging.info(f"mkdir {log_folder}")

    return logger