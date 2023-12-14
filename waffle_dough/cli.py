from waffle_utils.logger import initialize_logger

initialize_logger(
    file_path=Path(Config.LOG_DIR) / Config.LOG_FILE_NAME,
    log_format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s",
    console_level=Config.CONSOLE_LOG_LEVEL,
    file_level=Config.FILE_LOG_LEVEL,
    root_level=logging.INFO,
    backup_count=Config.BACKUP_COUNT,
    encoding=Config.LOG_ENCODING,
    when=Config.LOG_WHEN,
    interval=Config.LOG_INTERVAL,
)
