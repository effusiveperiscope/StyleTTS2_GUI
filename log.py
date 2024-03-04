import logging
class DisplayHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.text_contents = ""
        self.custom_handler = None

    def set_handler(self, custom_handler):
        self.custom_handler = custom_handler
    
    def emit(self, record):
        msg = self.format(record)
        print(msg)
        self.text_contents += msg + '\n'
        if self.custom_handler is not None:
            self.custom_handler(self.text_contents)

logger = logging.getLogger('styletts2')
logger.setLevel(logging.INFO)
display_handler = DisplayHandler()
display_handler.setFormatter(
    logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s'))
logger.addHandler(display_handler)