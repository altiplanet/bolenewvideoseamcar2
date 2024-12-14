"""Logging configuration for both file and notebook output."""
import logging
from IPython.display import display, HTML
from datetime import datetime
import sys

class NotebookHandler(logging.Handler):
    """Custom handler for Jupyter notebook output."""
    def emit(self, record):
        try:
            msg = self.format(record)
            if record.levelno >= logging.ERROR:
                display(HTML(f'<div style="color: red">{msg}</div>'))
            elif record.levelno >= logging.WARNING:
                display(HTML(f'<div style="color: orange">{msg}</div>'))
            else:
                display(HTML(f'<div style="color: black">{msg}</div>'))
        except Exception:
            self.handleError(record)

def setup_logging(notebook_mode=True):
    """Configure logging with both file and notebook output."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"seam_carving_{timestamp}.log"
    
    # Clear any existing handlers
    logging.getLogger().handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Console/Notebook handler
    if notebook_mode:
        console_handler = NotebookHandler()
    else:
        console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logging.info(f"Logging initialized. Log file: {log_file}")