"""Entrada do worker SQS. Execute: python batch/run_consumer.py"""
from __future__ import annotations

import logging

from dotenv import load_dotenv
from skin_app.logging_config import setup_logging

load_dotenv()

setup_logging()
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    from batch.consumer import run
    logger.info("Iniciando consumer SQS...")
    run()
