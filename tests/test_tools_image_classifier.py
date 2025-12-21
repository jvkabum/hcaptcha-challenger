import os
from pathlib import Path

import dotenv
from loguru import logger

from hcaptcha_challenger import ImageClassifier

dotenv.load_dotenv()

ic = ImageClassifier(gemini_api_key=os.getenv("GEMINI_API_KEY"), model="gemini-3-flash-preview")
CHALLENGE_VIEW_DIR = Path(__file__).parent.joinpath("challenge_view/image_label_binary")


async def test_gemini_image_classifier():
    challenge_screenshot = CHALLENGE_VIEW_DIR.joinpath("1.png")
    results = await ic(challenge_screenshot=challenge_screenshot)
    logger.debug(f'ToolInvokeMessage: {results.log_message}')
