# probe_openai.py

"""
Tests for OpenAI API connectivity.
"""

import os

from dotenv import load_dotenv
from httpx import HTTPStatusError
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), max_retries=0, timeout=20)

try:
    r = client.responses.create(
        model="gpt-4o-mini",
        input="Say 'pong' if you can see this.",
        max_output_tokens=16,  # >= 16
    )
    print("OK:", r.output_text)
except HTTPStatusError as e:
    resp = e.response
    print("STATUS:", resp.status_code)
    print("HEADERS:", dict(resp.headers))
    print("BODY:", resp.text)
    raise
