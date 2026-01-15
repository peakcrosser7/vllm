#!/usr/bin/env python

from enum import Enum, auto
import time
from typing import List
import requests
import multiprocessing as mp
import sys

class TestType(Enum):
    Dummy = auto()
    Real = auto()

SEED = 1234
SEED = None
PORT = 8235
NUM_REQUESTS = 1
MAX_NEW_TOKENS = 4096
IGNORE_EOS = False


ONE_PROMPT = ['请详细介绍一下北京和上海这两座城市, 各不少于10000字']


def hello(pid: int, max_new_tokens: int, ignore_eos: bool):
    headers = {
        "Content-Type": "application/json",
    }
    url = f"http://localhost:{PORT}/v1/chat/completions"

    messages = [{"role": "user", "content": ONE_PROMPT[0]}] 
    data = {
        "messages": messages,
        "max_tokens": max_new_tokens,
        "ignore_eos": ignore_eos,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "repetition_penalty": 1,
        "presence_penalty": 1.5,
        **({'seed': SEED} if SEED is not None else {}),
        "chat_template_kwargs": {"enable_thinking": False}
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        print(f"[PID {pid}] Response:\n {result['choices'][0]['message']['content']}\n {'-' * 40}\n", end='')
    else:
        print(f"Request failed with status code {response.status_code}")
        print("Response content:")
        print(response.content)

def main():
    procs: List[mp.Process] = []

    start = time.time()
    for pid in range(NUM_REQUESTS):
        proc = mp.Process(
            target=hello, args=(pid, MAX_NEW_TOKENS, IGNORE_EOS), daemon=True
        )
        proc.start()
        procs.append(proc)

    for _proc in procs:
        _proc.join()
        if _proc.exitcode != 0:
            sys.exit(_proc.exitcode)

    elapsed = time.time() - start
    output_tps = MAX_NEW_TOKENS * NUM_REQUESTS / elapsed
    print("\n")
    print(f"Generate {output_tps} tokens/s, elapsed: {elapsed} s, TPS {output_tps / NUM_REQUESTS}, TPOT {1000 / (output_tps / NUM_REQUESTS)}ms")


if __name__ == "__main__":
    main()