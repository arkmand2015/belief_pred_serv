# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 14:38:08 2020

@author: arkma
"""


import math
import json
from pathlib import Path
import yaml

def exception_handler(request, exception):
    print("Request failed")


def parallelize(user_id: str, timeline: list):
    with open("urls.yaml", 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    urls = data_loaded['url_list']
    headers = {"Content-Type": "application/json"}
    chunk_size = math.ceil(len(timeline) / len(urls))
    offset = 0
    requests_list = []
    for u in urls:
        _timeline = timeline[offset:offset + chunk_size]
        _timeline = [str(item) for item in _timeline]
        payload = {
            'users_batch': [
                {
                    "user_id": user_id,
                    "timeline": _timeline
                }
            ],
            'zero_shot_enabled': True
        }
        payload = json.dumps(payload)
        request = grequests.post(u, headers=headers, data=payload)
        requests_list.append(request)
        offset += len(_timeline)
    responses = grequests.map(requests_list, exception_handler=exception_handler)
    return responses
