#!/usr/bin/env python3

from io import BytesIO
from os.path import dirname
from urllib3 import PoolManager, HTTPResponse
from zipfile import ZipFile


def main():
    http = PoolManager()

    print('downloading OCR model')
    r: HTTPResponse = http.request('GET', 'http://www.fit.vutbr.cz/~ihradis/pero-models/ocr_LN_2019-12-18.zip')

    assert 200 <= r.status < 300, f"Request status {r.status}"

    print('unzipping OCR model')
    zfile = ZipFile(BytesIO(r.data))
    zfile.extractall(path=dirname(__file__))

    print('downloading enhancement model')
    r: HTTPResponse = http.request('GET', 'http://www.fit.vutbr.cz/~ihradis/pero-models/enhance_LN_2019-12-18.zip')

    assert 200 <= r.status < 300, f"Request status {r.status}"

    print('unzipping enhancement model')
    zfile = ZipFile(BytesIO(r.data))
    zfile.extractall(path=dirname(__file__))


if __name__ == "__main__":
    main()
