# !/usr/bin/python
# -*- coding: UTF-8 -*-

import random
from urllib.parse import quote

import urllib3
from bs4 import BeautifulSoup

http = urllib3.PoolManager()

header = {
    "Host": "xh.5156edu.com",
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:66.0) Gecko/20100101 Firefox/66.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "Referer": "https://www.baidu.com",
    "Connection": "keep-alive",
    "Content-Type": "application/x-www-form-urlencoded",
    # "Cookie": "__cfduid=dd0ef9ace9149f25ed3088bca7850f3801559805010; BAIDU_SSP_lcr=https://www.baidu.com/link?url=uH-S-eSdGL22bUvDb-0XyVSY-b-4vYadKF1hCSeVLPHuFA_gN8SF0GXKZsWPQRYp&wd=&eqid=de964d2a00035af8000000065cf8c10e; Hm_lvt_5269e069c39f6be04160a58a5db48db4=1559805983,1559806011,1559806017,1559806130; Hm_lpvt_5269e069c39f6be04160a58a5db48db4=1559806737; UM_distinctid=16b2b9e128adf-0d7d9875a65e8d8-3f6d4645-1fa400-16b2b9e128b368; CNZZDATA5176529=cnzz_eid%3D322929584-1559799756-null%26ntime%3D1559805156; _ga=GA1.2.1981161769.1559804908; _gid=GA1.2.1157098203.1559804908; directGo=1; vagueSearch=0; onlyStudent=0; _gat=1",
    "Upgrade-Insecure-Requests": "1",
    "Cache-Control": "max-age=0",
}

print(quote('法'.encode('gbk')))
r = http.request('POST',
                 'http://xh.5156edu.com/index.php',
                 headers=header, data='f_key=%B7%A8&f_type=zi&SearchString.x=27&SearchString.y=11')
print(r.status)
print(r.data.decode('gbk'))
exit(1)
link_file = open('link.txt', 'w')
char_file = open('gb2312-b.txt', 'r')
chars = char_file.readlines()
target_char = '啊'
start = False
for char in chars:
    char = char[:-1]
    if char == target_char:
        start = True
    if not start:
        continue
    query = quote(char)
    r = http.request('GET',
                     'http://zaojv.com/wordQueryDo.php?nsid=0&s=4595742426291063331&q=&wo=' + query + '&directGo=1',
                     headers=header)
    soup = BeautifulSoup(r.data.decode(), "html.parser", from_encoding="utf-8")
    content = soup.find(id='div_content')
    if content is None:
        print(char, "not find")
        continue
    else:
        print(char, 'success')
    nodes = soup.find(id='div_content').find_all('a')
    max_item = min(len(nodes), 10)
    samples = random.sample(nodes, max_item)
    link_file.write(char + " ")
    for link in samples:
        link_file.write(link['href'] + " ")
    link_file.write('\n')
