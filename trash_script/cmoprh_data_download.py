import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from urllib.parse import urlunparse



def download_file(url, local_filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename


def get_links(url):
    html_page = requests.get(url)
    soup = BeautifulSoup(html_page.content, 'html.parser')
    links = []
    for link in soup.findAll('a'):
        url_path = urlparse(url).path
        link_href = urljoin(url, link.get('href'))
        parsed_link_href = urlparse(link_href)
        cleaned_link_href = urlunparse((parsed_link_href.scheme, parsed_link_href.netloc, parsed_link_href.path, "", "", ""))
        if urlparse(cleaned_link_href).path.startswith(url_path):
            links.append(cleaned_link_href)
    return links


visited = set()

#
# def download_files_from_website(url, local_path):
#     links = get_links(url)
#     print(url)
#     for link in links:
#         if link not in visited:
#             visited.add(link)
#             if link.endswith('.nc'):
#                 local_filename = os.path.join(local_path, link.split('/')[-1])
#                 download_file(link, local_filename)
#             elif not link.endswith('.nc') and not link.endswith('.nc.md5'):
#                 download_files_from_website(link, local_path)

def download_files_from_website(url, local_path):
    links = get_links(url)
    print(url)
    for link in links:
        if link not in visited:
            visited.add(link)
            if link.endswith('.gz'):
                local_filename = os.path.join(local_path, link.split('/')[-1])
                download_file(link, local_filename)
            elif not link.endswith('.gz') and not link.endswith('.gz.md5'):
                download_files_from_website(link, local_path)

# 使用方式https://ftp.cpc.ncep.noaa.gov/precip/CMORPH_V1.0/BLD/0.25deg-DLY_EOD/GLB/
# url = 'https://www.ncei.noaa.gov/data/cmorph-high-resolution-global-precipitation-estimates/access/daily/0.25deg/?C=N;O=A'
url='https://ftp.cpc.ncep.noaa.gov/precip/CMORPH_V1.0/BLD/0.25deg-DLY_EOD/GLB/'
local_path = 'F:\\liusch\\CMOPRH_ftp\\'  # 你要保存文件的本地路径
download_files_from_website(url, local_path)
