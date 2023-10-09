import requests
from bs4 import BeautifulSoup
import os
import subprocess

# 爬取的年份范围
start_year = 2001
end_year = 2019

# 爬取的月份范围
months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

base_url = "https://www.ncei.noaa.gov/data/cmorph-high-resolution-global-precipitation-estimates/access/daily/0.25deg/"

for year in range(start_year, end_year + 1):
    for month in months:
        if month in ['01', '02']:
            continue
        url = base_url + str(year) + "/" + month + "/"
        print("Scraping URL: ", url)

        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")

        for link in soup.select("a[href$='.nc']"):
            file_url = url + link['href']

            # 使用wget命令下载文件
            subprocess.run(['wget', '-P', r'F:\liusch\CMORPH_link', file_url])
