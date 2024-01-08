# 爬取的年份范围
start_year = 2001
end_year = 2019

# 爬取的月份和日期范围
months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
days = ["{:02d}".format(day) for day in range(1, 32)]

base_url = "https://www.ncei.noaa.gov/data/cmorph-high-resolution-global-precipitation-estimates/access/daily/0.25deg/"

with open("download_links.txt", "w") as file:
    for year in range(start_year, end_year+1):
        for month in months:
            for day in days:
                file_url = base_url + "{}/{}/CMORPH_V1.0_ADJ_0.25deg-DLY_00Z_{}{}{}.nc\n".format(year, month, year, month, day)
                file.write(file_url)