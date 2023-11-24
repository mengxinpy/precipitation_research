# 爬取的年份范围
start_year = 2013
end_year = 2022

# 爬取的月份和日期范围
months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
days = ["{:02d}".format(day) for day in range(1, 32)]

base_url = "https://data.remss.com/amsr2/ocean/L3/v08.2/daily/"

with open("download_links_amsr2.txt", "w") as file:
    for year in range(start_year, end_year + 1):
        for month in months:
            for day in days:
                file_url = base_url + "{}/RSS_AMSR2_ocean_L3_daily_{}-{}-{}_v08.2.nc\n".format(year, year, month, day)
                file.write(file_url)
