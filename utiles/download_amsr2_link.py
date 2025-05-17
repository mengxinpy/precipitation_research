from datetime import datetime, timedelta


def generate_links(start_date, end_date, base_url, output_file):
    current_date = start_date
    with open(output_file, 'w') as f:
        while current_date < end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            year_str = current_date.strftime('%Y')
            url = base_url.format(year=year_str, date=date_str)
            f.write(url + '\n')
            current_date += timedelta(days=1)
    print(f'Link list successfully written to {output_file}')


start_date = datetime(2013, 1, 1)
end_date = datetime(2023, 1, 1)
base_url = "https://data.remss.com/amsr2/ocean/L3/v08.2/3day/{year}/RSS_AMSR2_ocean_L3_3day_{date}_v08.2.nc"
output_file = 'download_links.txt'

generate_links(start_date, end_date, base_url, output_file)
