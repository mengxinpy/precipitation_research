import cdsapi
import calendar
from subprocess import call


def idmDownloader(task_url, folder_path, file_name):
    """
    IDM下载器
    :param task_url: 下载任务地址
    :param folder_path: 存放文件夹
    :param file_name: 文件名
    :return:
    """
    # IDM安装目录
    idm_engine = r"F:\liusch\IDM\IDMan.exe"
    # 将任务添加至队列
    call([idm_engine, '/d', task_url, '/p', folder_path, '/f', file_name, '/a'])
    # 开始任务队列
    call([idm_engine, '/s'])


def download_era5_data(start_year, end_year, variable):
    c = cdsapi.Client()  # 创建用户
    dic = {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': variable,  # 使用函数参数
        'year': '',
        'month': '',
        'day': [],
        'time': [  # 小时
            '00:00', '01:00', '02:00', '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00', '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00', '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'
        ]
    }

    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            day_num = calendar.monthrange(y, m)[1]
            dic['year'] = str(y)
            dic['month'] = str(m).zfill(2)
            dic['day'] = [str(d).zfill(2) for d in range(1, day_num + 1)]

            r = c.retrieve('reanalysis-era5-single-levels', dic)
            url = r.location
            path = 'E:\\ERA5\\1980-2019\\' + variable
            filename = str(y) + str(m).zfill(2) + '.nc'
            idmDownloader(url, path, filename)


if __name__ == '__main__':
    download_era5_data(2010, 2010, 'mean_convective_precipitation_rate')
    # download_era5_data(2010, 2010, 'mean_total_precipitation_rate')
    # download_era5_data(2010, 2010, 'mean_large_scale_precipitation_rate')
