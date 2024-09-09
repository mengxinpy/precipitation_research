# -*- coding: utf-8 -*-
"""
Spyder 编辑器

这是一个临时脚本文件。
"""
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
    idm_engine = "F:\liusch\IDM\IDMan.exe"
    # 将任务添加至队列
    call([idm_engine, '/d', task_url, '/p', folder_path, '/f', file_name, '/a'])
    # 开始任务队列
    call([idm_engine, '/s'])


if __name__ == '__main__':
    c = cdsapi.Client()  # 创建用户

    # 数据信息字典
    dic = {
        'product_type': 'reanalysis',  # 产品类型
        'format': 'netcdf',  # 数据格式
        'variable': ['potential_vorticity'],  # 变量名称修删下载地址看看
        'pressure_level': ['1', '2', '3', '5', '7', '10', '20', '30', '50', '70', '100', '125', '150', '175', '200', '225', '250', '300', '350', '400', '450', '500', '550', '600',
                           '650', '700', '750', '775', '800', '825', '850', '875', '900', '925', '950', '975', '1000'],
        'year': '[]',  # 年，设为空
        'month': '[]',
        'day': '[]',  # 日，设为空
        'time': [
            '00:00',

            '06:00',

            '12:00',

            '18:00',

        ],
        'area': [
            90, -180, 0,
            180,
        ],

    }

    # 1973年11、12月没下载 1995年9、10、11、12月没下载

    # 通过循环批量下载1979年到2020年所有月份数据
    for y in range(1986, 2024):  # 遍历年
        for m in [11, 12, 1, 2, 3, 4]:  # 遍历月
            day_num = calendar.monthrange(y, m)[1]  # 根据年月，获取当月日数
            # day_num = calendar.monthrange(2020, m)[1]  # 根据年月，获取当月日数
            # 将年、月、日更新至字典中
            # dic['year'] = str(y)
            dic['year'] = str(y)
            dic['month'] = str(m).zfill(2)
            dic['day'] = [str(d).zfill(2) for d in range(1, day_num + 1)]
            r = c.retrieve('reanalysis-era5-pressure-levels', dic, )  # 文件下载器
            url = r.location
            path = 'G:\\ERA5\\1980-2019\\' + 'potential_vorticity'  # 存放文件夹不修改吗
            # filename = str(y) + str(m).zfill(2) + '.nc' # 文件名
            filename = str(y) + str(m).zfill(2) + '.nc'  # 文件名
            idmDownloader(url, path, filename)  # 添加进IDM中下载
