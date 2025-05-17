import xarray as xr
from pandas import period_range

from Function_common import get_refine_da_list
from Function_common import gradient_direction_comparison
from matplotlib import pyplot as plt
import numpy as np
from Processor_percentile_core import get_duration_frequency
from Processor_percentile_core import get_wet_frequency

# 定义关键字列表并获取相应的数据数组列表
key_list = ['season', 'duration', 'wet','power']
da_list = get_refine_da_list(key_list, unify=False)

# 加载之前保存的 AR(1) 数据集
ar1_JAS = xr.open_dataarray('AR_JAS.nc')
ar1_DJF = xr.open_dataarray('AR_DJF.nc')
power_3days=xr.open_dataarray('Fourier_3_days_ratio.nc')
power_hour=xr.open_dataarray('Fourier_period_ranges_ratio.nc').sel(period_range='6-12h')
AR_Coefficient=xr.open_dataarray('AR_Coefficient.nc')
AR_C=xr.open_dataarray('AR_C.nc')

# 生成的wet和duration
dr_gen = xr.open_dataarray('generated_data_mapped.nc')
gen_wet=get_wet_frequency(dr_gen)
gen_wet.name='gen_wet'
gen_duration=np.log10(get_duration_frequency(dr_gen))
gen_duration.name='gen_duration'


# 创建一个字典来存储所有数据，包括新的 AR(1) 数据
data_dict = {
    'season': da_list[0],
    'duration': da_list[1],
    'wet': da_list[2],
    'ar1_JAS': ar1_JAS,
    'ar1_DJF': ar1_DJF,
    'power': da_list[3],
    'power_3days': power_3days,
    'gen_wet': gen_wet,
    'gen_duration': gen_duration,
    'AR_C': AR_C,
    'AR_Coefficient': AR_Coefficient,
    'power_hour': power_hour,
    'dot': gradient_direction_comparison(
        da_list[1],
        da_list[2],
        direction_threshold=0,
        magnitude_percent_threshold=0.01
    )
}
gen_duration.plot()
plt.show()
# 将字典转换为 xarray 数据集
dataset = xr.Dataset(data_dict)

# 将数据集保存为 NetCDF 文件
dataset.to_netcdf('./internal_data/data_dict.nc', engine='netcdf4')
