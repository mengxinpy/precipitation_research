import importlib
import auto_run

importlib.reload(auto_run)

# 然后再次使用 main_month 函数
from auto_run import main_month

#%%
start_key = 'wet'
if start_key == 'wet30':
    percentile_key = 'lsprf'
else:
    percentile_key = start_key
data_set = 'era5'

main_month(f'wetday_month_vt_{start_key}', percentile_name=f'{percentile_key}_percentile', renew='001',
           data_set=data_set)
