### 解释
1. **数据重采样**：
   - 使用 `resample(time='M').sum()` 将日数据转换为月数据。

2. **计算每年的季节性指数**：
   - 定义 `calculate_seasonality_index` 函数，计算每年的季节性指数。
   - 计算公式为：
     $$SI = \frac{1}{P_{total}} \sum_{i=1}^{12} \left| P_i - \frac{P_{total}}{12} \right|$$
   - 其中，$P_i$ 是第$i$个月的降水量，$P_{total}$ 是年降水总量。

3. **按年分组并计算季节性指数**：
   - 使用 `groupby('time.year').apply(calculate_seasonality_index)` 按年分组，并应用季节性指数计算函数。

4. **求多年的季节性指数的平均**：
   - 使用 `mean(dim='year')` 求多年的季节性指数的平均值。

5. **输出**：
   - 函数返回每个格点的平均季节性指数。





```python
# %load_ext autoreload
# %autoreload 2
import importlib
import auto_run

importlib.reload(auto_run)


# 然后再次使用 main_month 函数
from auto_run import main_month

```


```python
start_key = 'wet'
if start_key == 'wet30':
    percentile_key = 'lsprf'
else:
    percentile_key = start_key
data_set = 'era5'

main_month(f'wetday_month_vt_{start_key}', percentile_name=f'{percentile_key}_percentile', renew='000',
           data_set=data_set)

```


```python

```




### 解释
1. **数据重采样**：
   - 使用 `resample(time='M').sum()` 将日数据转换为月数据。

2. **计算每年的季节性指数**：
   - 定义 `calculate_seasonality_index` 函数，计算每年的季节性指数。
   - 计算公式为：
     $$SI = \frac{1}{P_{total}} \sum_{i=1}^{12} \left| P_i - \frac{P_{total}}{12} \right|$$
   - 其中，$P_i$ 是第$i$个月的降水量，$P_{total}$ 是年降水总量。

3. **按年分组并计算季节性指数**：
   - 使用 `groupby('time.year').apply(calculate_seasonality_index)` 按年分组，并应用季节性指数计算函数。

4. **求多年的季节性指数的平均**：
   - 使用 `mean(dim='year')` 求多年的季节性指数的平均值。

5. **输出**：
   - 函数返回每个格点的平均季节性指数。


### 解释
1. **数据重采样**：
   - 使用 `resample(time='M').sum()` 将日数据转换为月数据。

2. **计算每年的季节性指数**：
   - 定义 `calculate_seasonality_index` 函数，计算每年的季节性指数。
   - 计算公式为：
     $$SI = \frac{1}{P_{total}} \sum_{i=1}^{12} \left| P_i - \frac{P_{total}}{12} \right|$$
   - 其中，$P_i$ 是第$i$个月的降水量，$P_{total}$ 是年降水总量。

3. **按年分组并计算季节性指数**：
   - 使用 `groupby('time.year').apply(calculate_seasonality_index)` 按年分组，并应用季节性指数计算函数。

4. **求多年的季节性指数的平均**：
   - 使用 `mean(dim='year')` 求多年的季节性指数的平均值。

5. **输出**：
   - 函数返回每个格点的平均季节性指数。

