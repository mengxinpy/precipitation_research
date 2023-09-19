import numpy as np
import cartopy.feature as cfeature
import geopandas as gpd
from shapely.geometry import MultiPolygon
import rasterio
from rasterio.features import rasterize
import matplotlib.pyplot as plt
from plot_ocean_mask import plot_ocean_mask

def create_ocean_mask_v7(lon_range, lat_range, grid_size):
    # 使用cartopy获取海洋数据
    ocean = cfeature.NaturalEarthFeature('physical', 'ocean', '110m')

    # 转换为GeoDataFrame对象
    ocean_geometries = list(ocean.geometries())
    ocean_geometry = MultiPolygon(ocean_geometries)
    ocean_gdf = gpd.GeoDataFrame(geometry=[ocean_geometry], crs="EPSG:4326")

    # 创建一个网格
    lon, lat = np.meshgrid(np.arange(lon_range[0], lon_range[1], grid_size),
                           np.arange(lat_range[0], lat_range[1], grid_size))

    # 创建一个空数组用于存储掩膜结果
    mask_shape = lat.shape
    ocean_mask = np.zeros(mask_shape, dtype=np.uint8)

    # 使用rasterio的rasterize函数生成掩膜
    transform = rasterio.transform.from_origin(lon_range[0], lat_range[1], grid_size, grid_size)
    ocean_mask = rasterize(ocean_gdf.geometry, out_shape=mask_shape, transform=transform, fill=0, default_value=1, dtype=np.uint8)

    # 将数据类型转换为bool
    ocean_mask = ocean_mask.astype(bool)

    return ocean_mask


# 用法示例
lon_range = (-180, 180)
lat_range = (-90, 90)
grid_size = 0.25
ocean_mask = create_ocean_mask_v7(lon_range, lat_range, grid_size)
# 假设您已经创建了一个海洋掩膜 'ocean_mask'
# 使用 binary_dilation 函数扩大掩膜范围
np.save('ocean_mask', ocean_mask)
