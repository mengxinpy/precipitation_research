def get_index_range(lon_range, lat_range, grid_size=0.25, mask_lon_start=-180, mask_lat_start=-90):
    lon_start, lon_end = lon_range
    lat_start, lat_end = lat_range

    # 计算索引范围
    lon_start_index = int((lon_start - mask_lon_start) / grid_size)
    lon_end_index = int((lon_end - mask_lon_start) / grid_size)
    lat_start_index = int((lat_start - mask_lat_start) / grid_size)
    lat_end_index = int((lat_end - mask_lat_start) / grid_size)

    return (lon_start_index, lon_end_index), (lat_start_index, lat_end_index)
#
#
# # 示例用法
# input_lon_range = (-150, -60)
# input_lat_range = (-20, 20)
# grid_size = 0.25
#
# lon_index_range, lat_index_range = get_index_range(input_lon_range, input_lat_range, grid_size)
# print("Longitude index range:", lon_index_range)
# print("Latitude index range:", lat_index_range)
