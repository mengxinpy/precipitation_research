import numpy as np


def era5_narea_ptop_klag_1deg(log_points, dr, bins, indices, area_top_per_all, sp_out):
    # 数据初始化
    result_klag = np.zeros((area_top_per_all.shape[0], len(log_points) + 2, len(bins), 100))
    raw_data = dr.values.squeeze()
    condition_wetday = raw_data > 1

    # 数据维度检验
    assert len(bins) == indices.max()
    assert indices.min() == 1
    assert result_klag.shape == (4, 27, 6, 100)

    # 执行计算
    for area_num in range(len(bins)):

        condition_area = (indices == area_num + 1)
        assert area_top_per_all.shape == (4, 6)

        for p, area_topper in enumerate(area_top_per_all):

            condition_topper = raw_data > area_topper[area_num]
            condition_topper_area = condition_topper & condition_area

            rain_area = raw_data[condition_wetday & condition_area]
            result_klag[p, 0, area_num, :] = np.nanpercentile(rain_area, np.arange(1, 101))
            rain_area_topper = raw_data[condition_topper_area & condition_wetday]
            result_klag[p, 1, area_num, :] = np.nanpercentile(rain_area_topper, np.arange(1, 101))
            assert len(log_points) == 25
            for ind, k in enumerate(log_points):
                condition_topper_area_klag = np.roll(condition_topper_area, shift=k, axis=0)
                condition_topper_area_klag = condition_topper_area_klag & condition_wetday
                condition_topper_area_klag[0:k, :, :] = False
                result_klag[p, ind + 2, area_num, :] = np.nanpercentile(raw_data[condition_topper_area_klag], np.arange(1, 101))

                print(f'area:{area_num} per:{p} time:{k}')
    np.save(sp_out, result_klag)
    return result_klag

# np.save(f'{path_out}\\result_klag_1deg_6area_topper', result_klag)
