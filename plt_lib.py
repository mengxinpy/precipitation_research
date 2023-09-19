from common_imports import *
from config import *

area_name_list = ['western_pacific', 'eastern_pacific', 'Atlantic_Ocean', 'indian_ocean']


def plt_singular(singular_matrix, th, ctr):
    for i in range(1, 4):
        plt.plot(np.square(singular_matrix[i, :]), '*')
    plt.legend(area_name_list[1:])
    plt.xlabel('kth')
    plt.xticks(range(0, 20, 2))
    plt.ylabel('singular_value')
    plt.title(th + '_' + ctr)
    plt.savefig('pcaPicture_' + th + '//decode_' + th + '_' + ctr + '.png')
    plt.close()


def loglog_singular(singular_matrix, th, ctr):
    for i in range(1, 4):
        plt.loglog(np.square(singular_matrix[i, :]), '*')
    plt.legend(area_name_list[1:])
    plt.xlabel('kth')
    plt.ylabel('singular_value_loglog')
    plt.title(th + '_' + ctr)
    plt.savefig('pcaPicture_' + th + '//decode_loglog_' + th + '_' + ctr + '.png')
    plt.close()


def draw_timeseries(v_matrix, th, a, out_num):
    # Define the start and end dates
    start_date = '2013-01-01'
    end_date = '2022-12-31'

    # Generate the complete date range
    dates = pd.date_range(start_date, end_date)

    # Define the missing dates
    missing_dates = pd.date_range('2013-05-10', '2013-05-14')
    leap_years = pd.date_range(start='2012-02-29', end=end_date, freq='4Y')  # Choose a leap year for start

    # Combine missing dates and leap years
    missing_dates = missing_dates.append(leap_years)

    # Remove the missing dates
    dates = dates[~dates.isin(missing_dates)]  # Negate the boolean series returned by isin with ~
    # dates = np.delete(dates, indices)
    # Rotate x-tick labels for better visibility
    # plt.tight_layout()
    # plt.show()
    for i in range(out_num):
        # Generate some random data
        fig, ax = plt.subplots()

        # Plotting data with dates
        ax.plot(dates, v_matrix[i])

        # Set x-axis major ticks to yearly interval
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        # plt.plot(v_matrix[i])
        save_path = 'pcaPicture_' + th + '\\' + area_name_list[a] + '\\' + area_name_list[a] + '_' + str(i) + th + 'time_series' + '.png'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.title(str(i))
        plt.savefig(save_path)
        plt.close()


def decode_matrix(data, indices):
    out_matrix = np.full(global_var, np.nan)
    out_matrix[indices[:, 0], indices[:, 1]] = data
    return out_matrix


def draw_area(reduced_matrix, indices, th, a, out_num):
    for i in range(out_num):
        for v in ['vapor', 'rain']:
            if v == 'vapor':
                decode_out_variables = np.flip(np.flipud(decode_matrix(reduced_matrix[:reduced_matrix.shape[0] // 2, i], indices)), axis=0)
            else:
                decode_out_variables = np.flip(np.flipud(decode_matrix(reduced_matrix[reduced_matrix.shape[0] // 2:, i], indices)), axis=0)  # 保存解码后的数据

            # 创建图形和坐标轴
            fig = plt.figure(figsize=(10, 5))
            ax = plt.axes(projection=ccrs.PlateCarree())

            ax.coastlines()
            vmin = np.nanmin(decode_out_variables)
            vmax = np.nanmax(decode_out_variables)

            if start_lon_list[a] < end_lon_list[a]:
                lon = np.linspace(start_lon_list[a], end_lon_list[a], global_var[1])
                if vmin < 0 < vmax:
                    div_norm = TwoSlopeNorm(vmin=float(vmin), vcenter=0, vmax=float(vmax))
                    c = ax.contourf(lon, lat, decode_out_variables, transform=ccrs.PlateCarree(), levels=100, cmap='RdBu_r', norm=div_norm)
                elif vmin > 0:
                    # cmap='RdBu_r'
                    c = ax.contourf(lon, lat, decode_out_variables, transform=ccrs.PlateCarree(), levels=100, cmap='Reds')
                else:
                    c = ax.contourf(lon, lat, decode_out_variables, transform=ccrs.PlateCarree(), levels=100, cmap='Blues')

            else:  # The segment crosses the 180/-180 longitude breakpoint
                lon1 = np.linspace(start_lon_list[a], 180, (180 - start_lon_list[a]) * 4)
                lon2 = np.linspace(-180, end_lon_list[a], ((90 - (180 - start_lon_list[a])) * 4))
                lon = np.concatenate((lon1, lon2))
                levels = np.linspace(vmin, vmax, 100)
                c = ax.contourf(lon1, lat, decode_out_variables[:, :240], transform=ccrs.PlateCarree(), levels=levels, cmap='RdYlBu_r')
                c = ax.contourf(lon2, lat, decode_out_variables[:, 240:], transform=ccrs.PlateCarree(), levels=levels, cmap='RdYlBu_r')

            # 添加颜色条
            fig.colorbar(c, ax=ax)
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
            gl.top_labels = False  # 关闭顶部的经度标签
            gl.right_labels = False  # 关闭右侧的纬度标签
            save_path = 'pcaPicture_' + th + '\\' + area_name_list[a] + '\\' + area_name_list[a] + '_' + str(i) + th + v + '.png'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.tight_layout()
            plt.title(v + '_' + str(i))
            plt.savefig(save_path)
            plt.close()
            np.save('pcaPicture_' + th + '//decode_' + v + str(a) + '_' + th, decode_out_variables)
