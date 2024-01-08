import subprocess


def run_script(script_path):
    subprocess.run(['python', script_path])


def main():
    # 执行第一个脚本
    run_script('lsprf_percentile.py')
    # run_script('lspf_percentile.py')
    # 执行第二个脚本
    run_script('era5_narea_ptop_klag-1deg.py')
    run_script('draw_memory_topper-1deg-6area.py')
    run_script('draw_distribution_topper-1deg-6area.py')


if __name__ == "__main__":
    main()
