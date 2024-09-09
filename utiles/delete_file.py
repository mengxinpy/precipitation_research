import os
import glob


def delete_files_matching_pattern(folder_path, pattern):
    """
    删除指定文件夹中所有匹配特定模式的文件。

    参数:
    folder_path: str - 要搜索的文件夹路径
    pattern: str - 要匹配的模式（例如'*hour*'）
    """
    # 使用glob找到所有匹配的文件
    files_to_delete = glob.glob(os.path.join(folder_path, pattern))

    # 遍历并删除这些文件
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            print(f"Deleted {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

    # 完成后打印消息
    print("Deletion complete.")


if __name__ == '__main__':
    # 使用函数示例
    folder_path = 'C:\\ERA5\\1980-2019\\large_scale_precipitation\\'
    pattern = '*processed_day_1*'
    delete_files_matching_pattern(folder_path, pattern)
