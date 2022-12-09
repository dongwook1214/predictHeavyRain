# 데이터 처리 코드

```python
import pandas as pd
import os

CSV_DATA_PATH = r"../data/AWS/시간단위"

# column to delete
drop_columns: list[str] = ['지점', '일사(MJ/m^2)',
                           '일조(hr)', '현지기압(hPa)', '해면기압(hPa)']

# resample by 3H
resample_str = "3H"

def set_file_list(dir_path: str = "./") -> list[tuple[str, str]]:
    """get abs paths of csvs"""
    return [(path, os.path.abspath(f'{dir_path}/{path}'))
            for path in os.listdir(dir_path) if '.csv' in path]

if __name__ == '__main__':
    # get csv files and turn it to pandas
    csv_paths: list[tuple[str, str]] = set_file_list(CSV_DATA_PATH)
    csv_pandas: list[list[str, pd.DataFrame]] = \
        [[name, pd.read_csv(path, encoding='ansi')] for name, path in csv_paths]
    for name, dataframe in csv_pandas:
        print(name)

    # merge data
    full_data: pd.DataFrame = \
        pd.concat([df for _, df in csv_pandas]).drop(drop_columns, axis=1)
    print(full_data.columns)
    del csv_pandas

    # cutting some datas and indexing
    full_data.reset_index(drop=True, inplace=True)
    full_data = full_data.interpolate(method='linear', limit_direction='forward')
    full_data['일시'] = \
        pd.to_datetime(full_data['일시'], format='%Y-%m-%d', errors='raise')
    full_data = full_data.set_index(["일시"])

    # resample it to 1 hour (adding date cumulative value)
    print(full_data.info())
    full_data = full_data.resample(rule='H').mean()

    # create 3 hours data
    df_3_hour: pd.DataFrame = pd.DataFrame()

    df_3_hour['기온(°C)'] = full_data['기온(°C)'].resample(resample_str).mean()
    df_3_hour['풍향(deg)'] = full_data['풍향(deg)'].resample(resample_str).mean()
    df_3_hour['풍속(m/s)'] = full_data['풍속(m/s)'].resample(resample_str).mean()
    df_3_hour['강수량(mm)'] = full_data['강수량(mm)'].resample(resample_str).sum()
    df_3_hour['습도(%)'] = full_data['습도(%)'].resample(resample_str).mean()

    # save as csv
    df_3_hour.to_csv('./3hours.csv', encoding='ansi')
```