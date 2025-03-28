import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
from scipy.stats import pearsonr


def get_top_n_areas_by_traffic(df, area_column, vehicle_column, year_column, top_n=10):
    """
    计算每个地区所有年份的交通流量总和，并返回流量最高的前 N 个地区。

    参数：
    - df: DataFrame, 包含交通数据
    - area_column: str, 地名的列名（如 'local_authority_name')
    - vehicle_column: str, 车辆总数的列名（如 'all_motor_vehicles')
    - year_column: str, 年份的列名（如 'year')
    - top_n: int, 需要获取的最高车流量的地区数量(默认前10个)

    返回：
    - top_n_areas: DataFrame, 包含地名和总车流量的前 N 个地区
    """
    
    # 根据地名和年份进行聚合，计算车辆总和
    result = df.groupby([area_column, year_column])[vehicle_column].sum().unstack(fill_value=0)

    # 计算每个地区所有年份的车流量总和
    total_vehicles_per_area = result.sum(axis=1)

    # 转换为 DataFrame 并重置索引
    total_vehicles_per_area_df = total_vehicles_per_area.reset_index()
    total_vehicles_per_area_df.columns = [area_column, f'total_{vehicle_column}']

    # 按车流量降序排序,并选出前N个地区
    top_n_areas = total_vehicles_per_area_df.sort_values(by=f'total_{vehicle_column}', ascending=False).head(top_n)

    return top_n_areas



import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore

def plot_gdhi_vs_traffic(df_gdhi_common, df_traffic_common, local_authority_name, common_years, traffic_column):
    """
    绘制指定 local_authority_name 在指定年份范围内的 GDHI 和交通数据（标准化）

    参数：
    - df_gdhi_common: DataFrame, 包含 GDHI 数据
    - df_traffic_common: DataFrame, 包含交通数据
    - local_authority_name: str, 需要分析的地方政府名称
    - common_years: range, 分析的时间范围 (例如 range(2000, 2023))
    - traffic_column: str, 交通数据的列名（例如 'all_motor_vehicles', 'all_HGVs', 'all_cars'）
    """

    # **第一部分：GDHI 数据处理**
    gdhi_data = df_gdhi_common[df_gdhi_common['local_authority_name'] == local_authority_name]

    # 确保年份列是字符串格式
    years = [str(year) for year in common_years]
    gdhi_data = gdhi_data[years].transpose()
    gdhi_data.columns = ['GDHI']
    gdhi_data.index = gdhi_data.index.astype(int)  # 转换索引为整数年份

    # **第二部分：交通数据处理**
    traffic_data = df_traffic_common[df_traffic_common['local_authority_name'] == local_authority_name]

    # 确保 traffic_column 存在
    if traffic_column not in traffic_data.columns:
        raise ValueError(f"'{traffic_column}' 不在交通数据表的列中，请检查列名是否正确。")

    # 按年份分组，计算总和
    traffic_yearly = traffic_data.groupby('year')[traffic_column].sum().reset_index()

    # **确保年份范围一致**
    gdhi_data = gdhi_data.loc[common_years]
    traffic_yearly = traffic_yearly[traffic_yearly['year'].isin(common_years)]

    # **标准化数据**
    gdhi_data['GDHI_normalized'] = zscore(gdhi_data['GDHI'])
    traffic_yearly[f'{traffic_column}_normalized'] = zscore(traffic_yearly[traffic_column])

    # **绘图**
    sns.set(style="whitegrid")  # 设置 Seaborn 样式
    plt.figure(figsize=(10, 6))  # 调整图像大小，使其更小

    # GDHI 变化曲线
    sns.lineplot(data=gdhi_data, x=gdhi_data.index, y='GDHI_normalized',
                 marker='o', color='b', label=f'{local_authority_name} GDHI (Normalized)', linewidth=2)

    # 交通流量变化曲线
    sns.lineplot(data=traffic_yearly, x='year', y=f'{traffic_column}_normalized',
                 marker='s', color='r', label=f'{traffic_column} (Normalized)', linestyle='--', linewidth=2)

    # **设置标题和标签**
    plt.title(f'{local_authority_name} GDHI and {traffic_column} Normalized by Year', fontsize=14)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Normalized Values', fontsize=12)
    plt.xticks(list(common_years), rotation=45)  # 确保 X 轴显示所有年份
    plt.legend()

    # **调整布局并显示**
    plt.tight_layout()
    plt.show()




def compute_pearson_correlation(df_gdhi_common, df_traffic_common, local_authority_name, common_years, traffic_column):
    """
    计算指定地区的 GDHI 和交通数据之间的 Pearson 相关系数，并返回合并后的数据。

    参数：
    - df_gdhi_common: DataFrame, 包含 GDHI 数据
    - df_traffic_common: DataFrame, 包含交通数据
    - local_authority_name: str, 需要分析的地方政府名称
    - common_years: range, 分析的时间范围 (例如 range(2000, 2023))
    - traffic_column: str, 交通数据的列名（例如 'all_motor_vehicles', 'all_HGVs'）

    返回：
    - Pearson 相关系数 (corr)
    - P 值 (p_value)
    - 合并后的 DataFrame (merged_df)
    """
    
    pd.options.mode.chained_assignment = None  # 关闭 SettingWithCopyWarning

    # **第一部分：筛选地区数据**
    traffic_data = df_traffic_common[df_traffic_common['local_authority_name'] == local_authority_name]
    gdhi_data = df_gdhi_common[df_gdhi_common['local_authority_name'] == local_authority_name]

    # **第二部分：转换 GDHI 数据（宽表转长表）**
    years = [str(year) for year in common_years]
    gdhi_long = gdhi_data[years].transpose().reset_index()
    gdhi_long.columns = ['year', 'GDHI']
    gdhi_long['year'] = gdhi_long['year'].astype(int)  # 转换年份为整数

    # **第三部分：确保数据年份匹配**
    traffic_data['year'] = traffic_data['year'].astype(int)
    gdhi_long = gdhi_long[gdhi_long['year'].isin(common_years)]
    traffic_data = traffic_data[traffic_data['year'].isin(common_years)]

    # **第四部分：检查交通数据列是否存在**
    if traffic_column not in traffic_data.columns:
        raise ValueError(f"列 '{traffic_column}' 不在交通数据表中，请检查输入的列名。")

    # **第五部分：合并数据**
    merged_df = pd.merge(traffic_data, gdhi_long, on='year')

    # **第六部分：计算相关系数**
    if len(merged_df) == 0:
        raise ValueError("合并后的数据为空，请检查输入的年份范围和地区名称。")

    corr, p_value = pearsonr(merged_df[traffic_column], merged_df['GDHI'])

    print(f"{local_authority_name} 地区的 {traffic_column} 和 GDHI 之间的 Pearson 相关系数: {corr:.4f}, P 值: {p_value:.4f}")

    return corr, p_value, merged_df



# 任意城市，任意时间范围，GDHI 和 所有交通数据列之间的 Pearson 相关性
# 找出最大相关性的一列
def compute_all_pearson_correlation(df_gdhi_common, df_traffic_common, local_authority_name, common_years, traffic_column_indices):
    """
    计算指定地区（如 'Kent'）在给定的年份范围内，GDHI 和各交通数据列之间的 Pearson 相关性，返回所有列的相关性结果。

    参数：
    - df_gdhi_common: DataFrame, 包含 GDHI 数据
    - df_traffic_common: DataFrame, 包含交通数据
    - local_authority_name: str, 需要分析的地方政府名称
    - common_years: range, 分析的时间范围 (例如 range(1997, 2023))
    - traffic_column_indices: list, 交通数据的列索引（如 [22, 23, ..., 34]，对应Excel的列）

    返回：
    - 所有列的相关性结果（字典列表）
    - 相关性最大列名（正数）
    - 相关性最大列（绝对值）
    """
    
    all_results = []
    max_corr_pos = None  # 最大正相关性
    max_p_value_pos = None
    best_column_pos = None  # 最大正相关性的列

    max_corr_abs = None  # 最大绝对值相关性
    max_p_value_abs = None
    best_column_abs = None  # 最大绝对值相关性的列

    # **第一部分：筛选地区数据**
    traffic_data = df_traffic_common[df_traffic_common['local_authority_name'] == local_authority_name]
    gdhi_data = df_gdhi_common[df_gdhi_common['local_authority_name'] == local_authority_name]

    # **第二部分：转换 GDHI 数据（宽表转长表）**
    years = [str(year) for year in common_years]
    gdhi_long = gdhi_data[years].transpose().reset_index()
    gdhi_long.columns = ['year', 'GDHI']
    gdhi_long['year'] = gdhi_long['year'].astype(int)  # 转换年份为整数

    # **第三部分：确保数据年份匹配**
    traffic_data['year'] = traffic_data['year'].astype(int)
    gdhi_long = gdhi_long[gdhi_long['year'].isin(common_years)]
    traffic_data = traffic_data[traffic_data['year'].isin(common_years)]

    # **第四部分：循环遍历各交通数据列索引，计算相关系数**
    for index in traffic_column_indices:
        # 获取 Excel 列序号对应的列名
        column_name = df_traffic_common.columns[index]

        # 检查该列是否全为0，若是则跳过
        if traffic_data[column_name].sum() == 0:
            print(f"列 {column_name} 全为 0，跳过此列。")
            continue  # 跳过全为0的列

        # **合并数据**
        merged_df = pd.merge(traffic_data, gdhi_long, on='year')

        # **计算相关系数**
        if len(merged_df) == 0:
            continue  # 如果合并后的数据为空，跳过

        corr, p_value = pearsonr(merged_df[column_name], merged_df['GDHI'])

        # 保存每列的相关性结果
        all_results.append({
            'column_name': column_name,
            'correlation': corr,
            'p_value': p_value
        })

        # **更新最大正相关性和对应的列**
        if max_corr_pos is None or corr > max_corr_pos:
            max_corr_pos = corr
            max_p_value_pos = p_value
            best_column_pos = column_name

        # **更新最大绝对值相关性**
        if max_corr_abs is None or abs(corr) > abs(max_corr_abs):
            max_corr_abs = corr
            max_p_value_abs = p_value
            best_column_abs = column_name

    # 如果没有找到合适的列，则输出提示信息
    if best_column_pos is None and best_column_abs is None:
        print(f"{local_authority_name} 地区，在年份范围 {common_years} 中没有找到有效的相关性数据。")
        return all_results, None, None, None, None

    # 打印所有列的相关性结果
    print(f"{local_authority_name} 地区，在年份范围 {common_years} 中，各列的 Pearson 相关性如下：")
    for result in all_results:
        print(f"列 {result['column_name']} 的相关性为: {result['correlation']:.4f}, P 值: {result['p_value']:.4f}")

    return all_results, best_column_pos, max_corr_pos, max_p_value_pos, best_column_abs, max_corr_abs, max_p_value_abs



# 定义封装的函数
def calculate_pearson_correlation(df_traffic, df_gdhi, years_range=range(1997, 2023), column_name='all_motor_vehicles'):
    """
    计算各地所有道路交通流量与GDHI之间的Pearson相关系数。

    参数：
    df_traffic : DataFrame, 包含交通数据
    df_gdhi : DataFrame, 包含GDHI数据
    years_range : 可选, 年份范围(默认为1997-2022)
    column_name : 可选, 要计算相关性的交通列名（默认为'all_motor_vehicles')

    返回：
    DataFrame, 包含每个local_authority_name的Pearson相关系数和P值
    """
    # 获取所有唯一的local_authority_name
    local_authorities = df_traffic['local_authority_name'].unique()

    # 准备一个列表来存储结果
    correlation_results = []

    # 对每个local_authority_name进行处理
    for la_name in local_authorities:
        # 筛选出当前local_authority_name的数据
        current_traffic = df_traffic[df_traffic['local_authority_name'] == la_name]
        current_gdhi_wide = df_gdhi[df_gdhi['local_authority_name'] == la_name]

        if current_traffic.empty or current_gdhi_wide.empty:
            continue  # 如果某个地方的数据为空，则跳过该地方

        years = [str(year) for year in years_range]  # 使用用户自定义的年份范围
        current_gdhi_long = current_gdhi_wide[years].transpose().reset_index()
        current_gdhi_long.columns = ['year', 'GDHI']

        # 将年份列转换为整数类型以便后续合并
        current_gdhi_long['year'] = current_gdhi_long['year'].astype(int)
        current_traffic['year'] = current_traffic['year'].astype(int)

        # 合并两个数据集
        merged_df = pd.merge(current_traffic, current_gdhi_long, on='year')

        if merged_df.empty or len(merged_df) < 2:  # 需要至少两个点才能计算相关性
            continue

        # 去除包含NaN或Inf的行
        merged_df = merged_df.replace([np.inf, -np.inf], np.nan).dropna(subset=[column_name, 'GDHI'])

        if merged_df.empty or len(merged_df) < 2:  # 再次检查是否还有足够的数据点
            continue

        # 计算交通流量列（如all_motor_vehicles）和GDHI之间的相关系数
        corr, p_value = pearsonr(merged_df[column_name], merged_df['GDHI'])
        
        # 将结果添加到列表中
        correlation_results.append({
            'local_authority_name': la_name,
            'pearson_corr': corr,
            'p_value': p_value,
            'column_name': column_name
        })

    # 将结果转换为DataFrame
    correlation_df = pd.DataFrame(correlation_results)

    # 按照相关系数的绝对值排序
    correlation_df['abs_pearson_corr'] = correlation_df['pearson_corr'].abs()
    correlation_df_sorted = correlation_df.sort_values(by='abs_pearson_corr', ascending=False)

    return correlation_df_sorted




def plot_traffic_gdhi_correlation(df_traffic, df_gdhi, years_range=range(1997, 2023), column_name='all_motor_vehicles'):
    """
    Calculate Pearson correlation between traffic data and GDHI and plot the results.

    Parameters:
    df_traffic : DataFrame, contains traffic data
    df_gdhi : DataFrame, contains GDHI data
    years_range : range, optional, the range of years for analysis (default is 1997-2022)
    column_name : str, the traffic column name to analyze (default is 'all_motor_vehicles')

    Returns:
    None
    """
    # Calculate Pearson correlation results
    correlation_results = calculate_pearson_correlation(df_traffic, df_gdhi, years_range, column_name)
    
    # Output the correlation results
    print(correlation_results)

    # Set the figure size
    plt.figure(figsize=(12, 10))

    # Create the barplot
    sns.barplot(x='pearson_corr', y='local_authority_name', data=correlation_results, hue='local_authority_name', 
                palette='coolwarm', orient='h')

    # Rotate the y-axis labels and adjust their font size
    plt.yticks(rotation=0, fontsize=8)  # Adjust the font size here

    # Set the plot title and labels
    plt.title(f'Pearson Correlation between {column_name} and GDHI by Local Authority', fontsize=14)
    plt.xlabel('Pearson Correlation', fontsize=12)
    plt.ylabel('Local Authority', fontsize=12)

    # Show the plot
    plt.show()
