import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(file_path):
    """
    Load data from a CSV file into a pandas DataFrame.
    
    Parameters:
    file_path (str): The path to the CSV file.
    
    Returns:
    pd.DataFrame: DataFrame containing the loaded data.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    

def clean_data(df, time_column=[]):
    """
    Clean data by removing null values and converting time column to datetime.
    
    Parameters:
    df (pd.DataFrame): DataFrame to clean.
    time_column (str): Name of the time column to convert to datetime.
    """
    df_temp = df.copy()
    # Remove null values
    df_temp.dropna(inplace=True)
    # Remove duplicates
    df_temp.drop_duplicates(inplace=True)
    # Convert time column to datetime if any
    if time_column:
        for col in time_column:
            if col in df_temp.columns:
                df_temp[col] = pd.to_datetime(df_temp[col])
    return df_temp

def plot_distribution(data, column_name, bins=30, kde=True, color='skyblue', log_scale= False):
    """
    Plots a distribution histogram with optional KDE overlay.
    
    Parameters:
    - data: pandas DataFrame
    - column_name: string, name of the numerical column to plot
    - bins: number of histogram bins
    - kde: bool, whether to overlay KDE curve
    - color: color of the histogram bars
    
    Returns:
    - Displays the plot
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column_name], bins=bins, kde=kde, color=color)
    plt.title(f'Distribution of {column_name}', fontsize=14)
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    if log_scale:
        plt.xscale('log')
        plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_bar(data, column_name, order=True, color='steelblue', rotation=45):
    """
    Plots a bar chart showing the count of each category in a specified column.

    Parameters:
    - data: pandas DataFrame
    - column_name: string, name of the categorical column
    - order: whether to sort categories by frequency
    - color: bar color
    - rotation: x-axis label rotation angle
    """
    plt.figure(figsize=(10, 6))
    
    if order:
        category_order = data[column_name].value_counts().index
    else:
        category_order = None

    sns.countplot(data=data, x=column_name, order=category_order, color=color)
    
    plt.title(f'Bar Plot of {column_name}', fontsize=14)
    plt.xlabel(column_name)
    plt.ylabel('Count')
    plt.xticks(rotation=rotation)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_violin(data, x=None, y=None, hue=None, title=None, color='lightgreen', rotate_xticks=True):  
    """
    Plots a violin plot showing the distribution of a continuous variable y by a categorical x variable split by hue categories.
    
    Parameters:
    - data: pandas DataFrame
    - x: main categorical variable (e.g., browser, source)
    - y: continuous numerical variable (e.g., purchase_value)
    - hue: subcategory for splitting (e.g., Class)
    - title: optional title for the plot
    - color: color of the violin plot
    - rotate_xticks: whether to rotate x-axis labels by 45 degrees for better readability
    """

    plt.figure(figsize=(10, 6))
    sns.violinplot(data=data, x=x, y=y, hue=hue, palette=[color])
    plt.title(title if title else f'Violin Plot of {y} vs {x}', fontsize=14)
    if rotate_xticks and x:
        plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_stacked_bar(data, x, hue):
    """
    Plots a stacked bar plot for a categorical x variable split by hue categories.
    
    Parameters:
    - data: pandas DataFrame
    - x: main categorical variable (e.g., browser, source)
    - hue: subcategory for stacking (e.g., Class)
    - palette: color scheme
    """
    counts = data.groupby([x, hue]).size().unstack(fill_value=0) # set the main category as index
    counts.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.title(f'Stacked Bar Plot: {x} by {hue}')
    plt.ylabel('Count')
    plt.xlabel(x)
    plt.xticks(rotation=45)
    plt.legend(title=hue)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

import pandas as pd

def map_ip_to_city(fraud_df, ip_map_df, ip_col='ip_address'):
    """
    Maps IPs in fraud_df to cities using ip_map_df's lower and upper bounds.
    Unmatched IPs are labeled 'Unknown'.

    Parameters:
    - fraud_df: DataFrame with IP addresses
    - ip_map_df: DataFrame with IP ranges (lower_bound, upper_bound) and city
    - ip_col: name of IP column in fraud_df

    Returns:
    - DataFrame enriched with a 'city' column
    """
    # Ensure sorting for merge_asof
    fraud_df_sorted = fraud_df.sort_values(by=ip_col).copy()
    ip_map_sorted = ip_map_df.sort_values(by='lower_bound_ip_address')

    # Use merge_asof to match based on lower_bound
    merged_df = pd.merge_asof(
        fraud_df_sorted,
        ip_map_sorted,
        left_on=ip_col,
        right_on='lower_bound_ip_address',
        direction='backward'
    )

    # Mark IPs that truly fall within the upper bound
    in_range_mask = merged_df[ip_col] <= merged_df['upper_bound_ip_address']

    # Set fallback city label
    merged_df['country'] = merged_df['country'].where(in_range_mask, 'Unknown')

    return merged_df