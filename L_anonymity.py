import pandas as pd
import matplotlib.pylab as pl
import matplotlib.patches as patches
import numpy as np

names = (
    'Latitude',
    'Longitude', 
    'zero', 
    'Altitude',
    'Date',
    'strDate',
    'strTime',
)

categorical = set((
    'Longitude', 
    'zero', 
    'Altitude',
    'Date',
    'strDate',
    'strTime',
))

def get_spans(df, partition, scale=None):
    spans = {}
    for column in df.columns:
        if column in categorical:
            span = len(df[column][partition].unique())
        else:
            span = df[column][partition].max()-df[column][partition].min()
        if scale is not None:
            span = span/scale[column]
        spans[column] = span
    return spans
    


def split(df, partition, column):
    dfp = df[column][partition]
    if column in categorical:
        values = dfp.unique()
        lv = set(values[:len(values)//2])
        rv = set(values[len(values)//2:])
        return dfp.index[dfp.isin(lv)], dfp.index[dfp.isin(rv)]
    else:        
        median = dfp.median()
        dfl = dfp.index[dfp < median]
        dfr = dfp.index[dfp >= median]
        return (dfl, dfr)

def is_k_anonymous(df, partition, sensitive_column, k=5):
    if len(partition) < k:
        return False
    return True

def partition_dataset(df, feature_columns, sensitive_column, scale, is_valid):
    finished_partitions = []
    partitions = [df.index]
    while partitions:
        partition = partitions.pop(0)
        spans = get_spans(df[feature_columns], partition, scale)
        for column, span in sorted(spans.items(), key=lambda x:-x[1]):
            lp, rp = split(df, partition, column)
            if not is_valid(df, lp, sensitive_column) or not is_valid(df, rp, sensitive_column):
                continue
            partitions.extend((lp, rp))
            break
        else:
            finished_partitions.append(partition)
    return finished_partitions

def build_indexes(df):
    indexes = {}
    for column in categorical:
        values = sorted(df[column].unique())
        indexes[column] = { x : y for x, y in zip(values, range(len(values)))}
    return indexes

def get_coords(df, column, partition, indexes, offset=0.1):
    if column in categorical:
        sv = df[column][partition].sort_values()
        l, r = indexes[column][sv[sv.index[0]]], indexes[column][sv[sv.index[-1]]]+1.0
    else:
        sv = df[column][partition].sort_values()
        next_value = sv[sv.index[-1]]
        larger_values = df[df[column] > next_value][column]
        if len(larger_values) > 0:
            next_value = larger_values.min()
        l = sv[sv.index[0]]
        r = next_value
    l -= offset
    r += offset
    return l, r

def get_partition_rects(df, partitions, column_x, column_y, indexes, offsets=[0.1, 0.1]):
    rects = []
    for partition in partitions:
        xl, xr = get_coords(df, column_x, partition, indexes, offset=offsets[0])
        yl, yr = get_coords(df, column_y, partition, indexes, offset=offsets[1])
        rects.append(((xl, yl),(xr, yr)))
    return rects

def get_bounds(df, column, indexes, offset=1.0):
    if column in categorical:
        return 0-offset, len(indexes[column])+offset
    return df[column].min()-offset, df[column].max()+offset

# Junior patch begin
def agg_categorical_column(series):
    return [','.join(str(v) for v in set(series))]
# Junior patch end

def agg_numerical_column(series):
    return [series.mean()]

def Average(list):
    return sum(float(v) for v in list)/len(list)


def build_anonymized_dataset(df, partitions, feature_columns, sensitive_column, max_partitions=None):
    aggregations = {}
    for column in feature_columns:
        if column in categorical:
            aggregations[column] = agg_categorical_column
        else:
            aggregations[column] = agg_numerical_column
    rows = []
    for i, partition in enumerate(partitions):
        if i % 100 == 1:
            print("Finished {} partitions...".format(i))
        if max_partitions is not None and i > max_partitions:
            break
        grouped_columns = df.loc[partition].agg(aggregations, squeeze=False)
        sensitive_counts = df.loc[partition].groupby(sensitive_column).agg({sensitive_column : 'count'})
        # Junior patch begin
        values = {}
        for latitude in grouped_columns.iloc[0]:
            LatArray = latitude.split(',')
            values.update({
                'Latitude' : Average(LatArray)
            })   
        for longitude in grouped_columns.iloc[1]:
            LongArray = longitude.split(',')
            values.update({
                'Longitude' : Average(LongArray)
            })
        # Junior patch end
        for sensitive_value, count in sensitive_counts[sensitive_column].items():
            if count == 0:
                continue
            values.update({
                sensitive_column : sensitive_value,
                'count' : count,
            })
            rows.append(values.copy())
    return pd.DataFrame(rows)



def diversity(df, partition, column):
    return len(df[column][partition].unique())

def is_l_diverse(df, partition, sensitive_column, l=2):
    return diversity(df, partition, sensitive_column) >= l

def t_closeness(df, partition, column, global_freqs):
    total_count = float(len(partition))
    d_max = None
    group_counts = df.loc[partition].groupby(column)[column].agg('count')
    for value, count in group_counts.to_dict().items():
        p = count/total_count
        d = abs(p-global_freqs[value])
        if d_max is None or d > d_max:
            d_max = d
    return d_max

def is_t_close(df, partition, sensitive_column, global_freqs, p=0.2):
    if not sensitive_column in categorical:
        raise ValueError("this method only works for categorical values")
    return t_closeness(df, partition, sensitive_column, global_freqs) <= p

def anonymity(filename):
    ###################### K-Anonymity Algorithm  #################################
    df = pd.read_csv(filename, sep=",", header=7, names=names, index_col=False, engine='python');
    print(df.head())

    for name in categorical:
        df[name] = df[name].astype('category')
    #### Implement a function that returns the spans (max-min for numerical columns, 
    # number of different values for categorical columns) of all columns for a partition of a dataframe
    full_spans = get_spans(df, df.index)
    print(full_spans)

    # we apply our partitioning method to two columns of our dataset, using "income" as the sensitive attribute
    feature_columns = ['Longitude', 'Altitude']
    sensitive_column = 'Date'
    finished_partitions = partition_dataset(df, feature_columns, sensitive_column, full_spans, is_k_anonymous)

    indexes = build_indexes(df)
    column_x, column_y = feature_columns[:2]
    rects = get_partition_rects(df, finished_partitions, column_x, column_y, indexes, offsets=[0.0, 0.0])
    print(rects[:10])

    #### Generating an k-Anonymous Dataset #####
    dfn = build_anonymized_dataset(df, finished_partitions, feature_columns, sensitive_column)
    #dfn.to_csv(r'2.csv', index=False, header=True)
    finished_l_diverse_partitions = partition_dataset(df, feature_columns, sensitive_column, full_spans, lambda *args: is_k_anonymous(*args) and is_l_diverse(*args))

    print(len(finished_l_diverse_partitions))

    # Implementing l-diversity (the naive way)
    column_x, column_y = feature_columns[:2]
    l_diverse_rects = get_partition_rects(df, finished_l_diverse_partitions, column_x, column_y, indexes, offsets=[0.0, 0.0])

    dfl = build_anonymized_dataset(df, finished_l_diverse_partitions, feature_columns, sensitive_column)


    global_freqs = {}
    total_count = float(len(df))
    group_counts = df.groupby(sensitive_column)[sensitive_column].agg('count')
    for value, count in group_counts.to_dict().items():
        p = count/total_count
        global_freqs[value] = p

    ##  Implementing t-closeness
    finished_t_close_partitions = partition_dataset(df, feature_columns, sensitive_column, full_spans, lambda *args: is_k_anonymous(*args) and is_t_close(*args, global_freqs))

    print(len(finished_t_close_partitions))

    dft = build_anonymized_dataset(df, finished_t_close_partitions, feature_columns, sensitive_column)
    dft.to_csv(f'{filename}_k.txt', header=None, index=None, sep=',', mode='w')
    return f'{filename}_k.txt'

