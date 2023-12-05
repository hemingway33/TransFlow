import pandas as pd
import hashlib
import re
import pickle
from Logger import Logger
import numpy as np
logging = Logger(level="info", name=__name__).getlog()


def format_query(source_query, params_dict):
    return source_query.format(**params_dict)


def concatenate_tokens(tokens, sep=','):
    if tokens is None:
        tokens = ["uunknownn"]
    if isinstance(tokens, str):
        tokens = [tokens]
    if not isinstance(tokens, list):
        tokens = list(tokens)
    if len(tokens) == 0:
        tokens = ["uunknownn"]
    return sep.join(["'%s'" % item for item in tokens])


def concatenate_numbers(numbers, sep=','):
    if numbers is None:
        numbers = [987656789]
    if not isinstance(numbers, list):
        numbers = list(numbers)
    if len(numbers) == 0:
        numbers = [987656789]
    return sep.join(["%s" % item for item in numbers])


def md5_hash(string):
    assert isinstance(string, str)
    md = hashlib.md5()
    md.update(string.encode('utf-8'))
    return md.hexdigest()


low_meanings = ['省$', '市$', '自治区$', '自治州$']
low_meaning_regex = "|".join(low_meanings)


def remove_location_unit(text):
    text = text.strip()
    return re.sub(low_meaning_regex, '', text)


def make_signature_from_random_numbers(numbers_list):
    num_str = ""
    for num in numbers_list:
        num = pd.to_numeric(num, errors='coerce')
        if pd.isna(num): num_str += "#"
        else: num_str += str(int(num))
    return md5_hash(num_str)


def anonymize_ent_name(source_ent_name):
    if pd.isna(source_ent_name): return None
    else:
        return source_ent_name[:1] + "****" + source_ent_name[-2:]


def xxxxxx(db, name):
    return db.query(name)


converters = {
        '审批结果': 'string',
        '决策时间': 'time',
        '样本标签': 'string',
        '发票id': 'string',
        '购方名称': 'string'
}


def typeConverter(df,convertersDict):
    columns = df.columns.tolist()
    for column in convertersDict:
        if column in columns:
            if convertersDict[column] is 'datetime':
                df[column] = pd.to_datetime(df[column], utc=True, infer_datetime_format=True).dt.tz_convert('Asia/Shanghai')
            if convertersDict[column] is 'string':
                df[column] = df[column].astype(str)
            if convertersDict[column] is 'numeric':
                df[column] = pd.to_numeric(df[column])
            if convertersDict[column] is 'time':
                df[column] = pd.to_datetime(df[column], infer_datetime_format=True).dt.tz_localize('Asia/Shanghai')
            if convertersDict[column] is 'datetime1':
                df[column] = pd.to_datetime(df[column], utc=True, unit='ms',  infer_datetime_format=True).dt.tz_convert('Asia/Shanghai')
    return df


def identify_trend(value_list):
    normalized_value_list = value_list / np.linalg.norm(value_list, ord=2)
    trend_filter = np.linspace(start=-1., stop=1., num=len(value_list))
    normalized_trend_filter = trend_filter / np.linalg.norm(trend_filter, ord=2)
    return np.round(np.dot(normalized_value_list, normalized_trend_filter), 4)


def dump_model_to_file(model, path):
    pickle.dump(model, open(path, "wb"))


def load_model_from_file(path):
    return pickle.load(open(path, 'rb'))