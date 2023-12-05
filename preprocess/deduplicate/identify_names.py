import re
import networkx as nx
import numpy as np
import pandas as pd
from datetime import datetime
from collections import OrderedDict
from multiprocessing import cpu_count
from scipy.sparse import coo_matrix
try: from cython_acc import lcs2
except Exception as e: print("Cannot import lcs2 cython module!! : {}".format(e))

CPU_CORE_NUM = int(cpu_count() * 3 / 4)


def rename_person_party(df, name_column):  # 是否个人交易对手记录

    # 个人识别坏字符
    person_stop_words = [" ", '_', '-', '，', ',', '先生', '女士', "个人", "车牌"]
    person_stop_words.extend(['１', '２', '３', '４', '５', '６', '７', '８', '９', '０', "\\(", "\\)", "\\（", "\\）"])

    def replace_bads(text):
        # 数字
        text = re.sub(r'\d', '', text)

        # 个人特别字符
        reg = "|".join(person_stop_words)
        text = re.sub(reg, '', text)
        return text

    simplified_corp_tokens = ['公司', '有限', '大学', '院', '银行', '超市', '教会', '集团', '监狱',
                              '师大', '财大', '部队', '小学', '政府', '所', '社', '学校', '合伙',
                              '局', '中心', '站', '厂', '委员会', '电视台', '部', '店']
    simplified_corp_reg = "|".join(simplified_corp_tokens)
    # 将所有短文本个人对手标记为“个人”
    try:
        df.loc[np.logical_and.reduce(
            (
                df[name_column].apply(replace_bads).str.len() <= 3,
                ~df[name_column].str.contains(simplified_corp_reg, regex=True),
            )
        ), "交易对手是否为个人"] = True
    except Exception:
        print("重命名个人交易对手失败，将跳过", str(Exception))

    return df


def identify_self_trans(df, name_column, self_name):  # 是否同名交易
    assert isinstance(self_name, str) and len(self_name) > 3
    df[name_column] = df[name_column].apply(lambda x: "" if pd.isna(x) else x)
    df["是否同名交易"] = df[name_column].apply(lambda x: x.startswith(self_name))
    return df


def parallelize_sim_measure_and_threshold(lhs_list, rhs_list, thresh, key_word_list, head_range_list, head_plus_list, minus_range_list, minus_value_list, plus_range_list, plus_value_list, match_type, core_num=CPU_CORE_NUM):
    start = datetime.now()
    print("使用lcs2算法，当前计算类型为: {}, 使用内核数为： {}, 使用门槛值为: {} ".format(match_type, core_num, thresh))
    print("lsh_list length: {}  rhs_list length: {} ".format(len(lhs_list), len(rhs_list)))
    print("cython 并行计算相似性矩阵，使用进程数：{} 开始时间：{}".format(core_num, start))
    rows, columns, values = lcs2.sparse_sim_matrix(lhs_list, rhs_list, thresh, key_word_list, head_range_list, head_plus_list, minus_range_list, minus_value_list, plus_range_list, plus_value_list, threads_num=core_num, match_type=match_type)
    assert len(rows) == len(columns)
    sparse_sim_matrix = coo_matrix((values, (rows, columns)), shape=(len(lhs_list), len(rhs_list)))
    end = datetime.now()
    print("cython 并行计算相似性矩阵完成，进程数：{} 结束时间：{}".format(core_num, end))
    print("总耗时:{}".format(end - start))
    return sparse_sim_matrix


def identify_same_parties(df, name_column, threshold=0.85):
    df[name_column] = df[name_column].apply(lambda x: "" if pd.isna(x) else x)

    def replace_bads(text):
        dict = OrderedDict({" ": "",
                            '0': "",
                            '1': "",
                            '2': "",
                            '3': "",
                            '4': "",
                            '5': "",
                            '6': "",
                            '7': "",
                            '8': "",
                            '9': "",
                            '_': "",
                            '-': "",
                            ',': "",
                            '，': "",
                            '、': "",
                            '(': "",
                            ')': "",
                            '（': "",
                            '）': "",
                            })
        for i in dict:
            text = text.replace(i, dict[i])
        return text

    def replace_stop_words(text):
        dict = OrderedDict({
            '省': '',
            '市': '',
            '县': '',
            '自治区': '',
            '先生': '',
            '女士': '',
            "股份": '股',
            "公司": "司",
            '责任': "",
            '有限': "",
            '分店': "店",
            '连锁': '锁',
            '商店': '店',
            '分部': '部',
            '药房': '药',
            '药店': '药',
            '医药': '药',
            '超市': '超',
            '餐饮': '餐',
            '电子商务': '电商',
            '信息技术': '信技',
            '投资发展': '投发',
            '分行': '行',
            '医院': '院',
            '贸易': '易',
            '管理': '管',
            '供应链': '应',
            '零部件': '零件',
            '汽车': '汽',
            '服务': '服',
            '劳防': '劳',
            '用品': '用',
            '工程': '工',
            '工贸': '贸',
            "商贸": '贸',
            "商业": '商',
            '新材料': '新料',
            '装饰': '饰',
            '国际': '国',
            '物流': '流',
            '货物运输代理': '货代',
            '代理': '代',
            '货运': '运',
            '保健': '健',
            '经营': '经',
            "集团": '团',
            "电子": '电',
            "科技": '科',
            "制造": '造',
            "机械": '械',
            "房地产": '房',
            "开发": '开',
            "技术": '术',
            "检测": '测',
            "银行": '银行银行',
            "还款": '还款还款',
            "支付": '支付支付',
        })
        for i in dict:
            text = text.replace(i, dict[i])
        return text

    source_names = df[name_column]
    df["预处理后的名称"] = source_names.apply(replace_bads).apply(replace_stop_words)
    names = df["预处理后的名称"].unique()
    names_correspondence = df[["预处理后的名称", name_column]].groupby("预处理后的名称", as_index=True).last().to_dict("index")

    def build_sparse_sim_matrix(names, threshold=0.85, core_num=CPU_CORE_NUM):
        key_word_list = ["院", "司"]
        head_range_list = [100, 100]
        head_plus_list = [0., 0.]
        minus_range_list = [4, 4]
        plus_range_list = [6, 6]
        minus_value_list = [0.2, 0.25]
        plus_value_list = [0.0, 0.2]

        sparse_sim_mat = parallelize_sim_measure_and_threshold(names, names, threshold, key_word_list,
                                                                              head_range_list, head_plus_list,
                                                                              minus_range_list, minus_value_list,
                                                                              plus_range_list, plus_value_list,
                                                                              match_type="self_match", core_num=core_num)
        return sparse_sim_mat

    sparse_sim_matrix = build_sparse_sim_matrix(names, threshold=threshold)
    graph = nx.from_scipy_sparse_matrix(sparse_sim_matrix)

    df["统一名称"] = df["预处理后的名称"]
    df["合并成员"] = df[name_column]

    components = sorted(nx.connected_components(graph), key=len)
    for connected_component in components:
        if len(connected_component) > 1:
            connected_names = names[list(connected_component)]
            print("component: {}".format(connected_names))
            names_cloud = ';'.join([names_correspondence[name][name_column] for name in connected_names])
            unified_name = min(connected_names)
            df["统一名称"].loc[df["统一名称"].apply(lambda symbol: symbol in connected_names)] = unified_name
            df["合并成员"].loc[df["统一名称"].apply(lambda symbol: symbol in connected_names)] = names_cloud
    print('合并关联户之前的买方有：{} /n 合并之后买方数量为: {}'.format(len(df["预处理后的名称"].unique()), len(df["统一名称"].unique())))
    return df
