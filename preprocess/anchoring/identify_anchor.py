import os
import pandas as pd
from datetime import datetime
from datas.data_source import LshDB
from settings import CONFIG_PATH
from settings import administrative_divisions_patt, organization_patt, stop_words_machine


def load_anchor_lsh():  # 加载白名单企业库
    lsh_local_file = os.path.join(CONFIG_PATH, "lean_core_buyer_LSH.pickle")
    core_hash_local_file = os.path.join(CONFIG_PATH, "core_hash_local_file.pickle")
    anchor_buyer_lsh = LshDB(50, 0.90, administrative_divisions_patt, organization_patt, stop_words_machine,True)
    anchor_buyer_lsh.set_local_lean_lsh_path(lsh_local_file).load_local_lean_lsh()
    anchor_buyer_lsh.load_local_core_hash(core_hash_local_file)
    return anchor_buyer_lsh


def sketch_core_brands(df, lsh_instance, party_name="对方户名"):
    source_names = list(df[party_name].unique())

    def query_and_profile_one(buyer_name):
        hash_value = lsh_instance.make_hash(buyer_name)
        matched_ = lsh_instance.local_lsh.query(hash_value)

        if lsh_instance.is_anonymized:
            sorted_ = list(sorted(matched_, key=lambda x: int(lsh_instance.anonymized_anchor_profile[x]['核企级别'])))
            if len(sorted_) > 0:
                res = lsh_instance.anonymized_anchor_profile[sorted_[0]]
                res["是否为核企"] = True
            else:
                res = {'核企类型': '', '核企级别': '', '核企自定义行业': '', '是否为核企': False, '匿名化编号': None}
            return res
        else:
            raise NotImplementedError("当前只支持匿名化下的sketch匹配！！！")

    start = datetime.now()
    print("开始核心企业sketching画像: {}".format(start))

    matched_cores = dict((name, query_and_profile_one(name)) for name in source_names)
    df = df.merge(df[party_name].apply(lambda name: pd.Series(matched_cores[name])), left_index=True, right_index=True)
    df['核企级别'] = df['核企级别'].astype(str)
    end = datetime.now()
    print("完成核心企业sketching画像, 耗时 {}".format(end - start))
    return df


