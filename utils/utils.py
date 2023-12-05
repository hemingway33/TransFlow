from datetime import datetime
import pandas as pd
import sqlite3
import hashlib
import re
import pickle
from Logger import Logger
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


def sketch_core_brands(df, lsh_instance):
    source_names = list(df['购方名称'].unique())

    def query_and_profile_one(buyer_name):
        hash_value = lsh_instance.make_hash(buyer_name)
        matched_ = xxxxxx(lsh_instance.local_lsh,hash_value)

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
    logging.info("开始核心企业sketching画像: {}".format(start))

    matched_cores = dict((name, query_and_profile_one(name)) for name in source_names)
    df = df.merge(df['购方名称'].apply(lambda name: pd.Series(matched_cores[name])), left_index=True, right_index=True)
    df['核企级别'] = df['核企级别'].astype(str)
    df['核企识别名'] = df['匿名化编号']
    del df['匿名化编号']

    end = datetime.now()
    logging.info("完成核心企业sketching画像, 耗时 {}".format(end - start))
    del lsh_instance
    return df


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
        '购方名称': 'string',
        '发票类型': 'string',
        '类别代码': 'string',
        '发票号码': 'numeric',
        '购方税号': 'string',
        '销方名称': 'string',
        '销方税号': 'string',
        '合计金额': 'numeric',
        '合计税额': 'numeric',
        '开票日期': 'datetime',
        '开票机号': 'numeric',
        '开票人': 'string',
        '备注': 'string',
        '状态': 'string',
        '确认状态': 'string',
        '认证状态': 'string',
        '认证时间': 'string',
        '发送状态': 'string',
        '交易流水号': 'string',
        '返回码': 'string',
        '返回码说明': 'string',
        '发送时间': 'string',
        '序号': 'string',
        '商品名称': 'string',
        '规格型号': 'string',
        '单位': 'string',
        '数量': 'numeric',
        '不含税价': 'numeric',
        '金额': 'numeric',
        '税率': 'numeric',
        '税额': 'numeric',
        '含税价': 'numeric',
        '价税合计': 'numeric',
        '核企识别名': 'string',
        '核企类型': 'string',
        '核企级别': 'string',
        '关联企业名称': 'string',
        '销方开票地址': 'string',
        '核心企业行业': 'string',
        '核企行业更新时间': 'time',
        '税收编码': 'string',
        '销方银行账号': 'string'
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


def load_invoice_data(path, invoice_table_name, detail_table_name, taxnos,mode='deploy'):
    query_invoice = """  
                                  SELECT saleinvoice.invoiceid as '发票id',
                                  saleinvoice.invoicetype as '发票类型',
                                  saleinvoice.invoicecode as '类别代码',
                                  saleinvoice.invoiceno as '发票号码',
                                  saleinvoice.buyername as '购方名称',
                                  saleinvoice.buyertaxno as '购方税号',
                                  saleinvoice.sellername as '销方名称',
                                  saleinvoice.sellertaxno as  '销方税号',
                                  saleinvoice.tatolamount as '合计金额',
                                  saleinvoice.taxrate as '税率',
                                  saleinvoice.totaltax as '合计税额',
                                  saleinvoice.invoicedate as '开票日期',
                                  saleinvoice.makeinvoicedeviceno as '开票机号',
                                  saleinvoice.makeinvoiceperson as '开票人',
                                  saleinvoice.selleraddtel as '销方开票地址',
                                  saleinvoice.sellerbankno as '销方银行账号',
                                  saleinvoice.comments as '备注',
                                  case saleinvoice.cancelflag WHEN "0" then '有效' ELSE '作废' end as '状态'
                           FROM `{}` saleinvoice
                           WHERE saleinvoice.sellertaxno IN
                           ( {} )
                           ORDER BY invoicedate
                           """.format(invoice_table_name, taxnos)

    query_invoice_detail = """
                            select
                            c.invoiceid as `发票id`,
                            c.invoicedetailno as `序号`,
                            c.detaillistflag as `发票明细类型`,
                            c.WARESNAME as `商品名称`,
                            c.STANDARDTYPE as `规格型号`,
                            c.CALCUNIT as `单位`,
                            c.QUANTITY as `数量`,
                            c.NOTAXPRICE as `不含税价`,
                            c.AMOUNT as `金额`,
                            c.TAXRATE as `税率`,
                            c.TAXAMOUNT as `税额`,
                            c.TAXPRICE as `含税价`,
                            c.AMOUNT + c.TAXAMOUNT as `价税合计`,
                            c.TAXCODE as `税收编码`
                            from `{}` c
                            where c.sellertaxno in ({})
                           """.format(detail_table_name, taxnos)

    cur = sqlite3.connect(path).cursor()
    if mode == 'deploy': cur.execute("PRAGMA KEY = 'Llsinvoice16999!'")
    query = cur.execute(query_invoice)
    cols = [column[0] for column in query.description]
    results_invoice = typeConverter(pd.DataFrame.from_records(data=query.fetchall(), columns=cols), converters)
    query = cur.execute(query_invoice_detail)
    cols = [column[0] for column in query.description]
    results_detail = typeConverter(pd.DataFrame.from_records(data=query.fetchall(), columns=cols), converters)
    cur.close()
    return results_invoice, results_detail


def dump_model_to_file(model, path):
    pickle.dump(model, open(path, "wb"))


def load_model_from_file(path):
    return pickle.load(open(path, 'rb'))