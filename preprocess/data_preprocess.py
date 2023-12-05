import logging
import re
from collections import OrderedDict
from datetime import datetime
import networkx as nx
import numpy as np
import pandas as pd
import pytz
from multiprocessing import cpu_count
from scipy import spatial
from scipy.sparse import coo_matrix
from textdistance import damerau_levenshtein
from configs import settings
from utils.Logger import Logger, logcode
logging = Logger(level="info", name=__name__).getlog()

try:
    from cython import lcs2
except Exception as e:
    logging.error("////////////////////////////////////////")
    logging.error("Cannot import lcs2 cython module!!")
    logging.error(e)
    logging.error("////////////////////////////////////////")


CPU_CORE_NUM = int(cpu_count() * 3 / 4)

"""
检测和处理数据质量问题工具集
发票数据标准化处理工具箱
"""


class DataQualityManager(object):

    @staticmethod
    def month_diff(old_date, new_date):
        return new_date.year * 12 + new_date.month - old_date.year * 12 - old_date.month

    # 省/市/县 主题词
    low_meanings = ['省$', '市$', '自治区$', '自治州$']
    low_meaning_regex = "|".join(low_meanings)

    @staticmethod
    def remove_location_unit(text):
        text = text.strip()
        return re.sub(DataQualityManager.low_meaning_regex, '', text)

    @staticmethod
    def extract_city(text, hierarchical_locations_dict, first=20):
        text = text[0:first]
        for province in hierarchical_locations_dict:
            if DataQualityManager.remove_location_unit(province) in text:
                for city in hierarchical_locations_dict[province]['sub']:
                    if DataQualityManager.remove_location_unit(city) in text:
                        return province + city, hierarchical_locations_dict[province]['sub'][city] + '00'
                first_item = next(iter(hierarchical_locations_dict[province]['sub'].items()))
                return province + first_item[0], first_item[1] + '00'

        for province in hierarchical_locations_dict:
            for city in hierarchical_locations_dict[province]['sub']:
                if DataQualityManager.remove_location_unit(city) in text:
                    return province + city, hierarchical_locations_dict[province]['sub'][city] + '00'

        return None, None

    @staticmethod
    def rename_person_buyer(df, name_column):  # 重命名个人买方数据行

        # 个人买方识别坏字符
        person_stop_words = [" ", '_', '-', '，', ',', '先生', '女士', "个人", "车牌"]
        person_stop_words.extend(['１', '２', '３', '４', '５', '６', '７', '８', '９', '０', "\\(", "\\)", "\\（", "\\）"])

        def replaceBads(text):
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
        # 将所有短文本个人买家标记为“个人”
        try:
            df.loc[np.logical_and.reduce(
                (
                    df[name_column].apply(replaceBads).str.len() <= 3,
                    ~df[name_column].str.contains(simplified_corp_reg, regex=True),
                )
            ), name_column] = "个人"
        except Exception:
            print("重命名个人买方失败， 跳过", str(Exception))
            pass

        return df

    @staticmethod
    def dropAffliates(df, affliates, name_column):
        affliated_df = df[df[name_column].isin(affliates)]
        df = df[~df[name_column].isin(affliates)]
        return df, affliated_df

    @staticmethod
    def parallelize_sim_measure_and_threshold(lhs_list, rhs_list, thresh, key_word_list, head_range_list, head_plus_list, minus_range_list, minus_value_list, plus_range_list, plus_value_list, match_type, core_num=CPU_CORE_NUM):
        start = datetime.now()
        logging.info("使用lcs2算法，当前计算类型为: {}, 使用内核数为： {}, 使用门槛值为: {} ".format(match_type, core_num, thresh))
        logging.info("lsh_list length: {}  rhs_list length: {} ".format(len(lhs_list), len(rhs_list)))
        logging.info("cython 并行计算相似性矩阵，使用进程数：{} 开始时间：{}".format(core_num, start))
        rows, columns, values = lcs2.sparse_sim_matrix(lhs_list, rhs_list, thresh, key_word_list, head_range_list, head_plus_list, minus_range_list, minus_value_list, plus_range_list, plus_value_list, threads_num=core_num, match_type=match_type)
        assert len(rows) == len(columns)
        sparse_sim_matrix = coo_matrix((values, (rows, columns)), shape=(len(lhs_list), len(rhs_list)))
        end = datetime.now()
        logging.info("cython 并行计算相似性矩阵完成，进程数：{} 结束时间：{}".format(core_num, end))
        logging.info("总耗时:{}".format(end - start))
        return sparse_sim_matrix

    @staticmethod
    def identifySameBuyers(df, name_column, threshold=0.85):

        def replaceBads(text):
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

        def replaceStopWords(text):
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
                "开发": '开'
            })
            for i in dict:
                text = text.replace(i, dict[i])
            return text

        source_names = df[name_column]
        df["预处理后的名称"] = source_names.apply(replaceBads).apply(replaceStopWords)
        names = df["预处理后的名称"].unique()
        names_correspondence = df[["预处理后的名称", name_column]].groupby("预处理后的名称", as_index=True).last().to_dict("index")

        def buildSparseSimMatrix(names, threshold=0.75, core_num=CPU_CORE_NUM):
            key_word_list = ["院", "司"]
            head_range_list = [100, 100]
            head_plus_list = [0., 0.]
            minus_range_list = [4, 4]
            plus_range_list = [6, 6]
            minus_value_list = [0.2, 0.25]
            plus_value_list = [0.0, 0.2]

            sparse_sim_mat = DataQualityManager.parallelize_sim_measure_and_threshold(names, names, threshold, key_word_list,
                                                                                      head_range_list, head_plus_list,
                                                                                      minus_range_list, minus_value_list,
                                                                                      plus_range_list, plus_value_list,
                                                                                      match_type="self_match", core_num=core_num)
            return sparse_sim_mat

        sparse_sim_matrix = buildSparseSimMatrix(names, threshold=threshold)
        graph = nx.from_scipy_sparse_matrix(sparse_sim_matrix)

        df["统一名称"] = df["预处理后的名称"]
        # df["合并成员"] = df[name_column]

        components = sorted(nx.connected_components(graph), key=len)
        for connected_component in components:
            if len(connected_component) > 1:
                connected_names = names[list(connected_component)]
                logging.debug("component: {}".format(connected_names))
                # names_cloud = ';'.join([names_correspondence[name][name_column] for name in connected_names])
                unified_name = min(connected_names)
                df["统一名称"][df["统一名称"].apply(lambda symbol: symbol in connected_names)] = unified_name
                # df["合并成员"][df["统一名称"].apply(lambda symbol: symbol in connected_names)] = names_cloud
        logging.debug('合并关联户之前的买方有：{} /n 合并之后买方数量为: {}'.format(len(df["预处理后的名称"].unique()), len(df["统一名称"].unique())))
        return df

    @staticmethod
    def identifySameCommodities(df, commodityNameColumn, threshold=0.9):
        def segmentCommodityName(line):
            def extractLabelandName(line):
                if line.startswith("*"):
                    try:
                        second_star = line[1:].find("*") + 1
                        seg_pos = min(second_star + 1, len(line))
                        label = line[0: seg_pos]
                        name = line[seg_pos: len(line)]
                    except:
                        return '', line
                    return label.strip('*').strip(), name
                else:
                    return '', line
            label, name = extractLabelandName(line)
            if len(name) == 0: name = "####"
            return label, name

        def buildSparseSimMatrix(names, threshold=0.75, core_num=CPU_CORE_NUM):
            key_word_list = ["@"]
            head_range_list = [4]
            head_plus_list = [0.2]
            minus_range_list = [0.]
            plus_range_list = [0.]
            minus_value_list = [0.]
            plus_value_list = [0.]

            sparse_sim_mat = DataQualityManager.parallelize_sim_measure_and_threshold(names, names, threshold,
                                                                                      key_word_list,
                                                                                      head_range_list, head_plus_list,
                                                                                      minus_range_list,
                                                                                      minus_value_list,
                                                                                      plus_range_list, plus_value_list,
                                                                                      match_type="self_match",
                                                                                      core_num=core_num)
            return sparse_sim_mat

        source_names = df[commodityNameColumn].replace(to_replace=np.nan, value="")
        if df.empty:
            df["预处理后的商品分类"] = df["商品名称"]
            df["预处理后的商品名称"] = df["商品名称"]
        else:
            df = df.merge(source_names.apply(lambda name: pd.Series(data=list(segmentCommodityName(name)), index=["预处理后的商品分类", "预处理后的商品名称"])), left_index=True, right_index=True)
        names = df["预处理后的商品名称"].unique()
        logging.info('{}:{}个'.format(logcode(94),len(names)))

        sparse_sim_matrix = buildSparseSimMatrix(names, threshold=threshold)
        graph = nx.from_scipy_sparse_matrix(sparse_sim_matrix)

        df["商品统一名称"] = df["预处理后的商品名称"]
        df["商品统一分类"] = df["预处理后的商品分类"].replace(to_replace='', value=np.nan)
        df["统一税收编码"] = df["税收编码"]

        components = sorted(nx.connected_components(graph), key=len)
        un = nx.utils.UnionFind()
        collected_labels = dict()

        for component in components:
            component = names[list(component)]
            component = sorted(component, key=lambda x: (len(x), x), reverse=True)
            unified_name = component[0]
            un.union(*component)
            component_set = df["商品统一名称"].apply(lambda symbol: symbol in component)
            tax_codes = set(df["统一税收编码"][component_set].replace('', np.nan).dropna().unique())

            collected_labels[unified_name] = {"统一税收编码": '/'.join(tax_codes),
                                              "商品统一分类": '/'.join(df["商品统一分类"][component_set].replace('', np.nan).dropna().unique())}

        logging.info('{}:{}种。'.format(logcode(95), len(df["商品统一名称"].unique())))

        df["商品统一名称"] = df["商品统一名称"].apply(lambda co: un.__getitem__(co))
        df["商品统一分类"] = df["商品统一名称"].apply(lambda co: collected_labels[un.__getitem__(co)]["商品统一分类"])
        return df

    @staticmethod
    def extract_phone_no(address):
        tel_phones = re.findall(r'\d+', str(address))
        tel_phone = max(tel_phones, key=len) if len(tel_phones) > 0 else None
        tel_phone = tel_phone if len(tel_phone) > 5 else None
        return tel_phone

    @staticmethod
    def extract_bank_and_no(bank_address):
        if pd.isna(bank_address): return None, None
        bank = re.sub(r'[0-9]', '', str(bank_address))
        bank_nos = re.findall(r'[\d\s]+', str(bank_address))
        bank_no = max(bank_nos, key=len) if len(bank_nos) > 0 else None
        return bank.replace(" ", "").strip() if bank is not None else None, bank_no.replace(" ", "").strip() if bank_no is not None else None

    @staticmethod
    def cleanCoreBrands(coreBrandsDict):
        # 处理核企识别名
        def replaceBads(text):
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

        def replaceStopWords(text):
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
                "开发": '开'
            })
            for i in dict:
                text = text.replace(i, dict[i])
            return text

        for brand in list(coreBrandsDict):
            processedBrand = replaceStopWords(replaceBads(brand.strip()))
            coreBrandsDict[processedBrand] = coreBrandsDict.pop(brand)
        return coreBrandsDict

    @staticmethod
    def cleanRelatedCompany(relatedCompanyDF):
        # 处理关联企业识别名
        def replaceBads(text):
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

        def replaceStopWords(text):
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
                "开发": '开'
            })
            for i in dict:
                text = text.replace(i, dict[i])
            return text

        names = relatedCompanyDF['关联企业名称'].unique()
        RelatedCompanies = list()
        for company in names:
            processedBrand = replaceStopWords(replaceBads(company.strip()))
            RelatedCompanies.append(processedBrand)
        return RelatedCompanies

    @staticmethod
    def naiveIdentifyCoreBrands(df, coreBrandsDict):
        coreBrands = list(coreBrandsDict.keys())
        coreBrands = [core for core in coreBrands if len(core) >= 3]  # 确保核心企业有意义，不被默认值乱入
        unified_industry = np.nan

        def core_rize(name):
            if pd.isna(name):
                return False, name, np.nan
            else:
                searched_cores = [core for core in coreBrands if name.startswith(core)]
                if len(searched_cores) > 0:
                    unified_name = min(searched_cores, key=len)
                    unified_industry = coreBrandsDict[unified_name]['核企行业']
                    return True, unified_name, unified_industry
                else:
                    return False, name, np.nan

        names = list(df['购方名称'].unique())

        core_correspondence = {}

        for name in names:
            is_core, unified_name, unified_industry = core_rize(name)
            core_correspondence[name] = {'is_core': is_core, 'unified_name': unified_name,
                                         'core_industry': unified_industry}

        df['是否核企_征审版'] = df['购方名称'].apply(lambda name: core_correspondence[name]['is_core'])
        df['统一名称_征审版'] = df['购方名称'].apply(lambda name: core_correspondence[name]['unified_name'])
        df['核企行业_征审版'] = df['购方名称'].apply(lambda name: core_correspondence[name]['core_industry'])
        return df

    @staticmethod
    def uglyCleanCoreBrands(coreBrandsDict):
        def replaceBads(text):
            dict = OrderedDict(
                {" ": "",
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
                 '(': "",
                 ')': "",
                 '（': "",
                 '）': "",
                 })
            for i in dict:
                text = text.replace(i, dict[i])
            return text

        def replaceStopWords(text):
            dict = OrderedDict({
                "公司": "司",
                '责任': "",
                '有限': ""
            })
            for i in dict:
                text = text.replace(i, dict[i])
            return text

        for brand in list(coreBrandsDict):
            processedBrand = replaceStopWords(replaceBads(brand.strip()))
            coreBrandsDict[processedBrand] = coreBrandsDict.pop(brand)
        return coreBrandsDict

    @staticmethod
    def identifyRelatedCompanies(df, RelatedCompaniesDict, name_column, threshold=0.9):

        def simMeasure(X, Y):
            # LCS文本相似度算法
            m = len(X)
            n = len(Y)
            minValue = max(min(m, n), 1)
            maxValue = max(minValue, max(m, n))
            # An (m+1) times (n+1) matrix
            C = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if X[i - 1] == Y[j - 1]:
                        C[i][j] = C[i - 1][j - 1] + 1
                    else:
                        C[i][j] = max(C[i][j - 1], C[i - 1][j])
            min_ratio = max(C[-1][-1] / float(minValue), 0.1)
            max_ratio = max(C[-1][-1] / float(maxValue), 0.1)
            return (2 * min_ratio * max_ratio) / (min_ratio + max_ratio)  # 调和平均

        # 进行关联企业匹配+打标签操作
        def simCalc(name, processedBrand):
            try:
                assert not pd.isna(name)
            except AssertionError:
                print('注意：有统一名称为空！')
                name = ''
            try:
                assert not pd.isna(processedBrand)
            except AssertionError:
                print('注意：有关联企业简化名为空！')
                processedBrand = ''
            sim = 0
            if len(name) >= 3 and len(processedBrand) >= 3:
                if not any([(key in processedBrand) for key in ['院']]):  # 修正因医院法院名字过于接近地名造成的核心企业误判
                    name_head = name[: np.minimum(len(name), len(processedBrand))]
                else:
                    name_head = name
                sim = simMeasure(name_head, processedBrand)
                # 医院调整为相对精确匹配
                if ('院' in name_head) and ('院' in processedBrand):
                    i_index = name_head.index('院')
                    j_index = processedBrand.index('院')  # 析出第一个'院'
                    if name_head[max(i_index - 4, 0):i_index] != processedBrand[max(j_index - 4, 0):j_index]:
                        sim = sim - 0.15
            return sim

        names = list(df[name_column].unique())
        RelatedCompanies = list(RelatedCompaniesDict)

        # 进行关联企业对照表构建
        toRelatedMap = {}
        if len(RelatedCompanies) > 0:
            for name in names:
                sim_list = list(map(lambda x: simCalc(name, x), RelatedCompanies))
                try:
                    max_sim = max(sim_list)
                    max_pos = sim_list.index(max_sim)
                    if max_sim >= threshold:
                        toRelatedMap[name] = RelatedCompanies[max_pos]
                        print('所考察统一名称与关联企业最匹配：{}----->{}'.format(name, RelatedCompanies[max_pos]))
                    else:
                        toRelatedMap[name] = np.nan
                except ValueError:
                    toRelatedMap[name] = np.nan
                    print('此次关联企业撞库未成功进行，请检查原因！ 已设置成默认空值。')

            df['关联企业识别名'] = df['统一名称'].apply(lambda nam: toRelatedMap[nam])
        else:
            df['关联企业识别名'] = np.nan
        # # 去重关联企业
        # RelatedCompaniesAll=RelatedCompanies + [item for item in toRelatedMap if item not in RelatedCompanies]
        # return RelatedCompaniesAll
        # 映射回原表
        df['是否为关联企业'] = df['关联企业识别名'].notnull()
        return df

    @staticmethod
    def getCoreBrandsBuyers(valid_df, observeTime, to_observe_window=pd.Timedelta(365, unit='D')):
        window_condition = np.logical_and((observeTime - to_observe_window) <= valid_df['年月'],
                                          valid_df['年月'] < observeTime)
        windowed_df = valid_df[window_condition]
        windowed_df_with_CoreBrands = windowed_df[windowed_df['是否为核企']]

        df_valid_with_CoreBrands_1_2_3 = valid_df[np.logical_or(np.logical_or(valid_df['核企级别'] == '1',
                                                                              valid_df['核企级别'] == '2'),
                                                                valid_df['核企级别'] == '3')]
        df_valid_by_yearMonth_for_1_2_3 = DataQualityManager.groupByMultipleColumns(
            df=df_valid_with_CoreBrands_1_2_3,
            sorted_columns_list=['年月'],
            aggregate_method='sum',
            as_index=False)

        if not windowed_df_with_CoreBrands.empty:
            coreBrands = windowed_df_with_CoreBrands['核企识别名'].unique()
            windowed_df_with_CoreBrands_1 = windowed_df_with_CoreBrands[windowed_df_with_CoreBrands['核企级别'] == '1']
            coreBrands_1 = windowed_df_with_CoreBrands_1['核企识别名'].unique()
            windowed_df_with_CoreBrands_2 = windowed_df_with_CoreBrands[windowed_df_with_CoreBrands['核企级别'] == '2']
            coreBrands_2 = windowed_df_with_CoreBrands_2['核企识别名'].unique()
            windowed_df_with_CoreBrands_3 = windowed_df_with_CoreBrands[windowed_df_with_CoreBrands['核企级别'] == '3']
            coreBrands_3 = windowed_df_with_CoreBrands_3['核企识别名'].unique()
            windowed_df_with_CoreBrands_4 = windowed_df_with_CoreBrands[windowed_df_with_CoreBrands['核企级别'] == '4']
            coreBrands_4 = windowed_df_with_CoreBrands_4['核企识别名'].unique()
            windowed_df_with_CoreBrands_5 = windowed_df_with_CoreBrands[windowed_df_with_CoreBrands['核企级别'] == '5']
            coreBrands_5 = windowed_df_with_CoreBrands_5['核企识别名'].unique()

            windowed_df_by_yearMonth_CoreBrands = DataQualityManager.groupByMultipleColumns(
                df=windowed_df_with_CoreBrands,
                sorted_columns_list=['统一名称', '年月'],
                aggregate_method='sum',
                as_index=False)

            windowed_df_by_yearMonth_CoreBrands_1_buyer = DataQualityManager.groupByMultipleColumns(
                df=windowed_df_with_CoreBrands_1,
                sorted_columns_list=['统一名称', '年月'],
                aggregate_method='sum',
                as_index=False)
            windowed_df_by_yearMonth_CoreBrands_2_buyer = DataQualityManager.groupByMultipleColumns(
                df=windowed_df_with_CoreBrands_2,
                sorted_columns_list=['统一名称', '年月'],
                aggregate_method='sum',
                as_index=False)
            windowed_df_by_yearMonth_CoreBrands_3_buyer = DataQualityManager.groupByMultipleColumns(
                df=windowed_df_with_CoreBrands_3,
                sorted_columns_list=['统一名称', '年月'],
                aggregate_method='sum',
                as_index=False)
        else:
            windowed_df_by_yearMonth_CoreBrands = windowed_df_with_CoreBrands
            windowed_df_by_yearMonth_CoreBrands_1_buyer = windowed_df_with_CoreBrands
            windowed_df_by_yearMonth_CoreBrands_2_buyer = windowed_df_with_CoreBrands
            windowed_df_by_yearMonth_CoreBrands_3_buyer = windowed_df_with_CoreBrands
            df_valid_by_yearMonth_CoreBrands_1_2_3 = windowed_df_with_CoreBrands
            coreBrands = np.array([])
            coreBrands_1 = np.array([])
            coreBrands_2 = np.array([])
            coreBrands_3 = np.array([])
            coreBrands_4 = np.array([])
            coreBrands_5 = np.array([])

        return windowed_df_by_yearMonth_CoreBrands, df_valid_by_yearMonth_for_1_2_3, windowed_df_by_yearMonth_CoreBrands_1_buyer, \
               windowed_df_by_yearMonth_CoreBrands_2_buyer, windowed_df_by_yearMonth_CoreBrands_3_buyer, \
               coreBrands, coreBrands_1, coreBrands_2, coreBrands_3, coreBrands_4, coreBrands_5

    @staticmethod
    def yearMonthlize(df, dateColumn):
        tz = pytz.timezone('Asia/Shanghai')
        df["年月"] = df[dateColumn].apply(
            lambda x: tz.localize(datetime(x.year, x.month, 15)))  # .dt.tz_convert('Asia/Shanghai')
        return df

    @staticmethod
    def truncated_mean(row, required_len):
        row_ = row.values.copy()
        if len(row_) < required_len:
            row_.resize(required_len, refcheck=False)
        return np.mean(np.sort(row_)[1:-1])

    @staticmethod
    def groupByMultipleColumns(df, sorted_columns_list, aggregate_method='sum', as_index=True, end=None, start=None,
                               sum_column=None,invoice_num=False):
        if not isinstance(sorted_columns_list, list):
            sorted_columns_list = [sorted_columns_list]
        df1 = pd.DataFrame(columns=df.columns.values)
        if aggregate_method is 'sum':
            df1 = df.groupby(by=sorted_columns_list, as_index=as_index).sum()
        if aggregate_method is 'mean':
            df1 = df.groupby(by=sorted_columns_list, as_index=as_index).mean()
        if aggregate_method is 'size':
            df1 = df.groupby(by=sorted_columns_list, as_index=as_index).size()
            if isinstance(df1, pd.Series):
                df1 = df1.to_frame()
        if aggregate_method is 'last':
            df1 = df.groupby(by=sorted_columns_list, as_index=as_index).last()
        if aggregate_method is 'first':
            df1 = df.groupby(by=sorted_columns_list, as_index=as_index).first()
        if aggregate_method is 'median':
            df1 = df.groupby(by=sorted_columns_list, as_index=as_index).median()
        if aggregate_method is 'min':
            df1 = df.groupby(by=sorted_columns_list, as_index=as_index).min()
        if aggregate_method is 'abs_sum':
            df1 = df.groupby(by=sorted_columns_list, as_index=as_index)[sum_column].apply(
                lambda row: np.sum(np.abs(row)))
        if aggregate_method is 'truncated_mean':
            df1 = df.groupby(by=sorted_columns_list, as_index=as_index)[sum_column].apply(lambda row:
                                                                                          DataQualityManager.
                                                                                          truncated_mean(row, 12))

        if sorted_columns_list == ['年月']:
            df1 = df1 if as_index else df1.set_index('年月')

            if invoice_num:
                df1 = pd.concat(
                    [df1, df.groupby('年月')['发票id'].count().to_frame().rename(columns={'发票id': '发票张数'})],
                    axis=1)
            df1 = df1 if as_index else df1.reset_index()

            if as_index:
                df1 = DataQualityManager.completeMonths(df1, start=start, end=end, set_index='年月')
            else:
                df1 = DataQualityManager.completeMonths(df1, start=start, end=end)

        return df1

    @staticmethod
    def completeMonths(df, start=None, end=None, set_index=None):
        try:
            assert not df.empty
        except AssertionError:
            logging.debug('准备填充的时序表为空!')
            return df
        # WARN: 如果要使用年月进行分组，注意补全年月数据，防止镂空，干扰计算结果：特别是mean的计算， 计算mean时要replace(to_replace=np.nan, value=0)
        if df.index.name != '年月':
            df.set_index('年月', inplace=True)
        assert not df.index.duplicated().any()
        if not pd.isna(start) and (start.year, start.month) < (df.index.min().year, df.index.min().month):
            df = df.append(pd.DataFrame(index=[start], columns=df.columns.values))
            df.index.name = '年月'
            df.sort_index(inplace=True, ascending=True)
        if not pd.isna(end) and (end.year, end.month) > (df.index.max().year, df.index.max().month):
            df = df.append(pd.DataFrame(index=[end], columns=df.columns.values))
            df.index.name = '年月'
            df.sort_index(inplace=True, ascending=True)
        df = df.resample('M').last().fillna(value=0)
        df = df.reset_index().drop_duplicates(subset='年月', keep='last')
        df['年月'] = df['年月'].apply(lambda dt: dt.replace(day=15))
        if not pd.isna(set_index):
            df = df.set_index(set_index)
        return df

    @staticmethod
    def conditionalBy(df, condition, aggregate_method='sum'):
        if aggregate_method is 'sum':
            return df[condition].sum()
        if aggregate_method is 'mean':
            return df[condition].mean()
        if aggregate_method is 'size':
            return df[condition].size()
        if aggregate_method is 'last':
            return df[condition].last()
        if aggregate_method is 'first':
            return df[condition].first()
        if aggregate_method is 'median':
            return df[condition].median()
        if aggregate_method is 'none':
            return df[condition]

    @staticmethod
    def selectByTimeRange(df, time_column, observeTime, to_observe_time_period=pd.DateOffset(months=12)):
        condition = np.logical_and((observeTime - to_observe_time_period) <= df[time_column],
                                   df[time_column] < observeTime)
        return df[condition]

    @staticmethod
    def getMatchedInvoiceDate(df1, df2):
        def getIdDateDict(df1):
            df = DataQualityManager.groupByMultipleColumns(df1[['发票id', '开票日期', '统一名称', '是否为核企']],
                                                           sorted_columns_list=['发票id'],
                                                           aggregate_method='last',
                                                           as_index=True)
            return df.to_dict('index')

        idDateDict = getIdDateDict(df1)
        df2['开票日期'] = df2['发票id'].apply(lambda _: idDateDict[_]['开票日期'])
        df2['统一名称'] = df2['发票id'].apply(lambda _: idDateDict[_]['统一名称'])
        df2['是否为核企'] = df2['发票id'].apply(lambda _: idDateDict[_]['是否为核企'])
        return df2

    @staticmethod
    def getMatchedInvoiceDate2(df1, df2):
        def getIdDateDict(df1):
            df = DataQualityManager.groupByMultipleColumns(df1[['发票id', '开票日期', '统一名称']],
                                                           sorted_columns_list=['发票id'],
                                                           aggregate_method='last',
                                                           as_index=True)
            return df.to_dict('index')

        idDateDict = getIdDateDict(df1)
        df2['开票日期'] = df2['发票id'].apply(lambda _: idDateDict[_]['开票日期'])
        df2['统一名称'] = df2['发票id'].apply(lambda _: idDateDict[_]['统一名称'])
        # df2['是否为核企'] = df2['发票id'].apply(lambda _: idDateDict[_]['是否为核企'])
        return df2

    @staticmethod
    def classifyBuyers(valid_df, observeTime, to_observe_window=pd.Timedelta(365, unit='D'), class1_threshold=9,
                       class2_threshold=6, class3_threshold=3, class4_threshold=2, class5_threshold=1,
                       agg_top_month=None, agg_bottom_month=None):
        condition = np.logical_and((observeTime - to_observe_window) <= valid_df['年月'], valid_df['年月'] < observeTime)
        windowed_df = valid_df[condition]
        windowed_df_by_yearMonth_buyer = DataQualityManager.groupByMultipleColumns(df=windowed_df,
                                                                                   sorted_columns_list=['统一名称', '年月'],
                                                                                   aggregate_method='sum',
                                                                                   as_index=False)
        buyers_relationship_months = DataQualityManager.groupByMultipleColumns(
            df=windowed_df_by_yearMonth_buyer[windowed_df_by_yearMonth_buyer["合计金额"] > 0.000001],
            sorted_columns_list=['统一名称'],
            aggregate_method='size')
        buyers_invoice_amounts = DataQualityManager.groupByMultipleColumns(df=windowed_df,
                                                                           sorted_columns_list=['统一名称'],
                                                                           aggregate_method='sum',
                                                                           as_index=False)[['统一名称', '合计金额']]

        truncated_windowed_df_by_yearMonth_buyer = windowed_df_by_yearMonth_buyer[~windowed_df_by_yearMonth_buyer["年月"]
            .isin([agg_bottom_month, agg_top_month])]

        buyers_invoice_amounts_truncated_mean = DataQualityManager.groupByMultipleColumns(
            df=truncated_windowed_df_by_yearMonth_buyer,
            sorted_columns_list=['统一名称'],
            aggregate_method='sum',
            sum_column='合计金额',
            as_index=True).reset_index()[['统一名称', '合计金额']]

        buyers_invoice_amounts_truncated_mean["合计金额"] = buyers_invoice_amounts_truncated_mean["合计金额"] / 10
        buyers_invoice_amounts_truncated_mean = buyers_invoice_amounts_truncated_mean.rename(
            columns={'合计金额': '月均开票额_全量掐头去尾后'})

        buyers_invoice_behaviors_ = DataQualityManager.groupByMultipleColumns(df=windowed_df,
                                                                              sorted_columns_list=['统一名称'],
                                                                              aggregate_method='last',
                                                                              as_index=False)[['统一名称',
                                                                                               '核企级别',
                                                                                               '买方所在城市',
                                                                                               '近一年作废比率',
                                                                                               '近一年红冲比率']]

        buyers_invoice_amounts = pd.merge(buyers_invoice_amounts, buyers_invoice_amounts_truncated_mean, on='统一名称')
        buyers_invoice_amounts = pd.merge(buyers_invoice_amounts, buyers_invoice_behaviors_, on='统一名称')
        buyers_invoice_amounts['近一年红冲作废率合计'] = buyers_invoice_amounts['近一年作废比率'] + buyers_invoice_amounts['近一年红冲比率']
        sum_all_buyers = buyers_invoice_amounts['合计金额'].sum()

        buyers_class1 = list(buyers_relationship_months[buyers_relationship_months.values >= class1_threshold].index)
        buyers_class2 = list(
            buyers_relationship_months[np.logical_and(buyers_relationship_months.values < class1_threshold,
                                                      buyers_relationship_months.values >= class2_threshold)].index)
        buyers_class3 = list(
            buyers_relationship_months[np.logical_and(buyers_relationship_months.values < class2_threshold,
                                                      buyers_relationship_months.values >= class3_threshold)].index)
        buyers_class4 = list(
            buyers_relationship_months[np.logical_and(buyers_relationship_months.values < class3_threshold,
                                                      buyers_relationship_months.values >= class4_threshold)].index)

        buyers_class5 = list(
            buyers_relationship_months[np.logical_and(buyers_relationship_months.values < class4_threshold,
                                                      buyers_relationship_months.values >= class5_threshold)].index)

        buyer_class1_dict = dict((name, 1) for name in buyers_class1)
        buyer_class2_dict = dict((name, 2) for name in buyers_class2)
        buyer_class3_dict = dict((name, 3) for name in buyers_class3)
        buyer_class4_dict = dict((name, 4) for name in buyers_class4)
        buyer_class5_dict = dict((name, 5) for name in buyers_class5)
        buyer_class1_dict.update(buyer_class2_dict)
        buyer_class1_dict.update(buyer_class3_dict)
        buyer_class1_dict.update(buyer_class4_dict)
        buyer_class1_dict.update(buyer_class5_dict)

        buyers_invoice_amounts = DataQualityManager.tagBuyers(to_tag_df=buyers_invoice_amounts,
                                                              test_dict=buyer_class1_dict, tag_name='动态白名单等级')
        buyers_invoice_amounts['近12月开票占全量比'] = buyers_invoice_amounts['合计金额'] / sum_all_buyers

        # 开票额度累计占比达到80%的买家数量
        def necessaryBuyersNum(buyers_invoice_amounts, percent_requirement=0.8):
            if buyers_invoice_amounts.empty:
                return np.nan
            buyers_invoice_amounts = buyers_invoice_amounts.sort_values(by='合计金额', ascending=False)
            buyers_invoice_amounts['累计开票金额'] = buyers_invoice_amounts['合计金额'].cumsum()
            buyers_invoice_amounts['累计开票金额占比'] = buyers_invoice_amounts['累计开票金额'] / sum_all_buyers
            buyers_invoice_amounts['累计开票金额占比是否大于等于指定比率'] = buyers_invoice_amounts['累计开票金额占比'].apply(
                lambda x: True if x >= percent_requirement else False)
            le = len(buyers_invoice_amounts[~buyers_invoice_amounts['累计开票金额占比是否大于等于指定比率']])
            return le + 1

        necessary_80_percent_buyer_num = necessaryBuyersNum(buyers_invoice_amounts, percent_requirement=0.8)
        return windowed_df_by_yearMonth_buyer, buyers_class1, buyers_class2, buyers_class3, buyers_class4, buyers_class5, buyers_invoice_amounts, necessary_80_percent_buyer_num

    @staticmethod
    def getBigBuyers(valid_df, observeTime, to_observe_window=pd.Timedelta(365, unit='D'), big_threshold=0.05):
        windowed_df = DataQualityManager.selectByTimeRange(df=valid_df,
                                                           observeTime=observeTime,
                                                           time_column='年月',
                                                           to_observe_time_period=to_observe_window)
        windowed_df_sum_by_buyer = DataQualityManager.groupByMultipleColumns(df=windowed_df,
                                                                             sorted_columns_list=['统一名称'],
                                                                             aggregate_method='sum',
                                                                             as_index=False)
        windowed_df_sum_by_buyer['开票额占比'] = windowed_df_sum_by_buyer['合计金额'] / (
                windowed_df_sum_by_buyer['合计金额'].sum() + 0.0001)
        big_buyers_rows = windowed_df_sum_by_buyer[np.round(windowed_df_sum_by_buyer['开票额占比'], 4) >= big_threshold]
        return big_buyers_rows

    @staticmethod
    def getTopBuyers(valid_df, observeTime, to_observe_window=pd.Timedelta(365, unit='D'), top=10):
        windowed_df = DataQualityManager.selectByTimeRange(df=valid_df,
                                                           observeTime=observeTime,
                                                           time_column='年月',
                                                           to_observe_time_period=to_observe_window)
        churn_rate = DataQualityManager.churn_rate(windowed_df.set_index('开票日期'), name_column='统一名称', freq='2M')
        customer_dev_rate = DataQualityManager.customer_dev_rate(windowed_df.set_index('开票日期'), name_column='统一名称',
                                                                 freq='2M')
        windowed_df_sum_by_buyer = DataQualityManager.groupByMultipleColumns(df=windowed_df,
                                                                             sorted_columns_list=['统一名称'],
                                                                             aggregate_method='sum',
                                                                             as_index=False).sort_values(by='合计金额',
                                                                                                         ascending=False
                                                                                                         )

        total_amount = windowed_df_sum_by_buyer['合计金额'].sum()

        # 十大买方排除掉金额非正的买方
        windowed_df_sum_by_buyer = windowed_df_sum_by_buyer[windowed_df_sum_by_buyer['合计金额'] > 0]
        giniIndex = DataQualityManager.giniIndex(list(windowed_df_sum_by_buyer.head(top)['合计金额'].values))
        top_buyers_percent = windowed_df_sum_by_buyer.sort_values(by=['合计金额', '统一名称'], ascending=False).head(top)[['统一名称', '合计金额']]
        top_buyers_percent['金额占比'] = 0

        if total_amount > 0:
            top_buyers_percent['金额占比'] = np.round(top_buyers_percent['合计金额'] / (total_amount + 0.0001), 4)

        return top_buyers_percent.head(top)['统一名称'].unique(), len(
            windowed_df_sum_by_buyer['统一名称'].unique()), giniIndex, churn_rate, customer_dev_rate, top_buyers_percent

    @staticmethod
    def getTopBuyersHistory(df, observeTime, topBuyersList):
        windowed_df = DataQualityManager.selectByTimeRange(df=df,
                                                           observeTime=observeTime,
                                                           time_column='年月',
                                                           to_observe_time_period=pd.Timedelta(3650, unit='D'))

        windowed_df_by_yearMonth_buyer = DataQualityManager.groupByMultipleColumns(df=windowed_df,
                                                                                   sorted_columns_list=['统一名称', '年月'],
                                                                                   aggregate_method='sum',
                                                                                   as_index=False)
        top_buyers_yearMonth = windowed_df_by_yearMonth_buyer[
            windowed_df_by_yearMonth_buyer['统一名称'].isin(topBuyersList)]
        return top_buyers_yearMonth.fillna(value=0)

    @staticmethod
    def getTopCommodities(valid_df, observeTime, to_observe_window=pd.Timedelta(365, unit='D'), top=10):
        windowed_df = DataQualityManager.selectByTimeRange(df=valid_df,
                                                           observeTime=observeTime,
                                                           time_column='年月',
                                                           to_observe_time_period=to_observe_window)
        windowed_df_sum_by_commodity = DataQualityManager.groupByMultipleColumns(df=windowed_df,
                                                                                 sorted_columns_list=['商品统一名称'],
                                                                                 aggregate_method='sum',
                                                                                 as_index=False).sort_values(by='价税合计',
                                                                                                             ascending=False
                                                                                                             )
        windowed_df_sum_by_commodity = windowed_df_sum_by_commodity[windowed_df_sum_by_commodity['价税合计'] > 0]
        windowed_df_sum_by_commodity = windowed_df_sum_by_commodity[np.logical_not(
            np.logical_or(windowed_df_sum_by_commodity['商品统一名称'].str.contains('合计'),
                          windowed_df_sum_by_commodity['商品统一名称'].str.contains('折扣')))]

        windowed_df_sum_by_commodity = windowed_df_sum_by_commodity[
            np.logical_not(windowed_df_sum_by_commodity['商品统一名称'].str.contains('销货清单'))]

        giniIndex = DataQualityManager.giniIndex(list(windowed_df_sum_by_commodity.head(top)['价税合计'].values))
        return windowed_df_sum_by_commodity.head(top)['商品统一名称'].unique(), len(
            windowed_df_sum_by_commodity['商品统一名称'].unique()), giniIndex

    @staticmethod
    def getTopCommodityCategory(valid_df, observeTime, to_observe_window=pd.Timedelta(365, unit='D'), top=20,
                                observe_total_amount=0, name_column='统一税收编码章'):
        windowed_df = DataQualityManager.selectByTimeRange(df=valid_df,
                                                           observeTime=observeTime,
                                                           time_column='年月',
                                                           to_observe_time_period=to_observe_window)

        windowed_df = windowed_df[np.logical_not(
            np.logical_or(windowed_df['商品统一名称'].str.contains('合计'),
                          windowed_df['商品统一名称'].str.contains('折扣')))]
        windowed_df = windowed_df[np.logical_not(
            windowed_df['商品统一名称'].str.contains('销货清单'))]

        windowed_df = windowed_df[windowed_df[name_column] != '']
        windowed_df = windowed_df[windowed_df[name_column].notna()]

        windowed_df_sum_by_commodity = DataQualityManager.groupByMultipleColumns(df=windowed_df,
                                                                                 sorted_columns_list=[name_column],
                                                                                 aggregate_method='sum',
                                                                                 as_index=False).sort_values(by='价税合计',
                                                                                                             ascending=False
                                                                                                             )
        windowed_df_sum_by_commodity = windowed_df_sum_by_commodity[
            np.round(windowed_df_sum_by_commodity['价税合计'], 2) > 0]

        windowed_df_sum_by_commodity['金额占比'] = 0
        if observe_total_amount > 0:
            windowed_df_sum_by_commodity['金额占比'] = np.round(windowed_df_sum_by_commodity['价税合计'] / observe_total_amount,
                                                            4)

        windowed_df_sum_by_commodity1 = windowed_df_sum_by_commodity[[name_column, '价税合计', '金额占比']].head(top)
        return windowed_df_sum_by_commodity1

    @staticmethod
    def getTopCommodityCategoryDeviation(valid_df, commodity_list, observeTime,
                                         to_observe_window=pd.Timedelta(365, unit='D'), top=20,
                                         name_column='统一税收编码章', time_column='年月'):
        windowed_df = DataQualityManager.selectByTimeRange(df=valid_df,
                                                           observeTime=observeTime,
                                                           time_column='开票日期',
                                                           to_observe_time_period=to_observe_window)

        if len(commodity_list) <= 0:
            return np.nan, pd.DataFrame()

        windowed_df = windowed_df[windowed_df[name_column].isin(commodity_list)]

        windowed_df_sum_by_commodity3 = DataQualityManager.groupByMultipleColumns(windowed_df,
                                                                                  sorted_columns_list=[time_column],
                                                                                  aggregate_method='sum',
                                                                                  as_index=False).sort_values(by='价税合计',
                                                                                                              ascending=False)

        windowed_df_sum_by_commodity_complete = DataQualityManager.completeMonths(windowed_df_sum_by_commodity3,
                                                                                  set_index=time_column,
                                                                                  start=observeTime - to_observe_window,
                                                                                  end=observeTime - pd.Timedelta(28,
                                                                                                                 'D'))

        mean_ = windowed_df_sum_by_commodity_complete['价税合计'].mean()

        if pd.isna(mean_):
            mean_ = 0

        commodity_deviation_sum = np.abs(windowed_df_sum_by_commodity_complete['价税合计'] - mean_).mean()

        commodity_mean_deviation = np.round(commodity_deviation_sum / (mean_ + 1.0), 4)

        return commodity_mean_deviation, windowed_df_sum_by_commodity_complete

    @staticmethod
    def getTopCommoditiesHistory(df, observeTime, topCommoditiesList):
        windowed_df = DataQualityManager.selectByTimeRange(df=df,
                                                           observeTime=observeTime,
                                                           time_column='年月',
                                                           to_observe_time_period=pd.Timedelta(3650, unit='D'))

        windowed_df_by_yearMonth_commodity = DataQualityManager.groupByMultipleColumns(df=windowed_df,
                                                                                       sorted_columns_list=['商品统一名称',
                                                                                                            '年月'],
                                                                                       aggregate_method='sum',
                                                                                       as_index=False)
        if windowed_df.empty:
            windowed_df_by_yearMonth_commodity = windowed_df
        top_commodities_yearMonth = windowed_df_by_yearMonth_commodity[
            windowed_df_by_yearMonth_commodity['商品统一名称'].isin(topCommoditiesList)]
        return top_commodities_yearMonth.fillna(value=0)

    @staticmethod
    def parse_city(taxno):
        if pd.isna(taxno):
            return np.nan, np.nan

        taxno = taxno.strip()

        if len(taxno) < 15:
            return np.nan, np.nan

        if taxno[0] == 'L':
            taxno = taxno[1:]

        def fetch_code(taxno):
            if len(taxno) in [15, 16]:
                return taxno[0: 6]
            elif len(taxno) in [17, 18, 19]:
                return taxno[2: 8]
            elif len(taxno) == 20:
                return taxno[0: 6]
            else:
                return taxno[0: 6]

        geo_code = fetch_code(taxno)
        city_code = geo_code[:-2] + '00'
        city_searched = settings.CITY_CODES.values(prefix=city_code)
        if len(city_searched) == 0:
            return city_code, np.nan
        elif len(city_searched) == 1:
            return city_code, city_searched[0]
        else:
            logging.warning("税号{} 匹配城市多于1个:{}".format(taxno, city_searched))
            return city_code, city_searched[0]

    @staticmethod
    def parse_province(taxno):
        if pd.isna(taxno):
            return np.nan, np.nan

        taxno = taxno.strip()

        if len(taxno) < 15:
            return np.nan, np.nan

        if taxno[0] == 'L':
            taxno = taxno[1:]

        def fetch_code(taxno):
            if len(taxno) in [15, 16]:
                return taxno[0: 6]
            elif len(taxno) in [17, 18, 19]:
                return taxno[2: 8]
            elif len(taxno) == 20:
                return taxno[0: 6]
            else:
                return taxno[0: 6]

        geo_code = fetch_code(taxno)
        province_code = geo_code[0:2]
        # city_searched = settings.PROVINCE_CODES.values(prefix=city_code)
        province_searched = settings.PROVINCE_CODES.values(prefix=province_code)

        if len(province_searched) == 0:
            return province_code, np.nan
        elif len(province_searched) == 1:
            return province_code, province_searched[0]
        else:
            logging.warning("税号{} 匹配省份多于1个:{}".format(taxno, province_searched))
            return province_code, province_searched[0]

    @staticmethod
    def get_city_name(city_code):
        if pd.isna(city_code) or city_code == '':
            return np.nan

        if not isinstance(city_code, str):
            city_code = str(city_code)

        city_code = city_code.strip()

        city_searched = settings.CITY_CODES.values(prefix=city_code)
        if len(city_searched) == 0:
            return np.nan
        elif len(city_searched) == 1:
            return city_searched[0]
        else:
            logging.warning("城市号{} 匹配城市多于1个:{}".format(city_code, city_searched))
            return city_searched[0]

    @staticmethod
    def get_mode_one(modes_series):
        if modes_series.empty:
            return np.nan
        else:
            return modes_series.iloc[0]

    @staticmethod
    def computeBuyerAnomalousInvoiceRatios(df1, observe_time, period_days=365):
        buyers = list(df1['统一名称'].unique())
        buyers_anomalous_inovice_ratios = dict()

        df1_valid_source = df1[df1["状态"] == "有效"]
        df1_cancel_source = df1[df1["状态"] == "作废"]
        df1_redCorrection_source = df1[np.logical_and(df1["合计金额"] < 0, df1["状态"] == "有效")]

        df1_valid_source_latest = DataQualityManager.selectByTimeRange(df1_valid_source,
                                                                       time_column='年月',
                                                                       observeTime=observe_time,
                                                                       to_observe_time_period=pd.Timedelta(period_days,
                                                                                                           unit='D'))
        df1_cancel_source_latest = DataQualityManager.selectByTimeRange(df1_cancel_source,
                                                                        time_column='年月',
                                                                        observeTime=observe_time,
                                                                        to_observe_time_period=pd.Timedelta(period_days,
                                                                                                            unit='D'))
        df1_redCorrection_source_latest = DataQualityManager.selectByTimeRange(df1_redCorrection_source,
                                                                               time_column='年月',
                                                                               observeTime=observe_time,
                                                                               to_observe_time_period=pd.Timedelta(
                                                                                   period_days, unit='D'))

        buyers_valid = DataQualityManager.groupByMultipleColumns(df=df1_valid_source_latest,
                                                                 sorted_columns_list=['统一名称'],
                                                                 aggregate_method='abs_sum',
                                                                 sum_column='合计金额',
                                                                 as_index=True).to_dict()  # series的to_dict方法与dataframe不一样

        buyers_valid_naked_sum = DataQualityManager.groupByMultipleColumns(df=df1_valid_source_latest,
                                                                           sorted_columns_list=['统一名称'],
                                                                           aggregate_method='sum',
                                                                           sum_column='合计金额',
                                                                           as_index=True)[
            '合计金额'].to_dict()  # series的to_dict方法与dataframe不一样

        buyers_cancel = DataQualityManager.groupByMultipleColumns(df=df1_cancel_source_latest,
                                                                  sorted_columns_list=['统一名称'],
                                                                  aggregate_method='abs_sum',
                                                                  sum_column='合计金额',
                                                                  as_index=True).to_dict()

        buyers_redCorrection = DataQualityManager.groupByMultipleColumns(df=df1_redCorrection_source_latest,
                                                                         sorted_columns_list=['统一名称'],
                                                                         aggregate_method='abs_sum',
                                                                         sum_column='合计金额',
                                                                         as_index=True).to_dict()

        for buyer in buyers:
            try:
                buyer_valid_ = buyers_valid[buyer]
            except:
                buyer_valid_ = np.nan
            try:
                buyer_valid_naked_sum_ = np.abs(buyers_valid_naked_sum[buyer])
            except:
                buyer_valid_naked_sum_ = np.nan
            try:
                buyer_cancel_ = np.abs(buyers_cancel[buyer])
            except:
                buyer_cancel_ = 0
            try:
                buyers_redCorrection_ = buyers_redCorrection[buyer]
            except:
                buyers_redCorrection_ = 0

            buyers_anomalous_inovice_ratios[buyer] = {
                '近一年红冲率': np.round(buyers_redCorrection_ / (buyer_valid_ + 0.0001), 4),
                '近一年作废率': np.round(buyer_cancel_ / (buyer_valid_naked_sum_ + buyer_cancel_ + 0.0001), 4)}
        return buyers_anomalous_inovice_ratios

    @staticmethod
    def identify_trend(value_list):
        normalized_value_list = value_list / np.linalg.norm(value_list, ord=2)
        trend_filter = np.linspace(start=-1., stop=1., num=len(value_list))
        normalized_trend_filter = trend_filter / np.linalg.norm(trend_filter, ord=2)
        return np.round(np.dot(normalized_value_list, normalized_trend_filter), 4)

    @staticmethod
    def pairNamesListsSimiliarity(lhs_NameList, rhs_NameList):
        lhs_set = set(lhs_NameList)
        rhs_set = set(rhs_NameList)
        intersection = lhs_set.intersection(rhs_set)
        sum_size = len(lhs_set) + len(rhs_set)
        inter_size = len(intersection)
        if sum_size > 0:
            return round(inter_size * 2 / float(sum_size),4)
        else:
            return 0.0

    @staticmethod
    def orderedPairNamesListsSimiliarity(lhs_NameList, rhs_NameList):
        if len(lhs_NameList) == 0 and len(rhs_NameList) == 0:
            return np.nan
        return 1.0 - np.round(
            damerau_levenshtein.distance(lhs_NameList, rhs_NameList) / max(len(lhs_NameList), len(rhs_NameList)), 4)

    @staticmethod
    def pairNumberSimilarity(lhs_df, rhs_df, columnName, numberName):
        lhs_names = lhs_df[columnName].unique().tolist()
        rhs_names = rhs_df[columnName].unique().tolist()

        names_list = list(set().union(lhs_names, rhs_names))

        names_list = [n for n in names_list if len(n) > 0 and str(n) != 'nan' and str(n) != 'null']

        for name in names_list:
            if name not in lhs_names:
                lhs_df = lhs_df.append(pd.Series([name, 0], index=[columnName, numberName]), ignore_index=True)
            if name not in rhs_names:
                rhs_df = rhs_df.append(pd.Series([name, 0], index=[columnName, numberName]), ignore_index=True)

        nameIndex = dict(zip(names_list, range(len(names_list))))
        lhs_df['rank'] = lhs_df[columnName].map(nameIndex)
        rhs_df['rank'] = rhs_df[columnName].map(nameIndex)
        lhs_df.sort_values(by='rank', ascending=True, inplace=True, na_position='last')
        rhs_df.sort_values(by='rank', ascending=True, inplace=True, na_position='last')

        lhs_df_compute, rhs_df_compute = lhs_df.loc[~pd.isna(lhs_df['rank']), :].copy(), rhs_df.loc[
                                                                                         ~pd.isna(rhs_df['rank']),
                                                                                         :].copy()
        lhs_num = lhs_df_compute[numberName].tolist()
        rhs_num = rhs_df_compute[numberName].tolist()

        similarity = np.round(spatial.distance.cosine(lhs_num, rhs_num), 4)

        if len(names_list) == 1 and len(lhs_df) == 1 and len(rhs_df) == 1:
            return 1.0

        return 1.0 - similarity

    @staticmethod
    def sequenceRegularity(discreteValueList, ngram=3):
        return np.nan

    @staticmethod
    def giniIndex(topNPercentageList):
        # 算法复杂度 O(n**2)
        g = np.nan
        if len(topNPercentageList) >= 2:
            # Mean absolute difference
            mad = np.abs(np.subtract.outer(topNPercentageList, topNPercentageList)).mean()
            # Relative mean absolute difference
            rmad = mad / np.mean(topNPercentageList)
            # Gini coefficient
            g = 0.5 * rmad
        else:
            g = 1.0
        return g

    @staticmethod
    def countZeros(ValueList):
        ValueList = np.array(ValueList)
        zeroList = np.where(np.round(ValueList, 6) > 0, 0, 1)
        return zeroList.sum()

    @staticmethod
    def longestContinuousZerosLen(ValueList):
        ValueList = np.array(ValueList)
        zeroList = np.where(np.round(ValueList, 6) > 0, 1, 0)
        max = 0
        counter = 0
        for elem in zeroList:
            if elem == 0:
                counter += 1
            else:
                counter = 0
            if max < counter:
                max = counter
        return max

    @staticmethod
    def findMissingIDs(invoice_IDs, detail_IDs):
        invoice_ID = set(invoice_IDs)
        detail_ID = set(detail_IDs)
        missed = list(invoice_ID - detail_ID)
        return missed, len(missed) / float(len(invoice_ID) + 1)


    @staticmethod
    def parseInterval(intervalString):
        sourceIntervalStr = intervalString
        intervalString = intervalString.strip()
        left_bracket = intervalString[0]
        intervalString = intervalString.replace(left_bracket, '')
        right_bracket = intervalString[-1]
        intervalString = intervalString.replace(right_bracket, '')
        lhs_str, rhs_str = intervalString.split(',')
        lhs_str = lhs_str.strip()
        rhs_str = rhs_str.strip()
        lhs, rhs = np.nan, np.nan
        if lhs_str is '-inf':
            lhs = -np.inf
        else:
            lhs = np.float(lhs_str)
        if rhs_str is 'inf':
            rhs = np.inf
        else:
            rhs = np.float(rhs_str)
        try:
            assert not np.isnan(lhs)
            assert not np.isnan(rhs)
            assert isinstance(lhs, float)
            assert isinstance(rhs, float)
        except AssertionError:
            print('当前区间 {} 解析有误，请检查原因！'.format(sourceIntervalStr))
            exit(1)
        return left_bracket, lhs, rhs, right_bracket

    @staticmethod
    def isBetween(interval, value):
        if interval[0] is '(':
            if interval[3] is ')':
                return bool(np.logical_and(value > interval[1], value < interval[2]))

        if interval[0] is '(':
            if interval[3] is ']':
                return bool(np.logical_and(value > interval[1], value <= interval[2]))

        if interval[0] is '[':
            if interval[3] is ')':
                return bool(np.logical_and(value >= interval[1], value < interval[2]))

        if interval[0] is '[':
            if interval[3] is ']':
                return bool(np.logical_and(value >= interval[1], value <= interval[2]))

    @staticmethod
    def ceilTheOutlier(values, lower_bound, upper_bound):
        return np.clip(values, a_min=lower_bound, a_max=upper_bound)

    @staticmethod
    def tagBuyers(to_tag_df, test_dict, tag_name):
        tests = list(test_dict.keys())

        def tag(name):
            if (not pd.isna(name)) and (name in tests):
                return test_dict[name]

        to_tag_df[tag_name] = to_tag_df['统一名称'].apply(lambda name: tag(name)).replace(to_replace=np.nan, value=0)
        return to_tag_df

    @staticmethod
    def keepOnlySomeCommodity(source_invoice_detail_df, commodity_column, match_flags_list, source_invoice_df,
                              exclude_flags_list):
        if not isinstance(match_flags_list, list):
            match_flags_list = [match_flags_list]

        if not isinstance(exclude_flags_list, list):
            exclude_flags_list = [exclude_flags_list]

        def hasFlags(commodity):
            if pd.isna(commodity) or commodity == '':
                return False
            if any(flag in commodity for flag in match_flags_list) and all(
                    flag_ not in commodity for flag_ in exclude_flags_list):
                return True
            else:
                return False

        targeted_source_invoice_detail_df = source_invoice_detail_df[
            source_invoice_detail_df[commodity_column].apply(hasFlags)]
        targeted_source_invoice_df = source_invoice_df[
            source_invoice_df['发票id'].isin(list(targeted_source_invoice_detail_df['发票id'].unique()))]
        return targeted_source_invoice_df, targeted_source_invoice_detail_df

    @staticmethod
    def keepOnlySpecifiedCoreBuyers(source_invoice_detail_df):
        return source_invoice_detail_df[source_invoice_detail_df['是否为特种核企'] == True]

    @staticmethod
    def churn_rate(source_df_with_time_index, name_column, freq='2M'):
        if source_df_with_time_index.empty:
            return np.nan
        cust_sets_by_freq = source_df_with_time_index.groupby(pd.Grouper(freq=freq))[name_column].apply(
            lambda x: set(x) if not pd.isna(x).all() else np.nan)
        cust_sets_by_freq = cust_sets_by_freq.mask(pd.isna(cust_sets_by_freq), set())
        churn_rates = (cust_sets_by_freq.shift(1) - cust_sets_by_freq).apply(
            lambda x: len(x) if not pd.isna(x) else np.nan) / \
                      cust_sets_by_freq.shift(1).apply(lambda x: len(x) if not pd.isna(x) else np.nan)
        mean_churn_rate = churn_rates.mean()
        return np.round(mean_churn_rate, 2)

    @staticmethod
    def customer_dev_rate(source_df_with_time_index, name_column, freq='2M'):
        if source_df_with_time_index.empty:
            return np.nan
        cust_sets_by_freq = source_df_with_time_index.groupby(pd.Grouper(freq=freq))[name_column].apply(
            lambda x: set(x) if not pd.isna(x).all() else np.nan)
        cust_sets_by_freq = cust_sets_by_freq.mask(pd.isna(cust_sets_by_freq), set())
        dev_rates = (cust_sets_by_freq - cust_sets_by_freq.shift(1)).apply(
            lambda x: len(x) if not pd.isna(x) else np.nan) \
                    / cust_sets_by_freq.apply(lambda x: len(x) if not pd.isna(x) else np.nan)
        mean_dev_rate = dev_rates.mean()
        return np.round(mean_dev_rate, 2)

    @staticmethod
    def qualityBuyerTagging(top_100_deal_table, latest_month=None, filters=settings.QUALITY_BUYERS_CONFIG):

        FEATURES = ['是否满足入池标准', '所满足标准',
                    '近12月购买月数', '远12月购买月数', '近9月购买月数', '环比远9月购买月数',
                    '近6月购买月数', '环比远6月购买数', '同比远6月购买数', '同比远6月含购买数',
                    '近3月购买月数', '环比远3月购买数', '近2月购买月数',
                    '近6个双月购买数', '远6个双月购买数',
                    '近3个双月购买数', '同比远3个双月购买数', '环比远3个双月购买数',
                    '近2个双月购买数', '同比远2个双月购买数', '环比远2个双月购买数',
                    '近4个三月购买数', '远4个三月购买数',
                    '近2个三月购买数', '同比远2个三月购买数', '环比远2个三月购买数',
                    '近24月开始合作月', '近24月最大可能合作次数', '近12月最大可能合作次数', '近6月最大可能合作次数',
                    '近24月最长连续零开票数', '近12月最长连续零开票数',
                    '近6月最长连续零开票数', '近3月最长连续零开票数',
                    '近12月开票额', '远12月开票额', '近8个三月累计开票额平均增长率',
                    '近4个三月累计开票额平均增长率', '近4个双月累计开票额平均增长率',
                    ]
        try:
            assert not top_100_deal_table.empty
        except AssertionError:
            print('WARN: 全量买方数据为空！')
            return pd.DataFrame(columns=FEATURES, index=['000000'])
        all_buyers = top_100_deal_table.columns.values
        top_100_deal_table.sort_index(ascending=True, inplace=True)
        if pd.isna(latest_month):
            latest_month = top_100_deal_table.index[-1]
        quality_taggers = pd.DataFrame(columns=FEATURES, index=all_buyers)

        def get_first_invoice_month_latest_24(column):
            try:
                return column.index[column.to_numpy().nonzero()[0][0]]
            except IndexError:
                return column.index[-1]

        quality_taggers['近24月开始合作月'] = (top_100_deal_table.apply(get_first_invoice_month_latest_24)).values
        quality_taggers['近24月开始合作月'] = quality_taggers['近24月开始合作月'].apply(
            lambda time: time.tz_localize('Asia/Shanghai'))

        def feasibleTime(dt):
            if dt >= latest_month - pd.Timedelta(365 * 2, 'D'):
                return dt
            else:
                return latest_month - pd.Timedelta(365 * 2, 'D')

        quality_taggers['近24月开始合作月'] = quality_taggers['近24月开始合作月'].apply(lambda dt: feasibleTime(dt))

        quality_taggers['近24月最大可能合作次数'] = np.minimum(
            np.ceil((latest_month - quality_taggers['近24月开始合作月']) / np.timedelta64(1, 'M')), 24)
        quality_taggers['近12月最大可能合作次数'] = np.where(quality_taggers['近24月最大可能合作次数'] >= 12, 12,
                                                   quality_taggers['近24月最大可能合作次数'])
        quality_taggers['近6月最大可能合作次数'] = np.where(quality_taggers['近24月最大可能合作次数'] >= 6, 6,
                                                  quality_taggers['近24月最大可能合作次数'])

        # 计算单月频率下的特征字段
        quality_taggers['近12月购买月数'] = (
            top_100_deal_table.tail(12).apply(lambda column: 12 - DataQualityManager.countZeros(column.values))).values
        quality_taggers['远12月购买月数'] = (top_100_deal_table.tail(24).head(12).apply(
            lambda column: 12 - DataQualityManager.countZeros(column.values))).values
        quality_taggers['近9月购买月数'] = (
            top_100_deal_table.tail(9).apply(lambda column: 9 - DataQualityManager.countZeros(column.values))).values
        quality_taggers['环比远9月购买月数'] = (
            top_100_deal_table.tail(18).head(9).apply(
                lambda column: 9 - DataQualityManager.countZeros(column.values))).values
        quality_taggers['近6月购买月数'] = (
            top_100_deal_table.tail(6).apply(lambda column: 6 - DataQualityManager.countZeros(column.values))).values
        quality_taggers['环比远6月购买数'] = (top_100_deal_table.tail(12).head(6).apply(
            lambda column: 6 - DataQualityManager.countZeros(column.values))).values
        quality_taggers['同比远6月购买数'] = (top_100_deal_table.tail(18).head(6).apply(
            lambda column: 6 - DataQualityManager.countZeros(column.values))).values
        quality_taggers['同比远6月含购买数'] = (top_100_deal_table.tail(18).head(7).apply(
            lambda column: 7 - DataQualityManager.countZeros(column.values))).values
        quality_taggers['近3月购买月数'] = (
            top_100_deal_table.tail(3).apply(lambda column: 3 - DataQualityManager.countZeros(column.values))).values
        quality_taggers['环比远3月购买数'] = (top_100_deal_table.tail(6).head(3).apply(
            lambda column: 3 - DataQualityManager.countZeros(column.values))).values
        quality_taggers['近2月购买月数'] = (
            top_100_deal_table.tail(2).apply(lambda column: 2 - DataQualityManager.countZeros(column.values))).values
        quality_taggers['近24月最长连续零开票数'] = (top_100_deal_table.apply(
            lambda column: DataQualityManager.longestContinuousZerosLen(
                column[quality_taggers.loc[column.name]['近24月开始合作月']:].values))).values
        quality_taggers['近12月最长连续零开票数'] = (top_100_deal_table.apply(
            lambda column: DataQualityManager.longestContinuousZerosLen(
                column[np.maximum(np.minimum(quality_taggers.loc[column.name]['近24月开始合作月'], column.index[-12]),
                                  column.index[-12]):].values))).values
        quality_taggers['近6月最长连续零开票数'] = (top_100_deal_table.apply(
            lambda column: DataQualityManager.longestContinuousZerosLen(
                column[np.maximum(np.minimum(quality_taggers.loc[column.name]['近24月开始合作月'], column.index[-6]),
                                  column.index[-6]):].values))).values
        quality_taggers['近3月最长连续零开票数'] = (top_100_deal_table.apply(
            lambda column: DataQualityManager.longestContinuousZerosLen(
                column[np.maximum(np.minimum(quality_taggers.loc[column.name]['近24月开始合作月'], column.index[-3]),
                                  column.index[-3]):].values))).values
        quality_taggers['近12月开票额'] = (top_100_deal_table.tail(12).sum()).values
        quality_taggers['远12月开票额'] = (top_100_deal_table.tail(24).head(12).sum()).values

        top_100_deal_table_2M = top_100_deal_table.resample('2M', closed='left', loffset='-1M').sum()
        quality_taggers['近6个双月购买数'] = (
            top_100_deal_table_2M.tail(6).apply(lambda column: 6 - DataQualityManager.countZeros(column.values))).values
        quality_taggers['远6个双月购买数'] = (top_100_deal_table_2M.tail(12).head(6).apply(
            lambda column: 6 - DataQualityManager.countZeros(column.values))).values
        quality_taggers['近3个双月购买数'] = (
            top_100_deal_table_2M.tail(3).apply(lambda column: 3 - DataQualityManager.countZeros(column.values))).values
        quality_taggers['同比远3个双月购买数'] = (top_100_deal_table_2M.tail(9).head(3).apply(
            lambda column: 3 - DataQualityManager.countZeros(column.values))).values
        quality_taggers['环比远3个双月购买数'] = (top_100_deal_table_2M.tail(6).head(3).apply(
            lambda column: 3 - DataQualityManager.countZeros(column.values))).values
        quality_taggers['近2个双月购买数'] = (
            top_100_deal_table_2M.tail(2).apply(lambda column: 2 - DataQualityManager.countZeros(column.values))).values
        quality_taggers['同比远2个双月购买数'] = (top_100_deal_table_2M.tail(8).head(2).apply(
            lambda column: 2 - DataQualityManager.countZeros(column.values))).values
        quality_taggers['环比远2个双月购买数'] = (top_100_deal_table_2M.tail(4).head(2).apply(
            lambda column: 2 - DataQualityManager.countZeros(column.values))).values

        top_100_deal_table_3M = top_100_deal_table.resample('3M', closed='left', loffset='-1M').sum()
        quality_taggers['近4个三月购买数'] = (
            top_100_deal_table_3M.tail(4).apply(lambda column: 4 - DataQualityManager.countZeros(column.values))).values
        quality_taggers['远4个三月购买数'] = (top_100_deal_table_3M.tail(8).head(4).apply(
            lambda column: 4 - DataQualityManager.countZeros(column.values))).values
        quality_taggers['近2个三月购买数'] = (
            top_100_deal_table_3M.tail(2).apply(lambda column: 2 - DataQualityManager.countZeros(column.values))).values
        quality_taggers['同比远2个三月购买数'] = (top_100_deal_table_3M.tail(6).head(2).apply(
            lambda column: 2 - DataQualityManager.countZeros(column.values))).values
        quality_taggers['环比远2个三月购买数'] = (top_100_deal_table_3M.tail(4).head(2).apply(
            lambda column: 2 - DataQualityManager.countZeros(column.values))).values

        def arithmatic_mean_since_first_buy(buyer_column, tail=4):
            try:
                valid_buyer_column = buyer_column[buyer_column.index[buyer_column.to_numpy().nonzero()[0][0]]:]
            except Exception:
                valid_buyer_column = buyer_column
            cumulative_valid_buyer_column = valid_buyer_column.cumsum()
            growth_percents = cumulative_valid_buyer_column.pct_change()
            return np.where(pd.isna(growth_percents.tail(tail).mean()), 0, growth_percents.tail(tail).mean())

        quality_taggers['近8个三月累计开票额平均增长率'] = (
            top_100_deal_table_3M.apply(lambda column: arithmatic_mean_since_first_buy(column, tail=8))).values
        quality_taggers['近4个三月累计开票额平均增长率'] = (
            top_100_deal_table_3M.apply(lambda column: arithmatic_mean_since_first_buy(column, tail=4))).values
        quality_taggers['近4个双月累计开票额平均增长率'] = (
            top_100_deal_table_2M.apply(lambda column: arithmatic_mean_since_first_buy(column, tail=4))).values

        def augment_sythetic_stats(quality_taggers):
            def percent(left, right):
                return np.clip((left - right) / (right + 0.1), -100, 100)

            quality_taggers['近/远12月购买月数差'] = np.where(quality_taggers['近24月最大可能合作次数'] >= 24,
                                                      quality_taggers['近12月购买月数'] - quality_taggers['远12月购买月数'], np.nan)
            quality_taggers['近/远12月购买月数差_绝对值'] = np.abs(quality_taggers['近/远12月购买月数差'])
            quality_taggers['近12月开票额增长率'] = percent(quality_taggers['近12月开票额'], quality_taggers['远12月开票额'])
            quality_taggers['近/环比远6月购买月数差'] = np.where(quality_taggers['近24月最大可能合作次数'] >= 12,
                                                       quality_taggers['近6月购买月数'] - quality_taggers['环比远6月购买数'], np.nan)
            quality_taggers['近/环比远6月购买月数差_绝对值'] = np.abs(quality_taggers['近/环比远6月购买月数差'])
            quality_taggers['近/同比远6月购买月数差'] = np.where(quality_taggers['近24月最大可能合作次数'] >= 18,
                                                       quality_taggers['近6月购买月数'] - quality_taggers['同比远6月购买数'], np.nan)
            quality_taggers['近/同比远6月购买月数差_绝对值'] = np.abs(quality_taggers['近/同比远6月购买月数差'])
            quality_taggers['近/远6个双月购买月数差'] = np.where(quality_taggers['近24月最大可能合作次数'] >= 23,
                                                       quality_taggers['近6个双月购买数'] - quality_taggers['远6个双月购买数'],
                                                       np.nan)
            quality_taggers['近/远6个双月购买月数差_绝对值'] = np.abs(quality_taggers['近/远6个双月购买月数差'])
            quality_taggers['近/环比远9月购买月数差'] = np.where(quality_taggers['近24月最大可能合作次数'] >= 18,
                                                       quality_taggers['近9月购买月数'] - quality_taggers['环比远9月购买月数'],
                                                       np.nan)
            quality_taggers['近/环比远9月购买月数差_绝对值'] = np.abs(quality_taggers['近/环比远9月购买月数差'])
            quality_taggers['近/远4个三月购买月数差'] = np.where(quality_taggers['近24月最大可能合作次数'] >= 22,
                                                       quality_taggers['近4个三月购买数'] - quality_taggers['远4个三月购买数'],
                                                       np.nan)
            quality_taggers['近/远4个三月购买月数差_绝对值'] = np.abs(quality_taggers['近/远4个三月购买月数差'])
            return quality_taggers

        quality_taggers = augment_sythetic_stats(quality_taggers)

        def judge(filter, buyer_features):
            status = True
            for or_condition in filter:
                interval = DataQualityManager.parseInterval(filter[or_condition])
                if not DataQualityManager.isBetween(interval, buyer_features[or_condition]):
                    status = False
            return status

        def select(buyer_features_row, filters):
            status = False
            for filter_ in filters:
                is_qualified_buyer = judge(filters[filter_], buyer_features_row)
                if is_qualified_buyer:
                    status = True
            return status

        def satisfiedConditions(buyer_features_row, filters):
            satisfied = ''
            for filter_ in filters:
                is_qualified_buyer = judge(filters[filter_], buyer_features_row)
                if is_qualified_buyer:
                    satisfied += filter_ + ' '
            return satisfied

        quality_taggers['是否满足入池标准'] = quality_taggers.apply(lambda row: select(row, filters=filters), axis=1)
        quality_taggers['所满足标准'] = quality_taggers.apply(lambda row: satisfiedConditions(row, filters=filters), axis=1)
        return quality_taggers

    @staticmethod
    def characterizeQualityBuyers(source_df_with_cancel, quality_buyers, observeTime):
        QualityBuyers_features = pd.DataFrame(columns=['销方名称', '稳定买方总开票月数', '稳定买方数量', '近12月稳定买方离差率',
                                                       '近12月稳定买方开票额', '远12月稳定买方开票额', '近12月稳定买方开票增长率',
                                                       '近6月稳定买方开票额', '环比远6月稳定买方开票额', '同比远6月稳定买方开票额',
                                                       '近6月同比稳定买方开票增长率', '近6月环比稳定买方开票增长率',
                                                       '近24月稳定买方最长连续零开票数', '近12月稳定买方最长连续零开票数',
                                                       '近12月稳定买方作废率', '近6月稳定买方作废率', '近3月稳定买方作废率',
                                                       '近12月稳定买方红冲率', '近6月稳定买方红冲率', '近3月稳定买方红冲率'])
        try:
            assert not source_df_with_cancel.empty
        except AssertionError:
            print('可供计算的稳定客户筛选前数据集为空！')
            return QualityBuyers_features
        qualified_invoice_df_with_cancel = source_df_with_cancel[source_df_with_cancel['统一名称'].isin(quality_buyers)]
        df1_valid_source = qualified_invoice_df_with_cancel[qualified_invoice_df_with_cancel["状态"] == "有效"]
        df1_valid_source['合计金额'] = df1_valid_source['合计金额'] + df1_valid_source['合计税额'].replace(to_replace=np.nan,
                                                                                               value=0)
        df1_cancel_source = qualified_invoice_df_with_cancel[qualified_invoice_df_with_cancel["状态"] == "作废"]
        df1_redCorrection_source = qualified_invoice_df_with_cancel[
            np.logical_and(qualified_invoice_df_with_cancel["合计金额"] < 0,
                           qualified_invoice_df_with_cancel["状态"] == "有效")]
        df1_valid = DataQualityManager.yearMonthlize(df1_valid_source, dateColumn="开票日期")

        first_invoice_date = qualified_invoice_df_with_cancel['开票日期'].min()
        if pd.isna(first_invoice_date):
            first_invoice_date = observeTime

        quality_buyers_count = len(quality_buyers)
        supplier_name = source_df_with_cancel['销方名称'].mode().iloc[0]
        total_invoice_months = np.ceil((observeTime - first_invoice_date) / np.timedelta64(1, 'M'))

        valid_latest_12 = DataQualityManager.selectByTimeRange(df=df1_valid, time_column='年月',
                                                               observeTime=observeTime,
                                                               to_observe_time_period=pd.Timedelta(365, unit='D'))
        valid_latest_12_total = valid_latest_12['合计金额'].sum()
        valid_last_12 = DataQualityManager.selectByTimeRange(df=df1_valid, time_column='年月',
                                                             observeTime=observeTime - pd.Timedelta(365, unit='D'),
                                                             to_observe_time_period=pd.Timedelta(365, unit='D'))
        valid_last_12_total = valid_last_12['合计金额'].sum()
        YoY_growth_12 = DataQualityManager.ceilTheOutlier(
            np.round((valid_latest_12_total - valid_last_12_total) / (valid_last_12_total + 0.0001), 2),
            lower_bound=-100, upper_bound=100)

        valid_latest_6 = DataQualityManager.selectByTimeRange(df=df1_valid, time_column='年月',
                                                              observeTime=observeTime,
                                                              to_observe_time_period=pd.Timedelta(30 * 6, unit='D'))
        valid_latest_6_total = valid_latest_6['合计金额'].sum()
        valid_further_6 = DataQualityManager.selectByTimeRange(df=df1_valid, time_column='年月',
                                                               observeTime=observeTime - pd.Timedelta(30 * 6, unit='D'),
                                                               to_observe_time_period=pd.Timedelta(30 * 6, unit='D'))
        valid_further_6_total = valid_further_6['合计金额'].sum()

        valid_last_6 = DataQualityManager.selectByTimeRange(df=df1_valid, time_column='年月',
                                                            observeTime=observeTime - pd.Timedelta(365, unit='D'),
                                                            to_observe_time_period=pd.Timedelta(30 * 6, unit='D'))
        valid_last_6_total = valid_last_6['合计金额'].sum()
        YoY_growth_6 = DataQualityManager.ceilTheOutlier(
            np.round((valid_latest_6_total - valid_last_6_total) / (valid_last_6_total + 0.0001), 2), lower_bound=-100,
            upper_bound=100)
        MoM_growth_6 = DataQualityManager.ceilTheOutlier(
            np.round((valid_latest_6_total - valid_further_6_total) / (valid_further_6_total + 0.0001), 2),
            lower_bound=-100, upper_bound=100)

        start_ = np.maximum(observeTime - pd.Timedelta(365 * 2, 'D'), first_invoice_date)
        df1_valid_24_sum_by_yearMonth = DataQualityManager.groupByMultipleColumns(df1_valid,
                                                                                  sorted_columns_list=['年月'],
                                                                                  as_index=False,
                                                                                  aggregate_method='sum',
                                                                                  start=start_,
                                                                                  end=observeTime - pd.Timedelta(28,
                                                                                                                 'D'))
        longest_continuous_zero_invoice_months_24 = DataQualityManager.longestContinuousZerosLen(
            df1_valid_24_sum_by_yearMonth['合计金额'].values)

        start_ = np.maximum(observeTime - pd.Timedelta(365, 'D'), first_invoice_date)
        df1_valid_12_sum_by_yearMonth = DataQualityManager.groupByMultipleColumns(valid_latest_12,
                                                                                  sorted_columns_list=['年月'],
                                                                                  as_index=False,
                                                                                  aggregate_method='sum',
                                                                                  start=start_,
                                                                                  end=observeTime - pd.Timedelta(28,
                                                                                                                 'D'))
        longest_continuous_zero_invoice_months_12 = DataQualityManager.longestContinuousZerosLen(
            df1_valid_12_sum_by_yearMonth['合计金额'].values)

        valid_latest_12_sum_by_yearMonth = DataQualityManager.groupByMultipleColumns(df=valid_latest_12,
                                                                                     sorted_columns_list=['年月'],
                                                                                     as_index=False,
                                                                                     aggregate_method='sum',
                                                                                     end=observeTime - pd.Timedelta(28,
                                                                                                                    'D'))
        mean_ = valid_latest_12_sum_by_yearMonth['合计金额'].mean()
        if pd.isna(mean_): mean_ = 0
        mean_deviations = np.abs(valid_latest_12_sum_by_yearMonth['合计金额'] - valid_latest_12_sum_by_yearMonth['合计金额'].mean()).mean()
        mean_deviation_ratio = np.round(mean_deviations / (mean_ + 0.0001), 2)

        latest_N_cancel_rate = {}
        latest_N_redCorrection_rate = {}
        month_freq_N = [12, 6, 3]

        for N in month_freq_N:

            df1_valid_absolute_sum = np.abs(
                DataQualityManager.selectByTimeRange(df1_valid_source, observeTime=observeTime,
                                                     time_column='年月',
                                                     to_observe_time_period=pd.Timedelta(N * 30, unit='D')
                                                     )["合计金额"]).sum()

            df1_cancel_absolute_sum = np.abs(
                DataQualityManager.selectByTimeRange(df1_cancel_source, observeTime=observeTime,
                                                     time_column='年月',
                                                     to_observe_time_period=pd.Timedelta(N * 30, unit='D')
                                                     )["合计金额"]).sum()

            df1_redCorrection_sum = np.abs(
                DataQualityManager.selectByTimeRange(df1_redCorrection_source, observeTime=observeTime,
                                                     time_column='年月',
                                                     to_observe_time_period=pd.Timedelta(N * 30, unit='D')
                                                     )["合计金额"]).sum()

            # 记录近N个月红冲率， 作废率
            latest_N_cancel_rate[N] = np.round(
                df1_cancel_absolute_sum / (df1_valid_absolute_sum + df1_cancel_absolute_sum + 0.0001), 4)
            if total_invoice_months < N:
                latest_N_cancel_rate[N] = np.nan
            latest_N_redCorrection_rate[N] = np.round(df1_redCorrection_sum / (df1_valid_absolute_sum + 0.0001), 4)
            if total_invoice_months < N:
                latest_N_redCorrection_rate[N] = np.nan

        QualityBuyers_features = pd.DataFrame(columns=['销方名称', '稳定买方总开票月数', '稳定买方数量', '近12月稳定买方离差率',
                                                       '近12月稳定买方开票额', '远12月稳定买方开票额', '近12月稳定买方开票增长率',
                                                       '近6月稳定买方开票额', '环比远6月稳定买方开票额', '同比远6月稳定买方开票额',
                                                       '近6月同比稳定买方开票增长率', '近6月环比稳定买方开票增长率',
                                                       '近24月稳定买方最长连续零开票数', '近12月稳定买方最长连续零开票数',
                                                       '近12月稳定买方作废率', '近6月稳定买方作废率', '近3月稳定买方作废率',
                                                       '近12月稳定买方红冲率', '近6月稳定买方红冲率', '近3月稳定买方红冲率'],
                                              data=[[supplier_name, total_invoice_months, quality_buyers_count,
                                                     mean_deviation_ratio,
                                                     valid_latest_12_total, valid_last_12_total, YoY_growth_12,
                                                     valid_latest_6_total, valid_further_6_total, valid_last_6_total,
                                                     YoY_growth_6, MoM_growth_6,
                                                     longest_continuous_zero_invoice_months_24,
                                                     longest_continuous_zero_invoice_months_12,
                                                     latest_N_cancel_rate[12], latest_N_cancel_rate[6],
                                                     latest_N_cancel_rate[3],
                                                     latest_N_redCorrection_rate[12], latest_N_redCorrection_rate[6],
                                                     latest_N_redCorrection_rate[3]]])
        return QualityBuyers_features

    @staticmethod
    def calculateLocalSalesPercent(df1_valid, observeTime, N_months_window, seller_province):
        valid_latest_N = DataQualityManager.selectByTimeRange(df=df1_valid, time_column='开票日期',
                                                              observeTime=observeTime,
                                                              to_observe_time_period=pd.DateOffset(
                                                                  months=N_months_window))

        recent_N_buyer_province_list = valid_latest_N['买方所在省份'].unique().tolist()

        recent_N_business_local_percent = np.nan

        valid_latest_N_total = valid_latest_N['合计金额'].sum()

        if pd.notna(seller_province) and seller_province != '':
            valid_latest_N_local = valid_latest_N[valid_latest_N['买方所在省份'] == seller_province]

            recent_N_local_business_sales = valid_latest_N_local['合计金额'].sum()

            if valid_latest_N_total > 0:
                recent_N_business_local_percent = recent_N_local_business_sales / valid_latest_N_total

        recent_N_buyer_province_list = [p for p in recent_N_buyer_province_list if pd.notna(p)]
        recent_N_business_province = len(recent_N_buyer_province_list)

        logging.debug(recent_N_buyer_province_list)

        observeTime1 = observeTime - pd.DateOffset(months=N_months_window)
        valid_last_N = DataQualityManager.selectByTimeRange(df=df1_valid, time_column='开票日期', \
                                                            observeTime=observeTime1, \
                                                            to_observe_time_period=pd.DateOffset(
                                                                months=N_months_window))

        last_N_buyer_province_list = valid_last_N['买方所在省份'].unique().tolist()
        last_N_buyer_province_list = [p for p in last_N_buyer_province_list if pd.notna(p)]
        last_N_business_province = len(last_N_buyer_province_list)

        logging.debug(last_N_buyer_province_list)

        buyer_province_N_change_rate = np.nan

        if last_N_business_province > 0:
            buyer_province_N_change_rate = (
                                                   recent_N_business_province - last_N_business_province) / last_N_business_province

        return recent_N_business_local_percent, recent_N_business_province, buyer_province_N_change_rate

    @staticmethod
    def calculateDeviationRate(df1_valid, observeTime, N_months_window):
        valid_latest_N = DataQualityManager.selectByTimeRange(df=df1_valid, time_column='开票日期',
                                                              observeTime=observeTime,
                                                              to_observe_time_period=pd.DateOffset(
                                                                  months=N_months_window))

        valid_latest_N_sum_by_yearMonth1 = DataQualityManager.groupByMultipleColumns(df=valid_latest_N,
                                                                                     sorted_columns_list=['年月'],
                                                                                     as_index=False,
                                                                                     aggregate_method='sum',
                                                                                     end=observeTime - pd.Timedelta(
                                                                                         28, 'D'),
                                                                                     start=observeTime - pd.DateOffset(
                                                                                         months=N_months_window))

        valid_latest_N_sum_by_yearMonth = DataQualityManager.completeMonths(valid_latest_N_sum_by_yearMonth1,
                                                                            start=observeTime - pd.DateOffset(
                                                                                months=N_months_window),
                                                                            end=observeTime - pd.Timedelta(28, 'D'))

        mean_ = valid_latest_N_sum_by_yearMonth['合计金额'].mean()
        if pd.isna(mean_):
            mean_ = 0
        mean_deviations = np.abs(
            valid_latest_N_sum_by_yearMonth['合计金额'] - mean_).mean()
        mean_deviation_ratio = np.round(mean_deviations / (mean_ + 1.0), 4)

        observeTime1 = observeTime - pd.DateOffset(months=N_months_window)
        observeYear1 = observeTime1.year
        observeMonth1 = observeTime1.month
        observeDay1 = observeTime1.day
        observeTime2 = pd.Timestamp(observeYear1, observeMonth1, np.minimum(observeDay1, 1)).tz_localize(
            'Asia/Shanghai')
        observeTime3 = observeTime2 - pd.Timedelta(15, 'D')

        valid_last_N = DataQualityManager.selectByTimeRange(df=df1_valid, time_column='开票日期',
                                                            observeTime=observeTime2, \
                                                            to_observe_time_period=pd.DateOffset(
                                                                months=N_months_window))

        valid_last_N_sum_by_yearMonth = DataQualityManager.groupByMultipleColumns(df=valid_last_N,
                                                                                  sorted_columns_list=['年月'],
                                                                                  as_index=False,
                                                                                  aggregate_method='sum',
                                                                                  end=observeTime3,
                                                                                  start=observeTime2 - pd.DateOffset(
                                                                                      months=N_months_window))
        mean_last = valid_last_N_sum_by_yearMonth['合计金额'].mean()
        if pd.isna(mean_last):
            mean_last = 0
        mean_deviations_last = np.abs(
            valid_last_N_sum_by_yearMonth['合计金额'] - mean_last).mean()

        mean_deviation_ratio_last = np.round(mean_deviations_last / (mean_last + 1.0), 4)

        logging.info("{}:{}".format(logcode(75, N_months_window), mean_deviation_ratio))
        logging.info("{}:{}".format(logcode(76, N_months_window), mean_deviation_ratio_last))

        mean_deviation_change_rate = np.round(abs(mean_deviation_ratio - mean_deviation_ratio_last), 4)
        logging.info("{}:{}".format(logcode(77, N_months_window), mean_deviation_change_rate))

        return mean_deviation_ratio, mean_deviation_ratio_last, mean_deviation_change_rate

    @staticmethod
    def is_foreign_denominated(money_number):
        money_number = pd.to_numeric(money_number, errors='coerce')
        if pd.isna(money_number): return False
        return not float(money_number).is_integer()
