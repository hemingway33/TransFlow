import os
import time
from collections import OrderedDict
import pandas as pd
from utils.dataSource import DBconnector, LshDB
import sys
from numpy import nan
from hashlib import sha512
from utils.utils import remove_location_unit
import re
import ahocorasick
from bloom_filter import BloomFilter
from utils.Logger import Logger, logcode

logging = Logger(level="info", name=__name__).getlog()


def load_organizations(organization_file):
    logging.info(logcode(4))
    organizations = set()

    if not os.path.exists(organization_file):
        raise IOError("NotFoundFile")

    with open(organization_file, 'r', encoding='UTF-8') as f_obj:
        for f_line in f_obj.readlines():
            try:
                line = f_line.split(',')
                organizations.add(line[0])
            except Exception:
                raise Exception("无法解析此行车牌区域号信息：{}".format(f_line))
    return organizations


def load_administrative_divisions(province_file, city_file, area_file):
    logging.info(logcode(3))

    bloom_ = BloomFilter(max_elements=4000, error_rate=0.001)
    divisions_dict = dict()

    if not os.path.exists(province_file) or not os.path.exists(city_file) or not os.path.exists(area_file):
        raise IOError("NotFoundFile")

    with open(province_file, 'r', encoding='UTF-8') as f_obj:
        for f_line in f_obj.readlines():
            try:
                line = f_line.split(',')
                ele = remove_location_unit(line[1])
                if len(ele) > 0:
                    bloom_.add(ele)
                    divisions_dict[ele] = str(line[0])
            except Exception:
                raise Exception("无法解析此行省主题词信息：{}".format(f_line))

    with open(city_file, 'r', encoding='UTF-8') as f_obj:
        for f_line in f_obj.readlines():
            try:
                line = f_line.split(',')
                ele = remove_location_unit(line[1])
                if len(ele) > 0:
                    bloom_.add(ele)
                    divisions_dict[ele] = str(line[0])
            except Exception:
                raise Exception("无法解析此行市主题词信息：{}".format(f_line))

    with open(area_file, 'r', encoding='UTF-8') as f_obj:
        for f_line in f_obj.readlines():
            try:
                line = f_line.split(',')
                ele = remove_location_unit(line[1])
                if len(ele) > 0:
                    bloom_.add(ele)
                    divisions_dict[ele] = str(line[0])
            except Exception:
                raise Exception("无法解析此行县主题词信息：{}".format(f_line))

    return bloom_, divisions_dict


def load_hierarchical_locations(province_file, city_file):
    logging.info(logcode(96))

    nested_location_dict = OrderedDict()
    city_dict = OrderedDict()

    if not os.path.exists(province_file) or not os.path.exists(city_file):
        raise IOError("NotFoundFile")

    with open(city_file, 'r', encoding='UTF-8') as f_obj:
        for f_line in f_obj.readlines():
            try:
                line = f_line.split(',')
                ele = line[1]
                if len(ele) > 0:
                    city_dict[str(line[0].strip())] = ele.strip()
            except Exception:
                raise Exception("无法解析此行市主题词信息：{}".format(f_line))

    with open(province_file, 'r', encoding='UTF-8') as f_obj:
        for f_line in f_obj.readlines():
            try:
                line = f_line.split(',')
                ele = line[1]
                if len(ele) > 0:
                    code = str(line[0]).strip()

                    province_record = {'code': code, 'sub': {city_dict[k]: k for k in sorted(city_dict.keys())
                                                             if k.startswith(code)}}
                    nested_location_dict[ele.strip()] = province_record
            except Exception:
                raise Exception("无法解析此行省主题词信息：{}".format(f_line))

    return nested_location_dict


def load_core_buyers(DBType='sqlite3'):
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
        '开票日期': 'time',
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
        '核企行业': 'string',
        '关联企业名称': 'string',
        '销方开票地址': 'string',
        '核心企业行业': 'string',
        '核企行业更新时间': 'time',
        '税收编码': 'string',
        '发票明细类型': 'numeric'
    }
    global core_buyers_df
    # 读取总体核心企业白名单客户: 来源H2O,LIMIT, ACCESS, 当前只约束到H2O
    if DBType == 'sqlite3':
        SQLITE_PATH = CONFIG_PATH + "/config_datas.db"
        CONFIG_CONN = DBconnector(DBType='sqlite3', host=SQLITE_PATH)
        conn = CONFIG_CONN

        logging.debug("当前白名单查询使用连接:{}".format(conn))
        query_core_brands = """SELECT distinct white_list_name as '核企识别名',
                                'NA' as '核企类型',
                                level AS '核企级别',
                                industry AS '核企行业'
                                from tm_white_list
                           """

    logging.debug("当前白名单使用查询语句:\n{}".format(query_core_brands))
    try:
        core_buyers_df = conn.query(query=query_core_brands).as_pandas().typeConverter(converters).getQueryResult()
        core_buyers_df["匿名化编号"] = core_buyers_df['核企识别名'].apply(lambda x: sha512(x.encode("utf-8")).hexdigest().upper()[:10])
        core_buyers_df.set_index('核企识别名', inplace=True)
        logging.debug("当前查询得到白名单数量:{}".format(len(core_buyers_df)))
        # 核企业级别若有重复，孰高者得
        core_buyers_df = core_buyers_df.sort_values(by=['核企级别'], ascending=False, na_position='last')
        core_buyers_df = core_buyers_df[~core_buyers_df.index.duplicated(keep='first')]
        logging.debug('当前核心企业总数：{}'.format(len(core_buyers_df)))
    except Exception as e:
        logging.error("ERROR:{}".format(e))
        core_buyers_df = pd.DataFrame(columns=['核企识别名', '核企类型', '核企级别', '核企行业'])
        logging.error('核心企业列表为空！！！：{}'.format(core_buyers_df))
    return core_buyers_df


def init_configuration():
    global stop_words_dict, stop_words_machine
    stop_words_dict = OrderedDict({
        '省': '',
        '市': '',
        '县': '',
        "新区": '',
        '区': '',
        '自治': '',
        "股份": '',
        "公司": "司",
        '有限': "",
        '责任': "",
        '控股': "箜",
        '分店': "店",
        "百货": "百",
        '连锁': '锁',
        '商店': '店',
        '分部': '部',
        "大药房": "要",
        '药房': '要',
        '药店': '要',
        '医药': '药',
        '超市': '曹',
        '餐饮': '餐',
        '电子商务': '电商',
        '信息技术': '信技',
        '投资发展': '投发',
        '分行': '行',
        '医院': '院',
        '附属': "附",
        '第': '',
        '贸易': '易',
        "物业管理": "物",
        "监督": "监",
        '管理': '管',
        '供应链': '应',
        '零部件': '零件',
        '汽车': '汽',
        '服务': '',
        '劳防': '劳',
        '用品': '用',
        "市政": '',
        "工程建设监理": "监理",
        '工程': '工',
        '工贸': '贸',
        "商贸": '贸',
        "商业": '',
        '新材料': '料',
        '装饰': '饰',
        '国际': '际',
        '物流': '流',
        '货物运输代理': '货代',
        '代理': '代',
        '货运': '运',
        '保健': '',
        '经营': '经',
        '交易': '易',
        "集团": '湍',
        "电子": '电',
        '高分子': '子',
        '农业': '农',
        "科技": '科',
        "制造": '造',
        "机械": '械',
        '工厂': '厂',
        "房地产经纪": '纪',
        "房地产": '房',
        "开发": '',
        "置业": "置",
        "医疗": "疗",
        "器械": "器",
        "技术": "",
        "建筑": "筑",
        "设计": "",
        "人寿保险": "人险",
        "财产保险": "财险",
        "保险": "保",
        "芯片": "片",
        "生物": "物",
        "半导体": "导",
        "设备": "备",
        "咨询": "咨",
        "网络": "网",
        "信用征信": "征",
        "发展和改革": "发改",
        "人民政府": "伏",
        "企业": "",
        "发展银行": "展垠",
        "发展": "",
        "实业": "实",
        "健康": "健",
        "工业和信息化": "工信",
        "信息": "",
        "塑胶": "塑",
        "模具": "模",
        "互联网": "互",
        "金融": "容",
        "农村": "村",
        "卫生站": "卫",
        "银行": "垠",
        "律师": "律",
        "事务所": "事",
        "知识产权": "权",
        "运营": "",
        "影视": "影",
        "电视广播": "播",
        "电视台": "视",
        "文化": "",
        "产业": "产",
        "制品": "",
        "融资租赁": "赁",
        "办公": "",
        "电源": "源",
        "精密": "",
        "光学眼镜": "镜",
        "光学": "",
        "五金": "五",
        "环保": "",
        "自动化": "自",
        "系统": "系",
        "机电": "",
        "电板板": "板",
        "道路": "",
        "酒店": "酒",
        "西医": "",
        "内科": "",
        "诊所": "诊",
        "工业": "工",
        "传播": "播",
        "传媒": "媒",
        "绝缘": "",
        "制冷": "",
        "特殊普通合伙": "",
        "合伙": "",
        "普通": "",
        "会计师": "会",
        "建设项目": "建",
        "建设": "建",
        "建安": "安",
        "建筑安装": "安",
        "项目": "",
        "销售": "",
        "旅行社": "旅",
        "造价": "",
        "产品": "",
        "医学诊断": "断",
        "地产": "地",
        "策划": "划",
        "营销": "",
        "艺术": "艺",
        "环境": "境",
        "地球物理": "球",
        "电气": "",
        "医疗美容": "美",
        "门诊部": "门",
        "智能": "",
        "消防": "消",
        "投资": "",
        "基金": "基",
        "电力": "电",
        "光电": "",
        "创意": "",
        "基础设施": "础",
        "商务": "",
        "监理": "监",
        "通讯设备": "讯",
        "通讯": '讯',
        "器材": "材",
        "人力资源": "人",
        "安防": "",
        "人工智能": "能",
        "教育": "教",
        "体育": "体",
        "修理": "修",
        "培训中心": "训",
        "培训": "训",
        "瑜伽": "",
        "留学": "",
        "公路": "路",
        "快速路": "速",
        "航空": "航",
        "股权": "",
        "税务师": "税",
        "配送": "送",
        "生鲜食品": "鲜",
        "生鲜": "",
        "食品": "食",
        "国家税务": "税",
        "国家财政": "财",
        "税务": "税",
        "财政": "财",
        "总局": "局",
        "社会": "社",
        "幼儿园": "圆",
        "硬质": "",
        "合金": "",
        "华东": "东",
        "华西": "西",
        "华北": "北",
        "华南": "南",
        "发电": "电",
        "钢化玻璃": "璃",
        "玻璃": "玻璃",
        "包装": "",
        "职介": "介",
        "婚介": "介",
        "电动车": "车",
        "针纺织品": "纺",
        "纺织品": "纺",
        "新型": "",
        "金属": "属",
        "焊接": "",
        "母婴护理": "护",
        "母婴": "婴",
        "护理": "护",
        "进出口": "口",
        "进口": "口",
        "出口": "口",
        "街道": "",
        "社区卫生": "卫",
        "强制隔离戒毒所": "戒",
        "戒毒所": "戒",
        "监狱": "狱",
        "安全防护": "护",
        "安全": "全",
        "防护": "护",
        "中医": "忠",
        "征收": "征",
        "房屋": "",
        "政策": "",
        "生态": "",
        "小学校": "晓",
        "小学": "晓",
        "中学校": "钟",
        "中学": "钟",
        "学校": "校",
        "大学": "汏",
        "药业": "药",
        "社区": "",
        "配件": "",
        "经纪": "纪",
        "假期": "",
        "办事处": "事",
        "广告": "告",
        "服装": "浮",
        "辅料": "辅",
        "结构": "",
        "线路": "",
        "印刷": "印",
        "印务": "印",
        "公关顾问": "关",
        "公关": "公",
        "顾问": "问",
        "药品": "药",
        "商场": "场",
        "村委会": "委",
        "工会委员会": "工会",
        "委员会": "员",
        "舞台": "台",
        "资源": "",
        "纺织": "纺",
        "商行": "行",
        "交电": "",
        "餐厅": "餐",
        "计划生育": "计",
        "生育": "育",
        "化妆品": "妆",
        "干部": "",
        "休养所": "养",
        "养老院": "老",
        "经济合作社": "作",
        "合作社": "",
        "经济": "",
        "综合": "",
        "中西医结合": "中西",
        "结合": "",
        "开发区": "开",
        "中心": "忻",
        "交通": "交",
        "制带": "带",
        "商标": "标",
        "无纺布": "布",
        "表面处理": "面",
        "居民": "",
        "照明": "照",
        "电器": "器",
        "快捷": "",
        "人民": "人",
        "新闻": "闻",
        "出版": "版",
        "机动车": "机",
        "驾驶": "驶",
        "职业": "职",
        "技能": "能",
        "经销部": "部",
        "物资": "资",
        "新能源": "薪",
        "气象": "象",
        "公安": "安",
        "民政": "政",
        "封装测试": "测",
        "封装": "封",
        "测试": "测",
        "石油": "石",
        "化工": "化",
        "加油站": "油",
        "速达": "素",
        "集成电路": "集路",
        "电路": "璐",
        "保温": "温",
        "饮食": "饮",
        "小额贷款": "小贷",
        "电线": "",
        "电缆": "缆",
        "实验": "",
        "零件": "",
        "装卸": "卸",
        "搬运": "搬",
        "警察": "警",
        "执法": "执",
        "服饰": "饰",
        "社会保障": "社保",
        "装备": "备",
        "设施": "",
        "运输": "输",
        "灯光": "灯",
        "村镇": "",
        "镇": "",
        "科学": "科",
        "仪器": "仪",
        "资产": "",
        "检验": "检",
        "针织": "织",
        "能源": "源",
        "成套": "",
        "信访局": "衅",
        "五谷道场": "谷场",
        "会议": "议",
        "展览": "览",
        "专利": "利",
        "外包": "",
        "耗材": "",
        "证券": "券",
        "财务": "财",
        "大宗": "宗",
        "商品": "品",
        "合成": "",
        "研究": "研",
        "饮料": "饮",
        "批发": "批",
        "便利": "便",
        "基因": "因",
        "阀门": "",
        "复合": "",
        "解放军": "军",
        "发行": "",
    })
    # pre-search for efficient replacing
    stop_words_machine = ahocorasick.Automaton()
    for word in stop_words_dict:
        stop_words_machine.add_word(word, (word, stop_words_dict[word]))
    stop_words_machine.make_automaton()
    global administrative_divisions_bloom, administrative_divisions_dict, administrative_divisions_patt
    # 中国行政划分
    administrative_divisions_bloom, administrative_divisions_dict = load_administrative_divisions(
        province_file=os.path.join(CONFIG_PATH, 'province'),
        city_file=os.path.join(CONFIG_PATH, 'city'),
        area_file=os.path.join(CONFIG_PATH, 'areas'))

    administrative_divisions_patt = "|".join(administrative_divisions_dict.keys())

    global organizations_types, organization_patt
    organizations_types = load_organizations(organization_file=os.path.join(CONFIG_PATH, 'organization'))
    organization_patt = "|".join(organizations_types)

    global shipping_service_key_words, port_charges_key_words, shipping_service_key_words_excludes, port_charges_key_words_excludes
    shipping_service_key_words, shipping_service_key_words_excludes = ["运输代理费", "运费", "海运代理费", "运输费",
                                                                       "订舱", "订仓", "包干"], ["港杂", "保险费"]
    port_charges_key_words, port_charges_key_words_excludes = ["港杂", "报关", "换单", "改单", "拖车", "THC",
                                                               "VGM", "提重", "安保", "提柜", "回空", "掏箱",
                                                               "文件", "封子", "操作", "查验", "仓储", "保险",
                                                               "运杂", "改配", "舱单", "场站", "还箱", "内装",
                                                               "打单", "电放", "单证", "设备单"], ["运输代理费", "运费", "运输费"]

    lsh_local_file = os.path.join(CONFIG_PATH, "lean_core_buyer_LSH.pickle")
    core_hash_local_file = os.path.join(CONFIG_PATH, "core_hash_local_file.pickle")
    global anchor_buyer_lsh, anonymize_lsh
    load_from_local = False
    anonymize_lsh = True
    lsh_num_perm = 50
    lsh_threshold = 0.90
    anchor_buyer_lsh = LshDB(num_perm=lsh_num_perm, threshold=lsh_threshold,
                             administrative_divisions_patt=administrative_divisions_patt,
                             organization_patt=organization_patt, stop_words_machine=stop_words_machine,
                             anonymize_lsh=anonymize_lsh)

    if load_from_local:
        logging.info('开始生成核心企业LSH库')
        global_core_buyers = load_core_buyers()
        global_core_buyers_dict = global_core_buyers.to_dict("index")
        anchor_buyer_lsh.batch_minhash(global_core_buyers_dict)
        start = time.time()
        try:
            anchor_buyer_lsh.set_local_lean_lsh_path(lsh_local_file).to_updatable_lean_lsh().to_local_lean_lsh()
            anchor_buyer_lsh.to_local_core_hash(core_hash_local_file)
            logging.info("生成核心企业LSH本地成功，用时{}".format(time.time() - start))
        finally:
            pass
    else:
        anchor_buyer_lsh.set_local_lean_lsh_path(lsh_local_file).load_local_lean_lsh()
        anchor_buyer_lsh.load_local_core_hash(core_hash_local_file)
        logging.info(logcode(5))


def init():
    global BASE_PATH, OUTPUT_PATH, CONFIG_PATH, DB_ENGINE
    global CITY_CODES, CITY_ADJUSTMENTS, WHITELIST_COMMODITY_DICT, WHITELIST_COMMODITY_DICT_LIVE, PROVINCE_CODES
    global buyers_profile

    global WHITE_LIST_BUYER_CONN, WHITE_BUYER_BIG_DATA_CONN, WHITE_LIST_BUYER_CONN_PYP, WHITE_LIST_BUYER_CONN_TMP  # 白名单买方库
    global INVOICE_CONN, QUALITY_BUYERS_CONFIG, PRODUCT_CODES, EXCEL_COLUMN_SORTING
    global PYP_ENTRY_BARS, JOINT_WHITE_BUYERS_CONFIG, DYNAMIC_WHITE_BUYERS_CONFIG

    global CONFIG_CONN, VARIABLE_CREDIT_SCORE_CONFIG, R_CONTAINER
    global REPORT_TEMPLATE_PATH,  HIERARCHICAL_LOCATIONS_DICT

    logging.info(logcode(1))
    
    if getattr(sys, 'frozen', False): BASE_PATH = sys._MEIPASS
    else: BASE_PATH = os.path.dirname(os.path.dirname(__file__))

    logging.info(logcode(2) + ': ' + BASE_PATH)

    OUTPUT_PATH = os.path.join(BASE_PATH, 'output')
    CONFIG_PATH = os.path.join(BASE_PATH, 'configs')

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(CONFIG_PATH, exist_ok=True)

    DB_ENGINE = "sqlite3"
    REPORT_TEMPLATE_PATH = os.path.join(CONFIG_PATH, "发票分析报告_ZJBL定制模版.xlsx")

    init_configuration()

    def load_city_trie(city_file='city.txt'):
        logging.info(logcode(6))
        from pytrie import SortedStringTrie as Trie
        zip_trie = Trie()

        if not os.path.exists(city_file):
            # raise IOError("NotFoundFile")
            return zip_trie

        with open(city_file, 'r', encoding='UTF-8') as f_obj:
            for f_line in f_obj.readlines():
                try:
                    line = f_line.split(',')
                    zip_trie[line[0]] = line[1]
                except Exception:
                    raise Exception("无法解析此行城市信息：{}".format(f_line))
        return zip_trie

    def load_province_trie(city_file='city.txt'):
        logging.info(logcode(7))
        from pytrie import SortedStringTrie as Trie
        zip_trie = Trie()

        if not os.path.exists(city_file):
            return zip_trie

        with open(city_file, 'r', encoding='utf-8') as f_obj:
            for f_line in f_obj.readlines():
                try:
                    line = f_line.split(',')
                    lvalue = re.sub('\n', '', line[1])
                    zip_trie[line[0]] = lvalue
                except Exception:
                    logging.info(line)
                    raise Exception("无法解析此行城市信息：{}".format(f_line))
        return zip_trie

    CITY_CODES = load_city_trie(city_file=os.path.join(CONFIG_PATH, 'city20181227.txt'))

    PROVINCE_CODES = load_province_trie(city_file=os.path.join(CONFIG_PATH, 'province20181227.txt'))

    HIERARCHICAL_LOCATIONS_DICT = load_hierarchical_locations(province_file=os.path.join(CONFIG_PATH, 'province'),
                                                              city_file=os.path.join(CONFIG_PATH, 'city'))

    PYP_ENTRY_BARS = OrderedDict({
        'E-企业规模-1': {'区间': '[20000000,inf)',     '补救': nan},
        'E-买方结构-1': {'区间': '[0.2,inf)',          '补救': nan},
        'E-成长性-1':   {'区间': '[-0.2,inf)',         '补救': nan},
        'E-成长性-2':   {'区间': '[-0.4,inf)',         '补救': nan},
        'E-稳定性-1':   {'区间': '(-inf,0.3]',         '补救': nan},
        'E-稳定性-2':   {'区间': '(-inf,0.15]',        '补救': nan},
        'E-连续性-1':   {'区间': '(-inf,3)',           '补救': nan},
        'E-连续性-2':   {'区间': '[24,inf)',           '补救': nan},
        'E-连续性-3':   {'区间': '(-inf,4]',           '补救': nan}
    })

    VARIABLE_CREDIT_SCORE_CONFIG = OrderedDict({
        'A': {'信用评分区间': '[90,inf)',  '基础额度调节系数': 1.1, '适用利率': 0.108, '最高限额系数': 0.02},
        'B': {'信用评分区间': '[80,90)',   '基础额度调节系数': 1.0, '适用利率': 0.144, '最高限额系数': 0.018},
        'C': {'信用评分区间': '[70,80)',   '基础额度调节系数': 0.7, '适用利率': 0.162, '最高限额系数': 0.015},
        'D': {'信用评分区间': '[60,70)',   '基础额度调节系数': 0.5, '适用利率': 0.180, '最高限额系数': 0.012},
        'E': {'信用评分区间': '[50,60)',   '基础额度调节系数': 0.3, '适用利率': 0.216, '最高限额系数': 0.008},
        'F': {'信用评分区间': '(-inf,50)', '基础额度调节系数': 0.0, '适用利率': 0.360, '最高限额系数': 0.000},
    })

    JOINT_WHITE_BUYERS_CONFIG = OrderedDict({
        1: {'内部标签': 1, '核额贡献系数': 0.7,  '红冲率区间': '(-inf,0.15]', '作废率区间': '(-inf,0.35]', '红冲作废率区间': '(-inf,0.40]',
            '最高限额贡献金额': 1000000, '最高限额贡献系数': 1.0, '静态系数加点1': 0.3, "静态系数加点2": 0.2,  "静态系数加点3": 0.1},
        2: {'内部标签': 2, '核额贡献系数': 0.5,  '红冲率区间': '(-inf,0.15]', '作废率区间': '(-inf,0.35]', '红冲作废率区间': '(-inf,0.40]',
            '最高限额贡献金额': 1000000, '最高限额贡献系数': 1.0, '静态系数加点1': 0.3, "静态系数加点2": 0.2,  "静态系数加点3": 0.1},
        3: {'内部标签': 3, '核额贡献系数': 0.3,  '红冲率区间': '(-inf,0.15]', '作废率区间': '(-inf,0.35]', '红冲作废率区间': '(-inf,0.40]',
            '最高限额贡献金额': 1000000, '最高限额贡献系数': 1.0, '静态系数加点1': 0.3, "静态系数加点2": 0.2,  "静态系数加点3": 0.1},
        4: {'内部标签': 4, '核额贡献系数': 0.2,  '红冲率区间': '(-inf,0.15]', '作废率区间': '(-inf,0.35]', '红冲作废率区间': '(-inf,0.40]',
            '最高限额贡献金额': 1000000, '最高限额贡献系数': 1.0, '静态系数加点1': 0.3, "静态系数加点2": 0.2,  "静态系数加点3": 0.1},
        5: {'内部标签': 5, '核额贡献系数': 0.1,  '红冲率区间': '(-inf,0.15]', '作废率区间': '(-inf,0.35]', '红冲作废率区间': '(-inf,0.40]',
            '最高限额贡献金额': 1000000, '最高限额贡献系数': 1.0, '静态系数加点1': 0.3, "静态系数加点2": 0.2,  "静态系数加点3": 0.1},
                                                })

    DYNAMIC_WHITE_BUYERS_CONFIG = OrderedDict({
        1: {'阈值': 10, '核额贡献系数': 0.9, '红冲率区间': '(-inf,0.15]', '作废率区间': '(-inf,0.35]', '红冲作废率区间': '(-inf,0.40]',
            '最高限额贡献金额': 1000000, '最高限额贡献系数': 1.0},
        2: {'阈值': 7, '核额贡献系数': 0.7, '红冲率区间': '(-inf,0.15]', '作废率区间': '(-inf,0.35]', '红冲作废率区间': '(-inf,0.40]',
            '最高限额贡献金额': 1000000, '最高限额贡献系数': 1.0},
        3: {'阈值': 5, '核额贡献系数': 0.5, '红冲率区间': '(-inf,0.15]', '作废率区间': '(-inf,0.35]', '红冲作废率区间': '(-inf,0.40]',
            '最高限额贡献金额': 1000000, '最高限额贡献系数': 1.0},
        4: {'阈值': 3, '核额贡献系数': 0.3, '红冲率区间': '(-inf,0.15]', '作废率区间': '(-inf,0.35]', '红冲作废率区间': '(-inf,0.40]',
            '最高限额贡献金额': 1000000, '最高限额贡献系数': 1.0},
        5: {'阈值': 1, '核额贡献系数': 0.1, '红冲率区间': '(-inf,0.15]', '作废率区间': '(-inf,0.35]', '红冲作废率区间': '(-inf,0.40]',
            '最高限额贡献金额': 1000000, '最高限额贡献系数': 1.0},
    })

    QUALITY_BUYERS_CONFIG = {
        'X1': {'近12月购买月数': '[1,12]', '远12月购买月数': '[1,12]', '近6月购买月数': '[1,6]'},
        'X2': {'近12月购买月数': '[2,12]', '远12月购买月数': '[2,12]', '近6月购买月数': '[1,6]'},
        'X3': {'近12月购买月数': '[3,12]', '远12月购买月数': '[3,12]', '近6月购买月数': '[1,6]'},
        'X9': {'近12月购买月数': '[9,12]', '远12月购买月数': '[9,12]', '近6月购买月数': '[1,6]'},
        'X6': {'近12月购买月数': '[6,12]', '远12月购买月数': '[6,12]', '近6月购买月数': '[1,6]'},
        'T1': {'同比远6月购买数': '[1,6]', '近3月购买月数': '[1,3]'},
        'T2': {'近2月购买月数': '[1,2]', '近12月最长连续零开票数': '[0,2]'},
        'T3': {'近3月购买月数': '[1,3]', '同比远6月含购买数': '[1,7]'},
    }

    PRODUCT_CODES = OrderedDict({
        'PYP': {'别名': '国际货代-票一拍项目', '准入规则': 'PYP_ENTRY_BARS',
                '联合白名单配置': 'JOINT_WHITE_BUYERS_CONFIG',
                '动态白名单配置': 'DYNAMIC_WHITE_BUYERS_CONFIG',
                '最高限额': 20000000.00, '审慎限额': 5000000}
    })

    buyers_profile = dict.fromkeys(["核心企业买方", "买方主要分布城市", "近一年十大买方", "远一年十大买方",
                                    "近一年十大核心企业买方", "远一年十大核心企业买方",
                                    "近一年动态白名单买方", "远一年动态白名单买方", "近一年稳定买方", "近一年买方数量",
                                    "电信核企", "快递核企", "近一年原买方开票占比", "近一年原买方行业关键字匹配"])

    EXCEL_COLUMN_SORTING = list(map(chr, range(ord('A'), ord('Z')+1))) + ["A" + x for x in list(map(chr, range(ord('A'), ord('Z')+1)))]

    R_CONTAINER = {"报告信息": {"报告编码": {"pos": "D4"}, "报告日期": {"pos": "G4"}, "报告分析策略": {"pos": "J4"},
                            "企业名称": {"pos": "D5"}, "企业税号": {"pos": "G5"}, "报告数字签名": {"pos": "J5"}},

                   "企业基础信息": {
                              "销项发票累计采集时长(月)": {"pos": "D13", "signature": 0},
                              "近1年销项开票规模(元)": {"pos": "F13", "signature": 1},
                              "近1年销项开票张数": {"pos": "H13", "signature": 1},
                              "最近一次销项开票时间": {"pos": "J13", "signature": 0},
                              "企业所在区域": {"pos": "D14", "signature": 0},
                              "企业联系方式": {"pos": "F14", "signature": 0},
                              "票面银行账户": {"pos": "H14", "signature": 0},
                              "账户开户银行": {"pos": "J14", "signature": 0},
                              "最新企业开票地址": {"pos": "D15", "signature": 0}
                              },

                   "分类加总开票时序": {"月度-倒时序": {"pos": "D37:AM37"},
                                "全量买方开票金额(元）-倒时序": {"pos": "D38:AM38"},
                                "动态白名单买方开票金额(元)-倒时序": {"pos": "D39:AM39"},
                                "静态白名单买方开票金额(元)-倒时序": {"pos": "D40:AM40"},
                                "全量买方开票笔数-倒时序": {"pos": "D41:AM41"},
                                "动态白名单买方开票笔数-倒时序": {"pos": "D42:AM42"},
                                "静态白名单买方开票笔数-倒时序": {"pos": "D43:AM43"}},

                   "十大下游交易买方分析": {
                       "十大买方开票占比对比": {
                           "远近12个月前十买家":              {"pos": "C48",'anonymize':'近12个月前十买家|远12个月前十买家'}
},
                       "十大买方基础指标":{"pos": "C60", "anonymize": '买方统一名称'}},

                   "下游交易买方TOP5区域分布": {
                       "省份":               {"pos": "C74:C78", "signature": 0},
                       "近12个月开票额(元)":  {"pos": "D74:D78", "signature": 0},
                       "近12月买方数量":      {"pos": "D74:D78", "signature": 0},
                       "远12个月开票额(元)":  {"pos": "E74:E78", "signature": 0},
                       "远12月买方数量":      {"pos": "F74:F78", "signature": 0},
                       "近12月开票增长率":    {"pos": "G74:G78", "signature": 0},
                       "近12月买方数量增长率": {"pos": "H74:H78", "signature": 0},
                       "近12月红冲比例":      {"pos": "I74:I78", "signature": 0},
                       "近12月作废比例":      {"pos": "K74:K78", "signature": 0},
                       "首次开票月份":        {"pos": "L74:L78", "signature": 0},
                       "最新开票月份":        {"pos": "M74:M78", "signature": 0}
                   },

                   "开票元信息": {
                       "企业历史开票名称/税号信息": {"pos": "D85"},
                       "企业历史开票地址信息": {"pos": "D90"},
                       "企业历史开票银行账户信息": {"pos": "D96"}},

                   "准入规则": {
                       "E-企业规模-1": {"企业实际表现": {"pos": "H111", "signature": 1}, "指标判断": {"pos": "I111"}},
                       "E-买方结构-1": {"企业实际表现": {"pos": "H112", "signature": 1}, "指标判断": {"pos": "I112"}},
                       "E-成长性-1":   {"企业实际表现": {"pos": "H113", "signature": 1}, "指标判断": {"pos": "I113"}},
                       "E-成长性-2":   {"企业实际表现": {"pos": "H114", "signature": 0}, "指标判断": {"pos": "I114"}},
                       "E-稳定性-1":   {"企业实际表现": {"pos": "H115", "signature": 1}, "指标判断": {"pos": "I115"}},
                       "E-稳定性-2":   {"企业实际表现": {"pos": "H116", "signature": 1}, "指标判断": {"pos": "I116"}},
                       "E-连续性-1":   {"企业实际表现": {"pos": "H117", "signature": 1}, "指标判断": {"pos": "I117"}},
                       "E-连续性-2":   {"企业实际表现": {"pos": "H118", "signature": 1}, "指标判断": {"pos": "I118"}},
                       "E-连续性-3":   {"企业实际表现": {"pos": "H119", "signature": 0}, "指标判断": {"pos": "I119"}},
                   },

                   "信用评分": {
                       "S-买方结构-1":   {"企业实际表现": {"pos": "G126", "signature": 1}, "得分": {"pos": "H126"}},
                       "S-买方结构-2":   {"企业实际表现": {"pos": "G127", "signature": 1}, "得分": {"pos": "H127"}},
                       "S-买方集中度-1": {"企业实际表现": {"pos": "G128", "signature": 1}, "得分": {"pos": "H128"}},
                       "S-销售稳定性-1": {"企业实际表现": {"pos": "G129", "signature": 0}, "得分": {"pos": "H129"}},
                       "S-销售稳定性-2": {"企业实际表现": {"pos": "G130", "signature": 1}, "得分": {"pos": "H130"}},
                       "S-销售连续性-1": {"企业实际表现": {"pos": "G131", "signature": 1}, "得分": {"pos": "H131"}},
                       "S-销售连续性-2": {"企业实际表现": {"pos": "G132", "signature": 1}, "得分": {"pos": "H132"}},
                       "S-销售成长性-1": {"企业实际表现": {"pos": "G133", "signature": 1}, "得分": {"pos": "H133"}},
                       "S-销售成长性-2": {"企业实际表现": {"pos": "G134", "signature": 0}, "得分": {"pos": "H134"}},
                       "S-销售成长性-3": {"企业实际表现": {"pos": "G135", "signature": 1}, "得分": {"pos": "H135"}},
                       "S-销售成长性-4": {"企业实际表现": {"pos": "G136", "signature": 1}, "得分": {"pos": "H136"}},
                       "S-业务画像-1":   {"企业实际表现": {"pos": "G137", "signature": 0}, "得分": {"pos": "H137"}},
                       "S-业务画像-2":   {"企业实际表现": {"pos": "G138", "signature": 1}, "得分": {"pos": "H138"}},
                       "S-业务画像-3":   {"企业实际表现": {"pos": "G139", "signature": 1}, "得分": {"pos": "H139"}},
                       "S-业务画像-4":   {"企业实际表现": {"pos": "G140", "signature": 0}, "得分": {"pos": "H140"}},
                   },

                   "信用测额": {
                       "L-基础-1":   {"额度值": {"pos": "G148", "signature": 1}, "可配参数": {"pos": "H148"}},
                       "L-利润-1":   {"额度值": {"pos": "G149", "signature": 1}, "可配参数": {"pos": "H149"}},
                       "L-调整-1":   {"额度值": {"pos": "G150", "signature": 1}}
                   },

                   "销项发票买方时序表": {
                       "TOP50买方时序表": {"pos": "A4",'anonymize':'统一名称'},
                       "近36个开票月份": {"pos": "E3:AN3"}
                   },

                   "销项发票分品类时序表": {
                       "海运费收入-近远12个月总额":        {"pos": "B4:C4"},
                       "港杂费收入-近远12个月总额":        {"pos": "B5:C5"},
                       "其他收入-近远12个月总额":          {"pos": "B6:C6"},
                       "近36个开票月份":                {"pos": "D3:AM3"},
                       "海运费收入-近36个开票月份开票量": {"pos": "D4:AM4"},
                       "港杂费收入-近36个开票月份开票量": {"pos": "D5:AM5"},
                       "其他收入-近36个开票月份开票量":   {"pos": "D6:AM6"},
                   },

                   }

    R_CONTAINER["报告信息"]["报告分析策略"]["value"] = "FORWARDERS-V1.0"
    R_CONTAINER["报告信息"]["报告编码"]["value"] = "2021000001"
    R_CONTAINER["信用评分"]["S-业务画像-4"]["企业实际表现"]["value"] = 0.1
    R_CONTAINER["信用测额"]["L-基础-1"]["可配参数"]["value"] = 2.
    R_CONTAINER["信用测额"]["L-利润-1"]["可配参数"]["value"] = 0.08

    logging.info(logcode(8))
