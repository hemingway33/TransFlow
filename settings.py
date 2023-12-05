import os
import sys
import re
import ahocorasick
from collections import OrderedDict
from bloom_filter import BloomFilter
from pytrie import SortedStringTrie as Trie


BASE_PATH = "/Users/xiaoqingyu/PycharmProjects/TransFlow"
print("项目根目录： {}".format(BASE_PATH))

CONFIG_PATH = os.path.join(BASE_PATH, 'configs')

OUTPUT_PATH = os.path.join(BASE_PATH, 'output')
os.makedirs(OUTPUT_PATH, exist_ok=True)


def load_organizations(organization_file):
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


low_meanings = ['省$', '市$', '自治区$', '自治州$']
low_meaning_regex = "|".join(low_meanings)


def remove_location_unit(text):
    text = text.strip()
    return re.sub(low_meaning_regex, '', text)


def load_administrative_divisions(province_file, city_file, area_file):

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


def load_city_trie(city_file='city.txt'):
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
                print(line)
                raise Exception("无法解析此行城市信息：{}".format(f_line))
    return zip_trie


def make_stop_word_machine():
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
    return stop_words_machine


administrative_divisions_bloom, administrative_divisions_dict = load_administrative_divisions(province_file=os.path.join(CONFIG_PATH, 'province'),
                                                                                              city_file=os.path.join(CONFIG_PATH, 'city'),
                                                                                              area_file=os.path.join(CONFIG_PATH, 'areas'))

administrative_divisions_patt = "|".join(administrative_divisions_dict.keys())

organizations_types = load_organizations(organization_file=os.path.join(CONFIG_PATH, 'organization'))
organization_patt = "|".join(organizations_types)

stop_words_machine = make_stop_word_machine()

