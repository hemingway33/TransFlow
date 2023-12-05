import ahocorasick
import pandas as pd
from pandas import read_excel
import pickle


class TransFlowTagger:
    def __int__(self):
        self.key_words_scheme_path_ = "流水标签规则文档v20220517.xlsx"
        self.tag_machine_path_ = ".pickle"
        self.tag_machine_ = None

    def make_tag_machine(self, persist=True):
        trans_intent_machine = ahocorasick.Automaton()
        key_words_scheme = read_excel(self.key_words_scheme_path_, sheet_name="流水标签词库管理表")
        print(key_words_scheme)
        if persist:
            assert key_words_scheme is not None
            with open(self.tag_machine_path_, 'wb') as file:
                pickle.dump(key_words_scheme, file)
        return self

    def load_tag_machine(self):
        self.tag_machine_ = pickle.loads(self.tag_machine_path_)
        return self


date = pd.DataFrame()

# 三级标签
date.loc[date[((date['fyzy'].str.contains("货款|采|预付|材料|模具|定金|订金|佣金|尾款|預付|铜材|钢材|板|样品|物料|销售款|纸箱款|电池|锂电|软件|打样|教材|教辅|台|套|个|汽油|花纸|纸箱|吊牌|配件|设备|布|料|摄像头|貨款|热水器|空调|冰箱|洗衣机|书|布料|线款|钉钮|胶带|拉链|电脑|合同款|食品|产品|锁|芯片|玻璃|机器|毛巾|包装盒|桌椅|筷子|气缸|润滑油|刀具|五金|轮胎|家具|物资|贸易款|机床|油费|油款|货|纸|铜|代销|网上购物")) & (~(date['fyzy'].str.contains("房租|平台|收银台|柜台|贷|证书|安装|财付通|支付宝|个人|借款"))) & (date['dfhm'] != date['zhmc'])) | (date['dfhm'].str.contains("客户备付金"))].index, 'sign'] = '货款往来（不区分资金方向）'
date.loc[date[(len(date['dfhm']) > 5) & ((date['fyzy'].str.contains("服务|运费|运输|运货费|货代费|货运代理费|代理费|快件费|港杂费|包装费|包装款|包装制作|加工|装修|工程|保安|快递|物流|劳务|培训|安装|保养|租车费|设计费|维修|维护|售后|洗水|增值服务|印刷费|打印费|招聘费|制作费|会展位费|安保费|充电费|会务费|展位费|评估费|会展|招聘|展会|FOB费|拖车费|检修|修理费|餐费|住宿费|仓储|建设费|机票费|餐饮费|油卡充值|收派服务费|施工|消防|速递款|代维费|广告费|推广费|宣传费|促销费|管理费|电信费|体检费|电话费|电信|减薄费")) | (date['dfhm'].str.contains("报关|速运")) | ((date['fyzy'].str.contains("电费")) & (date['jyjdbz'] == '贷') & (date['jyje'] > 0) & (~(date['fyzy'].str.contains("城市维护建设税"))) & (date['dfhm'] != date['zhmc'])))].index, 'sign'] = '服务往来（不区分资金方向）'
date.loc[date[date['fyzy'].str.contains("押金")].index, 'sign'] = '押金（不区分资金方向）'
date.loc[date[date['fyzy'].str.contains("退款|退汇")].index, 'sign'] = '退款（不区分资金方向）'
date.loc[date[date['fyzy'].str.contains("保证金")].index, 'sign'] = '保证金（不区分资金方向）'


# 二级标签
date.loc[date[date['sign'].isin(['货款往来（不区分资金方向）','服务往来（不区分资金方向）','押金（不区分资金方向）','退款（不区分资金方向）','保证金（不区分资金方向）'])].index,'sign2']='主营业务'

# 一级标签
date.loc[date[date['sign2'].isin(['主营业务', '日常运营', '非经常项目', '第三方服务'])].index, 'sign1'] = '经营类'


if __name__ == "__main__":
    TransFlowTagger().make_tag_machine()



