import pandas as pd
import numpy as np
import os
import settings
from preprocess.anchoring.identify_anchor import sketch_core_brands
from preprocess.preprocess import DataQualityManager
import warnings
from utils.Logger import Logger,logcode
logging = Logger(level="info", name=__name__).getlog()
warnings.filterwarnings("ignore")
from utils.utils import dump_model_to_file


def compute_invoice_features_and_score(supplier_name, slice_date, supplier_source, df1_full_hist, df2_full_hist, extract_quality_buyers=True,
                                       client_outpath='', save_source_data=False):
    start_run_time = pd.Timestamp.now()
    logging.info('start run :{} '.format(start_run_time))

    top_buyers_profile = pd.DataFrame()
    top_10_deal_table = pd.DataFrame()

    setting = settings.PRODUCT_CODES[supplier_source.strip()]
    settings.R_CONTAINER["报告信息"]["企业名称"]["value"] = supplier_name

    observeYear = slice_date.year
    observeMonth = slice_date.month
    observeTime = pd.Timestamp(observeYear, observeMonth, 1).tz_localize('Asia/Shanghai')
    settings.R_CONTAINER["报告信息"]["报告日期"]["value"] = observeTime.strftime("%Y-%m-%d")
    logging.info(logcode(11))

    if save_source_data:
        df1_full_hist.to_csv(os.path.join(client_outpath, "原始发票数据.csv"), encoding="utf-8-sig")
        logging.debug("存储发票原始数据集")
    df1_source_size = len(df1_full_hist)
    logging.info(logcode(12,df1_source_size))

    logging.info(logcode(13))
    df1_full_hist = df1_full_hist[~np.logical_and(df1_full_hist['合计金额'] < 0.0, df1_full_hist['状态'] == '作废')]
    logging.info("{}:{}".format(logcode(14),df1_full_hist.shape[0]))
    df1_full_hist.drop_duplicates(subset="发票id", inplace=True)

    df1_full_hist = df1_full_hist[~ (np.logical_or(df1_full_hist['购方名称'].isin(["NULL", ""]),
                                                   pd.isna(df1_full_hist['购方名称'])))]
    df1_full_hist['购方名称'] = df1_full_hist['购方名称'].str.strip(b'\x00'.decode())
    first_invoice_date = df1_full_hist[df1_full_hist["状态"] == "有效"]['开票日期'].min()

    last_invoice_date = df1_full_hist[df1_full_hist["状态"] == "有效"]['开票日期'].max()
    if pd.isna(last_invoice_date): last_invoice_date = first_invoice_date
    settings.R_CONTAINER["企业基础信息"]["最近一次销项开票时间"]["value"] = last_invoice_date

    num_buyers = len(df1_full_hist['购方名称'].unique())
    if num_buyers > 5000:
        logging.info("{}的买方数为{},为加快运行速度,限制购方数量为 TOP {}".format(supplier_name, num_buyers, 5000))
        df1_full_hist_back = df1_full_hist.groupby(by='购方名称', as_index=False).sum()
        df1_full_hist_back = df1_full_hist_back.sort_values(by='合计金额', ascending=False)
        df1_top1000_buyers = list(df1_full_hist_back.head(5000)['购方名称'].unique())
        df1_full_hist = df1_full_hist[df1_full_hist['购方名称'].isin(df1_top1000_buyers)]

    try:
        sellerTaxNo = df1_full_hist['销方税号'].unique().tolist()
        logging.info('{}:{}'.format(logcode(15),sellerTaxNo))
    except Exception as e:
        logging.info('{}:{}'.format(logcode(16),e))
        sellerTaxNo = ['XXX']

    if df1_full_hist.empty:
        logging.error('{}{}'.format(logcode(17),supplier_name))

    sellerTaxNo1 = sellerTaxNo[0]
    settings.R_CONTAINER["报告信息"]["企业税号"]["value"] = sellerTaxNo1

    seller_city_code, seller_city = DataQualityManager.parse_city(sellerTaxNo1)
    logging.info("{}{}， 城市为：{}".format(logcode(18), seller_city_code, seller_city))
    settings.R_CONTAINER["企业基础信息"]["企业所在区域"]["value"] = seller_city

    seller_province_code, seller_province = DataQualityManager.parse_province(sellerTaxNo1)
    logging.info("{}{}， 省份为：{}".format(logcode(19),seller_province_code, seller_province))

    df2_full_hist['开票日期'] = pd.merge(df1_full_hist, df2_full_hist, on="发票id", how='right')['开票日期']
    df2_full_hist.drop_duplicates(subset=["发票id", "序号", "金额"], inplace=True)
    df1_invoice_IDs = df1_full_hist[df1_full_hist["开票日期"] >= observeTime - pd.Timedelta(365 * 2, unit='D')]['发票id']
    missed_invoice_IDs, missedCountRatio = DataQualityManager.findMissingIDs(df1_invoice_IDs.unique(), df2_full_hist['发票id'].unique())
    logging.info('{}: {}'.format(logcode(93),missedCountRatio))

    logging.info(logcode(20))
    related_companies = pd.DataFrame(columns=['关联企业名称'], data=[[supplier_name]])
    related_companies_list = list(related_companies['关联企业名称'].unique())
    df1_full_hist, affliated_df_full_hist = DataQualityManager.dropAffliates(df=df1_full_hist, affliates=related_companies_list, name_column='购方名称')
    related_companies_temp = DataQualityManager.cleanRelatedCompany(related_companies.copy())
    df1 = DataQualityManager.selectByTimeRange(df=df1_full_hist, observeTime=observeTime, time_column='开票日期', to_observe_time_period=pd.Timedelta(365 * 3, unit='D'))
    del df1_full_hist

    recent_N_months_cancel_rate = {}
    recent_N_months_redcorrection_rate = {}
    recent_N_months_person_percent = {}

    logging.info(logcode(21))
    df1 = DataQualityManager.rename_person_buyer(df1, name_column='购方名称')

    NS = [6, 12]
    logging.info(logcode(22))
    for N in NS:
        df1_recent_N_months = DataQualityManager.selectByTimeRange(df=df1, observeTime=observeTime, time_column='开票日期', to_observe_time_period=pd.DateOffset(months=N))
        df1_recent_N_months_valid = df1_recent_N_months[df1_recent_N_months['状态'] == '有效']
        recent_N_months_valid_absolute_sum = np.abs(df1_recent_N_months_valid["合计金额"]).sum() + np.abs(df1_recent_N_months_valid["合计税额"]).sum()
        recent_N_months_invoice_total = df1_recent_N_months_valid['合计金额'].sum() + df1_recent_N_months_valid['合计税额'].sum()

        if N == 12: recent_12_months_invoice_total = recent_N_months_invoice_total
        df1_recent_N_months_cancel = df1_recent_N_months[df1_recent_N_months['状态'] == '作废']
        recent_N_months_cancel_total = np.abs(df1_recent_N_months_cancel['合计金额']).sum() + np.abs(df1_recent_N_months_cancel['合计税额']).sum()
        recent_N_months_cancel_rate[N] = np.round(recent_N_months_cancel_total / (recent_N_months_valid_absolute_sum + recent_N_months_cancel_total), 4)
        df1_recent_N_months_redcorrection = df1_recent_N_months_valid[df1_recent_N_months_valid['合计金额'] < 0]
        recent_N_months_redcorrection_total = np.abs(df1_recent_N_months_redcorrection['合计金额']).sum() + np.abs(df1_recent_N_months_redcorrection['合计税额']).sum()
        recent_N_months_redcorrection_rate[N] = np.round(recent_N_months_redcorrection_total / recent_N_months_valid_absolute_sum, 4)

        df1_recent_N_months_valid_person = df1_recent_N_months_valid[df1_recent_N_months_valid['购方名称'] == '个人']
        recent_N_months_person_total = np.abs(df1_recent_N_months_valid_person['合计金额']).sum() + np.abs(df1_recent_N_months_valid_person['合计税额']).sum()
        recent_N_months_person_percent[N] = recent_N_months_person_total / recent_N_months_valid_absolute_sum

    logging.info(logcode(23))
    df1 = df1[df1['购方名称'] != '个人']

    logging.info(logcode(24))
    df1 = DataQualityManager.identifySameBuyers(df1, name_column='购方名称', threshold=0.85)
    logging.info(logcode(25))
    df1_temp = DataQualityManager.identifyRelatedCompanies(df1, related_companies_temp, threshold=0.9, name_column='统一名称')  # 此处用统一名称更合适，原先是预处理后的名称

    related_companies_temp2 = df1_temp[df1_temp['关联企业识别名'].notnull()]['购方名称']
    if len(related_companies_temp2) > 0:
        df1, affliated_df1 = DataQualityManager.dropAffliates(df=df1, affliates=related_companies_temp2, name_column='购方名称')  # 此处用购方名称更合适，原先是预处理后的名称
    else:
        affliated_df1 = pd.DataFrame()

    logging.info('{}'.format(logcode(26)))
    df1 = sketch_core_brands(df1, settings.anchor_buyer_lsh)
    thisCoreBrands = df1['核企识别名'][df1['核企识别名'].notnull()].unique()
    logging.info('{}'.format(logcode(27,len(thisCoreBrands))))

    if not affliated_df_full_hist.empty:
        if not affliated_df1.empty:
            affliated__df = pd.concat([affliated_df_full_hist, affliated_df1], join='outer')
        else:
            affliated__df = affliated_df_full_hist
    else:
        if not affliated_df1.empty: affliated__df = affliated_df1
        else: affliated__df = pd.DataFrame()
    if not affliated__df.empty:
        affliated__df_recent_12 = DataQualityManager.selectByTimeRange(df=affliated__df, observeTime=observeTime,
                                                                       time_column='开票日期',
                                                                       to_observe_time_period=pd.DateOffset(months=12))

        affliated__df_recent_12 = affliated__df_recent_12[affliated__df_recent_12['状态'] == '有效']

        recent_12_months_related_companies_total = affliated__df_recent_12['合计金额'].sum() + affliated__df_recent_12['合计税额'].sum()

        recent_12_months_related_companies_percent = np.abs(recent_12_months_related_companies_total) / recent_12_months_invoice_total
    else:
        recent_12_months_related_companies_percent = 0.0
    logging.info("{}".format(logcode(28,recent_12_months_related_companies_percent)))

    df1['合计金额'] = df1['合计金额'] + df1['合计税额'].replace(to_replace=np.nan, value=0)

    df1 = df1[~np.logical_and(df1['状态'] == '作废', df1['合计金额'] < 0)]

    logging.info(logcode(29))
    df1 = DataQualityManager.yearMonthlize(df1, dateColumn="开票日期")
    buyer_anomolous_ratios = DataQualityManager.computeBuyerAnomalousInvoiceRatios(df1, observeTime, period_days=365)
    if len(buyer_anomolous_ratios) > 0:
        df1['近一年作废比率'] = df1.apply(lambda row: buyer_anomolous_ratios[row['统一名称']]['近一年作废率'], axis=1)
        df1['近一年红冲比率'] = df1.apply(lambda row: buyer_anomolous_ratios[row['统一名称']]['近一年红冲率'], axis=1)
    else:
        df1['近一年作废比率'] = np.nan
        df1['近一年红冲比率'] = np.nan

    logging.info(logcode(30))
    df1['买方所在省份'] = df1.apply(lambda row: DataQualityManager.parse_province(row['购方税号'])[1], axis=1)

    logging.info(logcode(31))
    df1['买方所在城市'] = df1.apply(lambda row: DataQualityManager.parse_city(row['购方税号'])[1], axis=1)

    logging.info(logcode(32))
    df1_valid_source = df1[df1["状态"] == "有效"]
    df1_cancel_source = df1[df1["状态"] == "作废"]
    df1_redCorrection_source = df1[np.logical_and(df1["合计金额"] < 0, df1["状态"] == "有效")]

    df1_valid = DataQualityManager.yearMonthlize(df1_valid_source, dateColumn="开票日期")
    df1_cancel = DataQualityManager.yearMonthlize(df1_cancel_source, dateColumn="开票日期")
    df1_redCorrection = DataQualityManager.yearMonthlize(df1_redCorrection_source, dateColumn="开票日期")
    df1_valid["是否美元发票"] = df1_valid["合计金额"].apply(lambda x: DataQualityManager.is_foreign_denominated(x))
    df1_valid["是否专用发票"] = df1_valid['发票类型'].isin(['s', '004'])

    df2_valid_source = df2_full_hist[df2_full_hist['发票id'].isin(list(df1_valid_source['发票id'].unique()))]
    del df2_full_hist

    df2_valid_source = DataQualityManager.getMatchedInvoiceDate(df1=df1_valid_source, df2=df2_valid_source)
    commodities_num_limit = 5000
    if len(df2_valid_source['商品名称'].unique()) > commodities_num_limit:
        df2_full_hist_back = df2_valid_source.groupby(by='商品名称', as_index=False).sum()
        df2_full_hist_back = df2_full_hist_back.sort_values(by='价税合计', ascending=False)
        df2_top5000_commodity = list(df2_full_hist_back.head(commodities_num_limit)['商品名称'].unique())
        df2_valid_source = df2_valid_source[df2_valid_source['商品名称'].isin(df2_top5000_commodity)]

    df2_valid_source = df2_valid_source[np.logical_not(np.logical_or(df2_valid_source['商品名称'].str.contains('合计'),
                                                                     df2_valid_source['商品名称'].str.contains('折扣')))]

    df2_valid_source["是否海运代理费"] = np.logical_and(
        df2_valid_source["商品名称"].str.contains("|".join(settings.shipping_service_key_words), regex=True),
        ~df2_valid_source["商品名称"].str.contains("|".join(settings.shipping_service_key_words_excludes), regex=True))
    df2_valid_source["是否港杂费"] = np.logical_and(
        df2_valid_source["商品名称"].str.contains("|".join(settings.port_charges_key_words), regex=True),
        ~df2_valid_source["商品名称"].str.contains("|".join(settings.port_charges_key_words_excludes), regex=True))
    df2_valid = DataQualityManager.yearMonthlize(df2_valid_source, dateColumn="开票日期")

    df2_valid_latest_12 = DataQualityManager.selectByTimeRange(df=df2_valid, time_column='年月', observeTime=observeTime, to_observe_time_period=pd.Timedelta(365, unit='D'))

    shipping_revenue_latest_12_months = df2_valid_latest_12[df2_valid_latest_12["是否海运代理费"] == True]["价税合计"].sum()
    settings.R_CONTAINER["准入规则"]["E-企业规模-1"]["企业实际表现"]["value"] = shipping_revenue_latest_12_months
    port_charges_latest_12_sum_by_yearMonth = DataQualityManager.groupByMultipleColumns(df=df2_valid_latest_12[df2_valid_latest_12["是否港杂费"] == True],
                                                                                        sorted_columns_list=['年月'], as_index=False,
                                                                                        aggregate_method='sum', end=observeTime - pd.Timedelta(28, 'D'))
    port_charges_latest_12_trend_monthly = DataQualityManager.identify_trend(port_charges_latest_12_sum_by_yearMonth['价税合计'].tolist())
    settings.R_CONTAINER["信用评分"]["S-业务画像-3"]["企业实际表现"]["value"] = port_charges_latest_12_trend_monthly

    shipping_revenue_sum_by_yearMonth = DataQualityManager.groupByMultipleColumns(df=df2_valid[df2_valid["是否海运代理费"] == True],
                                                                                  sorted_columns_list=['年月'], as_index=False,
                                                                                  aggregate_method='sum', end=observeTime - pd.Timedelta(28, 'D'))
    port_charges_sum_by_yearMonth = DataQualityManager.groupByMultipleColumns(df=df2_valid[df2_valid["是否港杂费"] == True],
                                                                              sorted_columns_list=['年月'], as_index=False,
                                                                              aggregate_method='sum', end=observeTime - pd.Timedelta(28, 'D'))

    df1_valid_sum_by_yearMonth = DataQualityManager.groupByMultipleColumns(df=df1_valid,
                                                                           sorted_columns_list=['年月'],
                                                                           as_index=False,
                                                                           aggregate_method='sum',
                                                                           end=observeTime - pd.Timedelta(28, 'D'),
                                                                           invoice_num=True)

    if pd.isna(first_invoice_date): first_invoice_date = observeTime

    first_invoice_date_with_specified_type = df1[np.logical_and(df1['发票类型'].isin(['s', '004']), df1["状态"] == "有效")]['开票日期'].min()
    if pd.isna(first_invoice_date_with_specified_type): first_invoice_date_with_specified_type = observeTime

    total_invoice_months_with_specified_type = DataQualityManager.month_diff(first_invoice_date_with_specified_type, observeTime)

    total_invoice_months = DataQualityManager.month_diff(first_invoice_date, observeTime)
    settings.R_CONTAINER["企业基础信息"]["销项发票累计采集时长(月)"]["value"] = total_invoice_months

    logging.info("{}".format(logcode(33, total_invoice_months)))

    valid_latest_12 = DataQualityManager.selectByTimeRange(df=df1_valid, time_column='年月',
                                                           observeTime=observeTime,
                                                           to_observe_time_period=pd.Timedelta(365, unit='D'))

    valid_latest_24 = DataQualityManager.selectByTimeRange(df=df1_valid, time_column='年月',
                                                           observeTime=observeTime,
                                                           to_observe_time_period=pd.Timedelta(365 * 2, unit='D'))

    valid_latest_6 = DataQualityManager.selectByTimeRange(df=df1_valid, time_column='开票日期',
                                                          observeTime=observeTime,
                                                          to_observe_time_period=pd.DateOffset(months=6))

    valid_invoice_frequency_latest_12 = len(valid_latest_12)
    settings.R_CONTAINER["企业基础信息"]["近1年销项开票张数"]["value"] = valid_invoice_frequency_latest_12
    valid_invoice_frequency_latest_24 = len(valid_latest_24)
    if total_invoice_months >= 24:
        invoice_frequency_YoY_growth = np.round((valid_invoice_frequency_latest_12 - valid_invoice_frequency_latest_24 + valid_invoice_frequency_latest_12) / (valid_invoice_frequency_latest_24 - valid_invoice_frequency_latest_12), 4)
    else: invoice_frequency_YoY_growth = np.nan
    settings.R_CONTAINER["准入规则"]["E-成长性-1"]["企业实际表现"]["value"] = invoice_frequency_YoY_growth

    foreign_denominated_invoice_ratio = np.round(valid_latest_12[valid_latest_12["是否美元发票"]==True]["合计金额"].sum() / (valid_latest_12["合计金额"].sum() + 0.01), 4)
    settings.R_CONTAINER["信用评分"]["S-业务画像-1"]["企业实际表现"]["value"] = foreign_denominated_invoice_ratio
    special_invoice_ratio = np.round(valid_latest_12[valid_latest_12["是否专用发票"]]["合计金额"].sum() / (valid_latest_12["合计金额"].sum() + 0.01), 4)
    settings.R_CONTAINER["信用评分"]["S-业务画像-2"]["企业实际表现"]["value"] = 1.0 - special_invoice_ratio
    valid_latest_12_total = valid_latest_12['合计金额'].sum()
    settings.R_CONTAINER["企业基础信息"]["近1年销项开票规模(元)"]["value"] = valid_latest_12_total

    if save_source_data:
        logging.debug(logcode(34))
        valid_latest_12.to_csv(os.path.join(client_outpath, "近1年有效开票数据.csv"), encoding="utf-8-sig")

    valid_latest_12_with_specified_type = DataQualityManager.selectByTimeRange(
        df=df1_valid[df1_valid['发票类型'].isin(['s', '004'])], time_column='年月',
        observeTime=observeTime,
        to_observe_time_period=pd.Timedelta(365, unit='D'))

    logging.info("{}".format(logcode(35, valid_latest_12['合计金额'].sum())))

    logging.info(logcode(36))
    valid_latest_3 = DataQualityManager.selectByTimeRange(df=df1_valid, time_column='年月',
                                                          observeTime=observeTime,
                                                          to_observe_time_period=pd.DateOffset(months=3))

    valid_latest_3_total = valid_latest_3['合计金额'].sum()

    latest_3_percent = round(valid_latest_3_total / (valid_latest_12_total + 0.1), 4)
    settings.R_CONTAINER["信用评分"]["S-买方集中度-1"]["企业实际表现"]["value"] = latest_3_percent
    df1_valid['开票日期_str']=df1_valid['开票日期'].dt.strftime("%Y%m%d")
    tax_df = df1_valid.groupby(['销方名称', '销方税号'], as_index=False).agg({"开票日期_str": ["min", 'max']})
    tax_df.columns = tax_df.columns.map(''.join)
    address_df = df1_valid.groupby(['销方开票地址'], as_index=False).agg({"开票日期_str": ["min", 'max']})
    address_df.columns = address_df.columns.map(''.join)

    settings.R_CONTAINER["开票元信息"]["企业历史开票名称/税号信息"]['value'] = tax_df.iloc[:2]
    settings.R_CONTAINER["开票元信息"]["企业历史开票地址信息"]['value'] = address_df.iloc[:3]
    bank_hist = df1_valid.groupby(['销方名称', '销方银行账号'], as_index=False).agg({"开票日期_str": ["max", 'min']})
    bank_hist.columns = bank_hist.columns.map(''.join)
    bank_hist.rename(columns={'开票日期_strmax':'最近开票时间','开票日期_strmin':'最早开票时间'},inplace=True)
    bank_hist["开户银行"] = bank_hist["销方银行账号"].apply(lambda x: DataQualityManager.extract_bank_and_no(x)[0])
    bank_hist["银行账号"] = bank_hist["销方银行账号"].apply(lambda x: DataQualityManager.extract_bank_and_no(x)[1])
    bank_hist = bank_hist.drop_duplicates(subset=["银行账号"])
    bank_hist = bank_hist[['销方名称','开户银行', '银行账号', '最早开票时间','最近开票时间']]
    settings.R_CONTAINER["开票元信息"]["企业历史开票银行账户信息"]['value'] = bank_hist.iloc[:3]

    newest_address = None
    tel_phone = None
    bank = None
    bank_no = None
    try:
        # address_city, address_city_code = DataQualityManager.extract_city(valid_latest_12['销方开票地址'].mode().iloc[0], settings.HIERARCHICAL_LOCATIONS_DICT)
        newest_address = valid_latest_12['销方开票地址'].mode().iloc[0]
        tel_phone = DataQualityManager.extract_phone_no(newest_address)
        bank_address = valid_latest_12['销方银行账号'].mode().iloc[0]
        bank, bank_no = DataQualityManager.extract_bank_and_no(bank_address)
    except Exception:
        logging.info("提取城市或银行账号类信息出错，将设定为空。")

    settings.R_CONTAINER["企业基础信息"]["最新企业开票地址"]["value"] = newest_address
    settings.R_CONTAINER["企业基础信息"]["企业联系方式"]["value"] = tel_phone
    settings.R_CONTAINER["企业基础信息"]["账户开户银行"]["value"] = bank
    settings.R_CONTAINER["企业基础信息"]["票面银行账户"]["value"] = bank_no

    recent_N_business_local_percent = {}
    recent_N_buyer_province_change_rate = {}
    recent_N_business_province = {}

    NS = [6, 12]
    logging.info(logcode(37))
    for N in NS:
        recent_N_business_local_percent[N], recent_N_business_province[N], recent_N_buyer_province_change_rate[
            N] = DataQualityManager.calculateLocalSalesPercent(df1_valid, observeTime, N, seller_province)

    valid_latest_12_sum_by_yearMonth = DataQualityManager.groupByMultipleColumns(df=valid_latest_12,
                                                                                 sorted_columns_list=['年月'],
                                                                                 as_index=False,
                                                                                 aggregate_method='sum',
                                                                                 end=observeTime - pd.Timedelta(
                                                                                     28, 'D'),
                                                                                 start=observeTime - pd.Timedelta(
                                                                                     365 * 1 - 1, 'D'))
    logging.debug(valid_latest_12_sum_by_yearMonth[["合计金额", "年月"]])
    if not valid_latest_12_sum_by_yearMonth.empty:
        top_month = valid_latest_12_sum_by_yearMonth.loc[
            valid_latest_12_sum_by_yearMonth[["合计金额"]].idxmax(axis=0)["合计金额"], "年月"]
        bottom_month = valid_latest_12_sum_by_yearMonth.loc[
            valid_latest_12_sum_by_yearMonth[["合计金额"]].idxmin(axis=0)["合计金额"], "年月"]
    else:
        top_month = None
        bottom_month = None

    start36_ = np.maximum(observeTime - pd.Timedelta(365 * 3, 'D'), first_invoice_date)
    df1_valid_36 = DataQualityManager.selectByTimeRange(df=df1_valid,
                                                        time_column='年月',
                                                        observeTime=observeTime,
                                                        to_observe_time_period=pd.Timedelta(365 * 3, unit='D'))

    df1_valid_36_sum_by_yearMonth = DataQualityManager.groupByMultipleColumns(df1_valid_36,
                                                                              sorted_columns_list=['年月'],
                                                                              as_index=False,
                                                                              aggregate_method='sum',
                                                                              start=start36_,
                                                                              end=observeTime - pd.Timedelta(28,
                                                                                                             'D'))

    cumulative_months_of_invoice_last_36_months = (np.round(df1_valid_36_sum_by_yearMonth['合计金额'], 4) > 0).sum()
    settings.R_CONTAINER["准入规则"]["E-连续性-2"]["企业实际表现"]["value"] = cumulative_months_of_invoice_last_36_months

    logging.info(logcode(38))
    dynamic_white_buyer_threshholds = eval('settings.' + setting['动态白名单配置'])
    class1_threshold = dynamic_white_buyer_threshholds[1]['阈值']
    class2_threshold = dynamic_white_buyer_threshholds[2]['阈值']
    class3_threshold = dynamic_white_buyer_threshholds[3]['阈值']
    class4_threshold = dynamic_white_buyer_threshholds[4]['阈值']
    class5_threshold = dynamic_white_buyer_threshholds[5]['阈值']
    logging.debug("{}：{}".format(logcode(39),dynamic_white_buyer_threshholds))

    df1_valid_12_by_yearMonth_buyer, buyers_class1, \
    buyers_class2, buyers_class3, buyers_class4, \
    buyers_class5, buyers_invoice_amounts_latest_12, \
    necessary_80_percent_buyer_num_latest_12 = DataQualityManager.classifyBuyers(valid_df=df1_valid,
                                                                                 observeTime=observeTime,
                                                                                 to_observe_window=pd.Timedelta(
                                                                                     365, unit='D'),
                                                                                 class1_threshold=class1_threshold,
                                                                                 class2_threshold=class2_threshold,
                                                                                 class3_threshold=class3_threshold,
                                                                                 class4_threshold=class4_threshold,
                                                                                 class5_threshold=class5_threshold,
                                                                                 agg_top_month=top_month,
                                                                                 agg_bottom_month=bottom_month)

    logging.debug('{}: {}'.format(logcode(40,1),buyers_class1))
    logging.debug('{}: {}'.format(logcode(40,2),buyers_class2))
    logging.debug('{}: {}'.format(logcode(40,3),buyers_class3))

    buyer_class1_dict = dict((name, 1) for name in buyers_class1)
    buyer_class2_dict = dict((name, 2) for name in buyers_class2)
    buyer_class3_dict = dict((name, 3) for name in buyers_class3)
    buyer_class4_dict = dict((name, 4) for name in buyers_class4)
    buyer_class5_dict = dict((name, 5) for name in buyers_class5)
    buyer_class1_dict.update(buyer_class2_dict)
    buyer_class1_dict.update(buyer_class3_dict)
    buyer_class1_dict.update(buyer_class4_dict)
    buyer_class1_dict.update(buyer_class5_dict)

    df1 = DataQualityManager.tagBuyers(to_tag_df=df1, test_dict=buyer_class1_dict, tag_name='爱蜂白名单等级')

    buyers_invoice_amounts_latest_12 = buyers_invoice_amounts_latest_12.sort_values(by='合计金额', ascending=False)
    buyers_invoice_profile = buyers_invoice_amounts_latest_12.copy()

    df1_valid_latest_6_months = DataQualityManager.selectByTimeRange(df1_valid,
                                                                     time_column='年月',
                                                                     observeTime=observeTime,
                                                                     to_observe_time_period=pd.DateOffset(months=6))

    df1_valid_latest_6_months_by_yearmonth_buyers = DataQualityManager.groupByMultipleColumns(
        df=df1_valid_latest_6_months,
        sorted_columns_list=['统一名称', '年月'],
        aggregate_method='sum',
        as_index=False)
    if save_source_data:
        df1_valid_latest_6_months_by_yearmonth_buyers.to_csv(
            os.path.join(client_outpath,"近6月买方合作开票情况{}.csv".format(supplier_name)),encoding='utf_8_sig')

    df1_valid_latest_6_months_by_yearmonth_buyers = df1_valid_latest_6_months_by_yearmonth_buyers[
        np.round(df1_valid_latest_6_months_by_yearmonth_buyers['合计金额'], 8) > 0]
    monthly_buyers_latest_6 = DataQualityManager.groupByMultipleColumns(
        df=df1_valid_latest_6_months_by_yearmonth_buyers,
        sorted_columns_list=['年月'],
        aggregate_method='size')

    if not monthly_buyers_latest_6.empty:
        buyers_avg_num_latest_6 = int(monthly_buyers_latest_6.mean())
    else:
        buyers_avg_num_latest_6 = np.nan
    logging.info('{}：{}'.format(logcode(41),buyers_avg_num_latest_6))

    buyers_class1_size = len(buyers_class1)
    if total_invoice_months < class1_threshold:
        buyers_class1_size = np.nan
    buyers_class2_size = len(buyers_class2)
    if total_invoice_months < class2_threshold:
        buyers_class2_size = np.nan
    buyers_class3_size = len(buyers_class3)
    if total_invoice_months < class3_threshold:
        buyers_class3_size = np.nan
    buyers_class4_size = len(buyers_class4)
    if total_invoice_months < class4_threshold:
        buyers_class4_size = np.nan
    buyers_class5_size = len(buyers_class5)
    if total_invoice_months < class5_threshold:
        buyers_class5_size = np.nan
    logging.info(logcode(42))
    logging.info(logcode(74))

    df1_valid_last_12_by_yearMonth_buyer, buyers_class1_last, buyers_class2_last, \
    buyers_class3_last, buyers_class4_last, buyers_class5_last, \
    buyers_invoice_amounts_last_12, necessary_80_percent_buyer_num_last_12 = DataQualityManager.classifyBuyers(
        valid_df=df1_valid,
        observeTime=observeTime - pd.Timedelta(365, unit='D'),
        to_observe_window=pd.Timedelta(365, unit='D'),
        class1_threshold=class1_threshold,
        class2_threshold=class2_threshold,
        class3_threshold=class3_threshold,
        class4_threshold=class4_threshold,
        class5_threshold=class5_threshold
    )

    logging.debug("{}：{}".format(logcode(43,1), buyers_class1_last))
    logging.debug("{}：{}".format(logcode(43,2), buyers_class2_last))
    logging.debug("{}：{}".format(logcode(43,3),buyers_class3_last))

    # 记录近12月1，2级白名单相对远12月1，2级白名单的变异系数
    white_buyer_1_and_2_this_year = buyers_class1 + buyers_class2
    white_buyer_1_and_2_last_year = buyers_class1_last + buyers_class2_last
    white_buyers_sim_measure_1_and2 = DataQualityManager.pairNamesListsSimiliarity(white_buyer_1_and_2_this_year,
                                                                                   white_buyer_1_and_2_last_year)

    if total_invoice_months < (12 + class1_threshold):
        white_buyers_sim_measure_1_and2 = np.nan

    white_buyer_1_and_2_and_3_this_year = white_buyer_1_and_2_this_year + buyers_class3
    white_buyer_1_and_2_and_3_last_year = white_buyer_1_and_2_last_year + buyers_class3_last
    white_buyers_sim_measure_1_and2_and3 = DataQualityManager.pairNamesListsSimiliarity(
        white_buyer_1_and_2_and_3_this_year,
        white_buyer_1_and_2_and_3_last_year)
    if total_invoice_months < (12 + class1_threshold):
        white_buyers_sim_measure_1_and2_and3 = np.nan

    buyers_class1_last_size = len(buyers_class1_last)
    if total_invoice_months < (12 + class1_threshold):
        buyers_class1_last_size = np.nan
    buyers_class2_last_size = len(buyers_class2_last)
    if total_invoice_months < (12 + class2_threshold):
        buyers_class2_last_size = np.nan
    buyers_class3_last_size = len(buyers_class3_last)
    if total_invoice_months < (12 + class3_threshold):
        buyers_class3_last_size = np.nan
    buyers_class4_last_size = len(buyers_class4_last)
    if total_invoice_months < (12 + class4_threshold):
        buyers_class4_last_size = np.nan
    buyers_class5_last_size = len(buyers_class5_last)
    if total_invoice_months < (12 + class5_threshold):
        buyers_class5_last_size = np.nan

    settings.R_CONTAINER["信用评分"]["S-销售成长性-2"]["企业实际表现"]["value"] = buyers_class1_size - buyers_class1_last_size

    month_freq_N = [12, 9, 6, 3, 1]
    YoY_total_growth = {}
    YoY_mean_growth = {}
    MOM_total_growth = {}
    MOM_mean_growth = {}

    for N in month_freq_N:
        this_year_N = DataQualityManager.selectByTimeRange(df=df1_valid_sum_by_yearMonth,
                                                           observeTime=observeTime,
                                                           time_column='年月',
                                                           to_observe_time_period=pd.Timedelta(30 * N, unit='D'))['合计金额']
        this_year_N_sum = this_year_N.sum()
        if total_invoice_months < N:
            this_year_N_sum = np.nan
        this_year_N_mean = this_year_N.mean()
        if pd.isna(this_year_N_mean) and total_invoice_months >= N:
            this_year_N_mean = 0.0

        last_year_N = DataQualityManager.selectByTimeRange(df=df1_valid_sum_by_yearMonth,
                                                           observeTime=observeTime - pd.Timedelta(365, unit='D'),
                                                           time_column='年月',
                                                           to_observe_time_period=pd.Timedelta(30 * N, unit='D'))[
            '合计金额']
        last_year_N_sum = last_year_N.sum()
        if total_invoice_months < 12 + N:
            last_year_N_sum = np.nan
        last_year_N_mean = last_year_N.mean()
        if pd.isna(last_year_N_mean) and total_invoice_months >= 12 + N:
            last_year_N_mean = 0.0

        YoY_total_growth[N] = DataQualityManager.ceilTheOutlier(
            np.round((this_year_N_sum - last_year_N_sum) / (last_year_N_sum + 0.0001), 4), lower_bound=-100,
            upper_bound=100)
        YoY_mean_growth[N] = DataQualityManager.ceilTheOutlier(
            np.round((this_year_N_mean - last_year_N_mean) / (last_year_N_mean + 0.0001), 4), lower_bound=-100,
            upper_bound=100)
        logging.debug("近{}个月开票总额同比去年增长率为：{}, 开票月均额上同比增长率为:{}".format(N, YoY_total_growth[N], YoY_mean_growth[N]))

        this_year_last_N = DataQualityManager.selectByTimeRange(df=df1_valid_sum_by_yearMonth,
                                                                observeTime=observeTime - pd.Timedelta(30 * N,
                                                                                                       unit='D'),
                                                                time_column='年月',
                                                                to_observe_time_period=pd.Timedelta(30 * N,
                                                                                                    unit='D'))[
            '合计金额']
        this_year_last_N_sum = this_year_last_N.sum()
        if total_invoice_months < (N + N):
            this_year_last_N_sum = np.nan
        this_year_last_N_mean = this_year_last_N.mean()
        if pd.isna(this_year_last_N_mean) and total_invoice_months >= (N + N):
            this_year_last_N_mean = 0.0

        MOM_total_growth[N] = DataQualityManager.ceilTheOutlier(
            np.round((this_year_N_sum - this_year_last_N_sum) / (this_year_last_N_sum + 0.0001), 4),
            lower_bound=-100, upper_bound=100)
        MOM_mean_growth[N] = DataQualityManager.ceilTheOutlier(
            np.round((this_year_N_mean - this_year_last_N_mean) / (this_year_last_N_mean + 0.0001), 4),
            lower_bound=-100, upper_bound=100)

    settings.R_CONTAINER["信用评分"]["S-销售成长性-4"]["企业实际表现"]["value"] = MOM_mean_growth[6]

    logging.info(logcode(44))
    this_year_N_sum_by_buyer_sum = {}

    buyers_class_1_and_2 = buyers_class1 + buyers_class2
    buyers_class_1_and_2_and_3 = buyers_class1 + buyers_class2 + buyers_class3

    dynamic_ents_df = \
        DataQualityManager.groupByMultipleColumns(df1_valid_36[df1_valid_36['统一名称'].isin(buyers_class_1_and_2_and_3)],
                                                  sorted_columns_list=['年月'], as_index=False, aggregate_method='sum',
                                                  start=start36_, end=observeTime - pd.Timedelta(28, 'D'),
                                                  invoice_num=True)

    this_year_N_white_buyers_1_and_2_sum = {}
    this_year_N_white_buyers_1_and_2_and_3_sum = {}
    last_year_N_white_buyers_1_and_2_sum = {}
    last_year_N_white_buyers_1_and_2_and_3_sum = {}

    this_year_N_white_buyers_1_and_2_mean = {}
    this_year_N_white_buyers_1_and_2_and_3_mean = {}

    this_year_N_white_buyers_1_and_2_to_total_ratio = {}
    this_year_N_white_buyers_1_and_2_and_3_to_total_ratio = {}

    this_year_N_white_buyers_1_and_2_YoY_growth = {}
    this_year_N_white_buyers_1_and_2_and_3_YoY_growth = {}
    this_year_N_white_buyers_1_and_2_and_3_period_growth = {}
    last_N_months_white_buyers_1_and_2_and_3_sum = {}
    this_year_N_white_buyers_1_and_2_and_3_mean_deviation_ratio = {}

    for N in month_freq_N:
        this_year_N_by_yearMonth = DataQualityManager.selectByTimeRange(df=df1_valid_12_by_yearMonth_buyer,
                                                                        time_column='年月',
                                                                        observeTime=observeTime,
                                                                        to_observe_time_period=pd.Timedelta(30 * N,
                                                                                                            unit='D'))
        this_year_N_sum_by_buyer = DataQualityManager.groupByMultipleColumns(df=this_year_N_by_yearMonth,
                                                                             sorted_columns_list=['统一名称'],
                                                                             as_index=False,
                                                                             aggregate_method='sum')
        this_year_N_sum_by_buyer_sum[N] = this_year_N_sum_by_buyer["合计金额"].sum()

        this_year_N_white_buyers_1_and_2 = this_year_N_sum_by_buyer[
            this_year_N_sum_by_buyer['统一名称'].isin(buyers_class_1_and_2)]
        this_year_N_white_buyers_1_and_2_and_3 = this_year_N_sum_by_buyer[
            this_year_N_sum_by_buyer['统一名称'].isin(buyers_class_1_and_2_and_3)]

        this_year_N_white_buyers_1_and_2_sum[N] = this_year_N_white_buyers_1_and_2["合计金额"].sum()
        if total_invoice_months < np.maximum(N, class1_threshold):
            this_year_N_white_buyers_1_and_2_sum[N] = np.nan

        this_year_N_white_buyers_1_and_2_mean[N] = this_year_N_white_buyers_1_and_2["合计金额"].replace(
            to_replace=np.nan, value=0).mean()
        if pd.isna(this_year_N_white_buyers_1_and_2_mean[N]) and total_invoice_months >= class1_threshold:
            this_year_N_white_buyers_1_and_2_mean[N] = 0.0

        this_year_N_white_buyers_1_and_2_and_3_sum[N] = this_year_N_white_buyers_1_and_2_and_3["合计金额"].sum()
        if total_invoice_months < np.maximum(N, class1_threshold):
            this_year_N_white_buyers_1_and_2_and_3_sum[N] = np.nan

        this_year_N_white_buyers_1_and_2_and_3_mean[N] = this_year_N_white_buyers_1_and_2_and_3["合计金额"].replace(
            to_replace=np.nan, value=0).mean()
        if pd.isna(this_year_N_white_buyers_1_and_2_and_3_mean[N]) and total_invoice_months >= class1_threshold:
            this_year_N_white_buyers_1_and_2_and_3_mean[N] = 0.0

        this_year_N_white_buyers_1_2_3 = this_year_N_by_yearMonth[this_year_N_by_yearMonth['统一名称'].isin(buyers_class_1_and_2_and_3)]

        if len(this_year_N_white_buyers_1_and_2_and_3) > 0:
            this_year_N_sum_by_buyer_monthly = DataQualityManager.groupByMultipleColumns(
                df=this_year_N_white_buyers_1_2_3, sorted_columns_list=['年月'], as_index=False, aggregate_method='sum')

            this_year_N_months_white_buyers_complete = DataQualityManager.completeMonths(
                this_year_N_sum_by_buyer_monthly,
                set_index='年月', start=observeTime - pd.Timedelta(30 * N, unit='D'),
                end=observeTime - pd.Timedelta(28, 'D'))

            white_buyers_monthly_mean = 0
            if len(this_year_N_months_white_buyers_complete) > 0:
                white_buyers_monthly_mean = this_year_N_months_white_buyers_complete['合计金额'].mean()

            this_year_N_white_buyers_1_and_2_and_3_deviation = np.abs(
                this_year_N_months_white_buyers_complete['合计金额'] - white_buyers_monthly_mean).mean()

            this_year_N_white_buyers_1_and_2_and_3_mean_deviation_ratio[N] = np.round(
                this_year_N_white_buyers_1_and_2_and_3_deviation / (white_buyers_monthly_mean + 1.0), 4)

        else:
            this_year_N_white_buyers_1_and_2_and_3_mean_deviation_ratio[N] = np.nan

        logging.info("{}：{}".format(logcode(45,N), this_year_N_white_buyers_1_and_2_sum[N]))
        logging.info("{}：{}".format(logcode(46,N), this_year_N_white_buyers_1_and_2_and_3_sum[N]))

        this_year_N_white_buyers_1_and_2_to_total_ratio[N] = np.round(
            this_year_N_white_buyers_1_and_2_sum[N] / (this_year_N_sum_by_buyer_sum[N] + 0.0001), 4)
        logging.info("{}：{}".format(logcode(47, N), this_year_N_white_buyers_1_and_2_to_total_ratio[N]))
        this_year_N_white_buyers_1_and_2_and_3_to_total_ratio[N] = np.round(
            this_year_N_white_buyers_1_and_2_and_3_sum[N] / (this_year_N_sum_by_buyer_sum[N] + 0.0001), 4)
        logging.info("{}：{}".format(logcode(48, N),this_year_N_white_buyers_1_and_2_and_3_to_total_ratio[N]))

        logging.info("{}：{}".format(logcode(49, N), this_year_N_sum_by_buyer_sum[N]))

        last_year_N_by_yearMonth = DataQualityManager.selectByTimeRange(df=df1_valid_last_12_by_yearMonth_buyer,
                                                                        time_column='年月',
                                                                        observeTime=observeTime - pd.Timedelta(365,
                                                                                                               unit='D'),
                                                                        to_observe_time_period=pd.Timedelta(30 * N,
                                                                                                            unit='D'))

        last_year_N_sum_by_buyer = DataQualityManager.groupByMultipleColumns(df=last_year_N_by_yearMonth,
                                                                             sorted_columns_list=['统一名称'],
                                                                             as_index=False,
                                                                             aggregate_method='sum')

        buyers_class_1_and_2_last = buyers_class1_last + buyers_class2_last
        buyers_class_1_and_2_and_3_last = buyers_class_1_and_2_last + buyers_class3_last

        last_year_N_white_buyers_1_and_2 = last_year_N_sum_by_buyer[
            last_year_N_sum_by_buyer['统一名称'].isin(buyers_class_1_and_2_last)]
        last_year_N_white_buyers_1_and_2_and_3 = last_year_N_sum_by_buyer[
            last_year_N_sum_by_buyer['统一名称'].isin(buyers_class_1_and_2_and_3_last)]

        last_year_N_white_buyers_1_and_2_sum[N] = last_year_N_white_buyers_1_and_2["合计金额"].sum()
        if total_invoice_months < np.maximum(12 + N, 12 + class1_threshold):
            last_year_N_white_buyers_1_and_2_sum[N] = np.nan

        last_year_N_white_buyers_1_and_2_and_3_sum[N] = last_year_N_white_buyers_1_and_2_and_3["合计金额"].sum()
        if total_invoice_months < np.maximum(12 + N, 12 + class1_threshold):
            last_year_N_white_buyers_1_and_2_and_3_sum[N] = np.nan

        logging.info("{}：{}".format(logcode(50,N), last_year_N_white_buyers_1_and_2_sum[N]))
        logging.info("{}：{}".format(logcode(51,N), last_year_N_white_buyers_1_and_2_and_3_sum[N]))

        this_year_N_white_buyers_1_and_2_YoY_growth[N] = DataQualityManager.ceilTheOutlier(np.round(
            (this_year_N_white_buyers_1_and_2_sum[N] - last_year_N_white_buyers_1_and_2_sum[N]) / (
                    last_year_N_white_buyers_1_and_2_sum[N] + 0.0001), 4), lower_bound=-100, upper_bound=100)
        logging.info("{}：{}".format(logcode(52,N), this_year_N_white_buyers_1_and_2_YoY_growth[N]))
        this_year_N_white_buyers_1_and_2_and_3_YoY_growth[N] = DataQualityManager.ceilTheOutlier(np.round(
            (this_year_N_white_buyers_1_and_2_and_3_sum[N] - last_year_N_white_buyers_1_and_2_and_3_sum[N]) / (
                    last_year_N_white_buyers_1_and_2_and_3_sum[N] + 0.0001), 4), lower_bound=-100,
            upper_bound=100)
        logging.info("{}：{}".format(logcode(53,N), this_year_N_white_buyers_1_and_2_and_3_YoY_growth[N]))

        df1_valid_last_N_months_by_yearMonth_buyer, buyers_class1_last_N_months, buyers_class2_last_N_months, \
        buyers_class3_last_N_months, buyers_class4_last_N_months, buyers_class5_last_N_months, \
        buyers_invoice_amounts_last_N_months, necessary_80_percent_buyer_num_last_N_months = DataQualityManager.classifyBuyers(
            valid_df=df1_valid,
            observeTime=observeTime - pd.DateOffset(months=N),
            to_observe_window=pd.DateOffset(months=12),
            class1_threshold=class1_threshold,
            class2_threshold=class2_threshold,
            class3_threshold=class3_threshold,
            class4_threshold=class4_threshold,
            class5_threshold=class5_threshold
        )

        logging.debug("{}：{}".format(logcode(55,1),buyers_class1_last_N_months))
        logging.debug("{}：{}".format(logcode(55,2),buyers_class2_last_N_months))
        logging.debug("{}：{}".format(logcode(55,3),buyers_class3_last_N_months))

        last_N_months_by_yearMonth = DataQualityManager.selectByTimeRange(
            df=df1_valid_last_N_months_by_yearMonth_buyer,
            time_column='年月',
            observeTime=observeTime - pd.DateOffset(
                months=N),
            to_observe_time_period=pd.DateOffset(
                months=
                N))

        last_N_months_sum_by_buyer = DataQualityManager.groupByMultipleColumns(df=last_N_months_by_yearMonth,
                                                                               sorted_columns_list=['统一名称'],
                                                                               as_index=False,
                                                                               aggregate_method='sum')

        last_N_months_white_buyers_1_and_2_and_3 = last_N_months_sum_by_buyer[
            last_N_months_sum_by_buyer['统一名称'].isin(this_year_N_white_buyers_1_2_3)]

        last_N_months_white_buyers_1_and_2_and_3_sum[N] = last_N_months_white_buyers_1_and_2_and_3["合计金额"].sum()
        if total_invoice_months < 2 * N + class1_threshold:
            last_N_months_white_buyers_1_and_2_and_3_sum[N] = np.nan

        this_year_N_white_buyers_1_and_2_and_3_period_growth[N] = DataQualityManager.ceilTheOutlier(np.round(
            (this_year_N_white_buyers_1_and_2_and_3_sum[N] - last_N_months_white_buyers_1_and_2_and_3_sum[N]) / (
                    last_N_months_white_buyers_1_and_2_and_3_sum[N] + 1.0), 4), lower_bound=-100,
            upper_bound=100)
        logging.info("{}：{}".format(logcode(56,N), this_year_N_white_buyers_1_and_2_and_3_period_growth[N]))
        logging.info(logcode(54))

    settings.R_CONTAINER["准入规则"]["E-买方结构-1"]["企业实际表现"]["value"] = this_year_N_white_buyers_1_and_2_and_3_to_total_ratio[12]
    settings.R_CONTAINER["信用评分"]["S-买方结构-2"]["企业实际表现"]["value"] = this_year_N_white_buyers_1_and_2_and_3_to_total_ratio[12]

    logging.info(logcode(57))
    df1_valid_12_by_yearMonth_CoreBrands_buyer, df1_by_yearMonth_for_1_2_3, df1_valid_12_by_yearMonth_CoreBrands_1_buyer, \
    df1_valid_12_by_yearMonth_CoreBrands_2_buyer, df1_valid_12_by_yearMonth_CoreBrands_3_buyer, \
    coreBrands_this, coreBrands_1, coreBrands_2, coreBrands_3, coreBrands_4, coreBrands_5 = DataQualityManager.getCoreBrandsBuyers(
        valid_df=df1_valid, observeTime=observeTime, to_observe_window=pd.Timedelta(365, unit='D'))

    logging.debug('{}: {}'.format(logcode(58,1),coreBrands_1))
    logging.debug('{}: {}'.format(logcode(58,2),coreBrands_2))
    logging.debug('{}: {}'.format(logcode(58,3),coreBrands_3))

    df1_valid_12_by_yearMonth_CoreBrands_buyer1 = df1_valid_12_by_yearMonth_CoreBrands_buyer[
        df1_valid_12_by_yearMonth_CoreBrands_buyer['合计金额'] > 0]

    coreBrands_relationship_months = DataQualityManager.groupByMultipleColumns(
        df=df1_valid_12_by_yearMonth_CoreBrands_buyer1, sorted_columns_list=['统一名称'], aggregate_method='size')
    freqent_coreBrands_buyers = list(
        coreBrands_relationship_months[coreBrands_relationship_months.values >= 3].index)
    freqent_coreBrands_buyer_size = len(freqent_coreBrands_buyers)
    logging.info('{}:{}'.format(logcode(59), freqent_coreBrands_buyer_size))

    invoice_months_with_1_2_3_coreBrands = np.ceil(
        (observeTime - df1_by_yearMonth_for_1_2_3['年月'].min()) / np.timedelta64(1, 'M'))
    logging.info("{}:{}".format(logcode(60), invoice_months_with_1_2_3_coreBrands))

    df1_valid_core_brands = df1_valid[np.logical_or(np.logical_or(df1_valid['核企级别'] == '1', df1_valid['核企级别'] == '2'),
                                                    df1_valid['核企级别'] == '3')]

    static_state_core_df = DataQualityManager.groupByMultipleColumns(df=df1_valid_core_brands,sorted_columns_list=['年月'], as_index=False,
                           aggregate_method='sum', end=observeTime - pd.Timedelta(28, 'D'), invoice_num=True)


# if len(df1_valid_core_brands) and len(df1_valid_last_12_months_CoreBrands) > 0:
    invoice_months_coreBrands_with_condition = np.ceil((df1_valid_core_brands['年月'].max() - df1_valid_core_brands['年月'].min()) / np.timedelta64(1, 'M'))

    logging.info("{}:{}".format(logcode(61), invoice_months_coreBrands_with_condition))

    NS = [6, 12]
    coreBrands_mean_deviation_ratio = {}
    core_brands_quarterly_deviation = {}

    for N in NS:
        df1_valid_N_months = DataQualityManager.selectByTimeRange(df=df1_valid,
                                                                  time_column='年月',
                                                                  observeTime=observeTime,
                                                                  to_observe_time_period=pd.DateOffset(months=
                                                                                                       N))

        df1_valid_N_months_CoreBrands = df1_valid_N_months[df1_valid_N_months['是否为核企'] == True]

        if len(df1_valid_N_months_CoreBrands) > 0:
            df1_by_yearMonth_selected = DataQualityManager.groupByMultipleColumns(df1_valid_N_months_CoreBrands,
                                                                                  sorted_columns_list=['年月'],
                                                                                  aggregate_method='sum')
            df1_by_yearMonth_complete = DataQualityManager.completeMonths(df1_by_yearMonth_selected,
                                                                          start=observeTime - pd.DateOffset(
                                                                              months=N),
                                                                          end=observeTime - pd.Timedelta(28,
                                                                                                         unit='D'))
            coreBrands_mean_ = df1_by_yearMonth_complete['合计金额'].mean()
            if pd.isna(coreBrands_mean_):
                coreBrands_mean_ = 0

            df1_by_yearMonth_for_1_2_3_mean_deviations = np.abs(
                df1_by_yearMonth_complete['合计金额'] - coreBrands_mean_).mean()

            coreBrands_mean_deviation_ratio[N] = np.round(
                df1_by_yearMonth_for_1_2_3_mean_deviations / (coreBrands_mean_ + 1.0), 4)

            valid_corebrands_sum_by_month = DataQualityManager.groupByMultipleColumns(
                df=df1_valid_N_months_CoreBrands,
                sorted_columns_list=[
                    '年月'],
                as_index=False,
                aggregate_method='sum',
                end=observeTime - pd.Timedelta(
                    28, 'D'),
                start=observeTime - pd.DateOffset(
                    months=N))

            tlist1 = []
            for i in range(2, len(valid_corebrands_sum_by_month), 3):
                moving_avg_sum = (valid_corebrands_sum_by_month.loc[i - 1, '合计金额'] +
                                  valid_corebrands_sum_by_month.loc[i - 2, '合计金额'] +
                                  + valid_corebrands_sum_by_month.loc[i, '合计金额']) / 3
                tlist1.append(moving_avg_sum)

            mean_ = np.mean(tlist1)

            tlist2 = [np.abs(i - mean_) for i in tlist1]

            mean_deviations = np.mean(tlist2)

            core_brands_quarterly_deviation[N] = np.round(mean_deviations / (mean_ + 0.0001), 4)

        else:
            coreBrands_mean_deviation_ratio[N] = np.nan

            core_brands_quarterly_deviation[N] = np.nan

        logging.info('{} : {}'.format(logcode(62,N),coreBrands_mean_deviation_ratio[N]))

    zero_invoice_monthts_1_2_3_coreBrands_latest_12 = {}
    zero_invoice_monthts_1_2_3_coreBrands_last_12 = {}
    zero_freq = [2, 3, 6]
    for N in zero_freq:
        df1_valid_sum_by_yearMonth_CoreBrands_1_2_3 = DataQualityManager.selectByTimeRange(
            df=df1_by_yearMonth_for_1_2_3,
            time_column='年月',
            observeTime=observeTime,
            to_observe_time_period=pd.Timedelta(30 * N, unit='D'))
        zero_invoice_monthts_1_2_3_coreBrands_latest_12[N] = N - df1_valid_sum_by_yearMonth_CoreBrands_1_2_3[
            df1_valid_sum_by_yearMonth_CoreBrands_1_2_3['合计金额'] > 0].shape[0]
        if total_invoice_months < N:
            zero_invoice_monthts_1_2_3_coreBrands_latest_12[N] = np.nan

        df1_valid_sum_by_yearMonth_CoreBrands_1_2_3_last = DataQualityManager.selectByTimeRange(
            df=df1_by_yearMonth_for_1_2_3,
            time_column='年月',
            observeTime=observeTime - pd.Timedelta(365, unit='D'),
            to_observe_time_period=pd.Timedelta(30 * N, unit='D'))

        zero_invoice_monthts_1_2_3_coreBrands_last_12[N] = N - df1_valid_sum_by_yearMonth_CoreBrands_1_2_3_last[
            df1_valid_sum_by_yearMonth_CoreBrands_1_2_3_last['合计金额'] > 0].shape[0]
        if total_invoice_months < 12 + N:
            zero_invoice_monthts_1_2_3_coreBrands_last_12[N] = np.nan
    logging.info('{}：{}'.format(logcode(63),zero_invoice_monthts_1_2_3_coreBrands_latest_12))
    logging.info('{}：{}'.format(logcode(64),zero_invoice_monthts_1_2_3_coreBrands_last_12))

    if len(thisCoreBrands) == 0:
        zero_invoice_monthts_1_2_3_coreBrands_latest_12[2] = np.nan
        zero_invoice_monthts_1_2_3_coreBrands_latest_12[3] = np.nan
        zero_invoice_monthts_1_2_3_coreBrands_latest_12[6] = np.nan
        zero_invoice_monthts_1_2_3_coreBrands_last_12[2] = np.nan
        zero_invoice_monthts_1_2_3_coreBrands_last_12[3] = np.nan
        zero_invoice_monthts_1_2_3_coreBrands_last_12[6] = np.nan
        invoice_months_with_1_2_3_coreBrands = 0
        coreBrands_mean_deviation_ratio[6] = np.nan
        coreBrands_mean_deviation_ratio[12] = np.nan

    buyers_coreBrands1_size = len(coreBrands_1)
    buyers_coreBrands2_size = len(coreBrands_2)
    buyers_coreBrands3_size = len(coreBrands_3)

    df1_valid_last_12_by_yearMonth_CoreBrands_buyer, df1_by_yearMonth_CoreBrands_1_2_3, df1_valid_last_12_by_yearMonth_CoreBrands_1_buyer, \
    df1_valid_last_12_by_yearMonth_CoreBrands_2_buyer, df1_valid_last_12_by_yearMonth_CoreBrands_3_buyer, \
    coreBrands_last, coreBrands_1_last, coreBrands_2_last, coreBrands_3_last, \
    coreBrands_4_last, coreBrands_5_last = DataQualityManager.getCoreBrandsBuyers(
        valid_df=df1_valid,
        observeTime=observeTime - pd.Timedelta(365, unit='D'),
        to_observe_window=pd.Timedelta(365, unit='D'))
    logging.debug('{}: {}'.format(logcode(65,1), len(coreBrands_1_last)))
    logging.debug('{}: {}'.format(logcode(65,2), len(coreBrands_2_last)))
    logging.debug('{}: {}'.format(logcode(65,3), len(coreBrands_3_last)))

    coreBrands_buyers_sim_measure = DataQualityManager.pairNamesListsSimiliarity(coreBrands_this, coreBrands_last)
    if total_invoice_months < (12 + 6):
        coreBrands_buyers_sim_measure = np.nan

    df1_valid_last_12_by_yearMonth_CoreBrands_buyer1 = df1_valid_last_12_by_yearMonth_CoreBrands_buyer[
        df1_valid_last_12_by_yearMonth_CoreBrands_buyer['合计金额'] > 0]
    coreBrands_relationship_months_last = DataQualityManager.groupByMultipleColumns(
        df=df1_valid_last_12_by_yearMonth_CoreBrands_buyer1,
        sorted_columns_list=['统一名称'],
        aggregate_method='size')
    freqent_coreBrands_buyers_last = list(
        coreBrands_relationship_months_last[coreBrands_relationship_months_last.values >= 3].index)
    freqent_coreBrands_buyer_last_size = len(freqent_coreBrands_buyers_last)
    logging.info('{}:{}'.format(logcode(66), freqent_coreBrands_buyer_last_size))

    buyers_coreBrands1_size_last = len(coreBrands_1_last)
    if total_invoice_months < (12 + 6):
        buyers_coreBrands1_size_last = np.nan
    buyers_coreBrands2_size_last = len(coreBrands_2_last)
    if total_invoice_months < (12 + 6):
        buyers_coreBrands2_size_last = np.nan
    buyers_coreBrands3_size_last = len(coreBrands_3_last)
    if total_invoice_months < (12 + 6):
        buyers_coreBrands3_size_last = np.nan

    logging.info(logcode(67))

    this_year_N_coreBrands_buyers_sum = {}
    last_year_N_coreBrands_buyers_sum = {}

    this_year_N_coreBrands_buyers_to_total_ratio = {}
    this_year_N_coreBrands_buyers_YoY_growth = {}

    this_year_N_coreBrands_buyers_period_growth = {}
    last_N_months_coreBrands_buyers_sum = {}

    for N in month_freq_N:
        this_year_N_by_yearMonth_coreBrands = DataQualityManager.selectByTimeRange(
            df=df1_valid_12_by_yearMonth_CoreBrands_buyer,
            time_column='年月',
            observeTime=observeTime,
            to_observe_time_period=pd.Timedelta(30 * N, unit='D'))
        this_year_N_sum_by_coreBrands_buyer = DataQualityManager.groupByMultipleColumns(
            df=this_year_N_by_yearMonth_coreBrands,
            sorted_columns_list=['统一名称'],
            as_index=False,
            aggregate_method='sum')

        this_year_N_coreBrands_buyers_sum[N] = this_year_N_sum_by_coreBrands_buyer["合计金额"].sum()
        if total_invoice_months < N:
            this_year_N_coreBrands_buyers_sum[N] = np.nan
        logging.info("{}：{}".format(logcode(68,N), this_year_N_coreBrands_buyers_sum[N]))

        this_year_N_coreBrands_buyers_to_total_ratio[N] = np.round(
            this_year_N_coreBrands_buyers_sum[N] / (this_year_N_sum_by_buyer_sum[N] + 0.0001), 4)
        logging.info("{}：{}".format(logcode(69,N), this_year_N_coreBrands_buyers_to_total_ratio[N]))

        last_year_N_by_yearMonth_coreBrands = DataQualityManager.selectByTimeRange(
            df=df1_valid_last_12_by_yearMonth_CoreBrands_buyer,
            time_column='年月',
            observeTime=observeTime - pd.Timedelta(365, unit='D'),
            to_observe_time_period=pd.Timedelta(30 * N, unit='D'))

        last_year_N_sum_by_coreBrands_buyer = DataQualityManager.groupByMultipleColumns(
            df=last_year_N_by_yearMonth_coreBrands,
            sorted_columns_list=['统一名称'],
            as_index=False,
            aggregate_method='sum')

        last_year_N_coreBrands_buyers_sum[N] = last_year_N_sum_by_coreBrands_buyer["合计金额"].sum()
        if total_invoice_months < 12 + N:
            last_year_N_coreBrands_buyers_sum[N] = np.nan

        logging.info("{}：{}".format(logcode(70,N), last_year_N_coreBrands_buyers_sum[N]))

        this_year_N_coreBrands_buyers_YoY_growth[N] = DataQualityManager.ceilTheOutlier(
            np.round((this_year_N_coreBrands_buyers_sum[N] - last_year_N_coreBrands_buyers_sum[N]) /
                     (last_year_N_coreBrands_buyers_sum[N] + 0.0001), 4), lower_bound=-100, upper_bound=100)
        logging.info("{}：{}".format(logcode(71,N), this_year_N_coreBrands_buyers_YoY_growth[N]))

        df1_valid_last_N_months_by_yearMonth_CoreBrands_buyer, df1_last_N_months_by_yearMonth_CoreBrands_1_2_3, df1_valid_last_N_months_by_yearMonth_CoreBrands_1_buyer, \
        df1_valid_last_N_months_by_yearMonth_CoreBrands_2_buyer, df1_valid_last_N_months_by_yearMonth_CoreBrands_3_buyer, \
        coreBrands_last_N_months, coreBrands_1_last_N_months, coreBrands_2_last_N_months, coreBrands_3_last_N_months, \
        coreBrands_4_last_N_months, coreBrands_5_last_N_months = DataQualityManager.getCoreBrandsBuyers(
            valid_df=df1_valid,
            observeTime=observeTime - pd.DateOffset(months=N),
            to_observe_window=pd.DateOffset(months=N))
        logging.debug('{}: {}'.format(logcode(72,N), coreBrands_1_last_N_months))
        logging.debug('{}: {}'.format(logcode(72,N), coreBrands_2_last_N_months))
        logging.debug('{}: {}'.format(logcode(72,N), coreBrands_3_last_N_months))

        # 记录远12月一、二、三级核心企业白名单买方数量：远12月至少有6个月，此步计算才有较纯的经济信号意义
        buyers_coreBrands1_size_last_N_months = len(coreBrands_1_last_N_months)
        if total_invoice_months < (2 * N):
            buyers_coreBrands1_size_last_N_months = np.nan
        buyers_coreBrands2_size_last_N_months = len(coreBrands_2_last)
        if total_invoice_months < (2 * N):
            buyers_coreBrands2_size_last_N_months = np.nan
        buyers_coreBrands3_size_last_N_months = len(coreBrands_3_last)
        if total_invoice_months < (2 * N):
            buyers_coreBrands3_size_last_N_months = np.nan

        logging.info(logcode(73))

        last_N_months_coreBrands_buyers_sum[N] = df1_valid_last_N_months_by_yearMonth_CoreBrands_buyer["合计金额"].sum()
        this_year_N_coreBrands_buyers_period_growth[N] = DataQualityManager.ceilTheOutlier(
            np.round((this_year_N_coreBrands_buyers_sum[N] - last_N_months_coreBrands_buyers_sum[N]) /
                     (last_N_months_coreBrands_buyers_sum[N] + 0.0001), 4), lower_bound=-100, upper_bound=100)
        logging.info("{}：{}".format(logcode(74,N), this_year_N_coreBrands_buyers_YoY_growth[N]))

    settings.R_CONTAINER["准入规则"]["E-成长性-2"]["企业实际表现"]["value"] = this_year_N_coreBrands_buyers_YoY_growth[12]
    settings.R_CONTAINER["信用评分"]["S-买方结构-1"]["企业实际表现"]["value"] = this_year_N_coreBrands_buyers_to_total_ratio[12]
    settings.R_CONTAINER["信用评分"]["S-销售成长性-3"]["企业实际表现"]["value"] = this_year_N_coreBrands_buyers_YoY_growth[6]

    start24_ = np.maximum(observeTime - pd.Timedelta(365 * 2, 'D'), first_invoice_date)
    valid_latest_12_sum_by_yearMonth = DataQualityManager.groupByMultipleColumns(df=valid_latest_12,
                                                                                 sorted_columns_list=['年月'],
                                                                                 as_index=False,
                                                                                 aggregate_method='sum',
                                                                                 end=observeTime - pd.Timedelta(
                                                                                     28, 'D'),
                                                                                 start=observeTime - pd.DateOffset(
                                                                                     months=12))

    valid_latest_24_sum_by_yearMonth_right = DataQualityManager.groupByMultipleColumns(df=valid_latest_24,
                                                                                       sorted_columns_list=['年月'],
                                                                                       as_index=False,
                                                                                       aggregate_method='sum',
                                                                                       end=observeTime - pd.Timedelta(
                                                                                           28, 'D'),
                                                                                       start=start24_)

    mean_deviation_ratio_N = {}
    mean_deviation_ratio_last_N = {}
    mean_deviation_change_rate_N = {}

    NS = [6, 12]
    for N in NS:
        mean_deviation_ratio_N[N], mean_deviation_ratio_last_N[N], mean_deviation_change_rate_N[
            N] = DataQualityManager.calculateDeviationRate(df1_valid, observeTime, N)

    longest_continuous_zero_invoice_months_latest_12 = DataQualityManager.longestContinuousZerosLen(
        valid_latest_12_sum_by_yearMonth['合计金额'].values)

    latest_12_trend_monthly = DataQualityManager.identify_trend(valid_latest_12_sum_by_yearMonth['合计金额'].tolist())

    logging.info(logcode(78))

    latest_12_months_by_buyer = DataQualityManager.groupByMultipleColumns(df=valid_latest_12,
                                                                          sorted_columns_list=['统一名称'],
                                                                          aggregate_method='sum',
                                                                          as_index=False).sort_values(by='合计金额',
                                                                                                      ascending=False)

    latest_12_months_by_buyer = latest_12_months_by_buyer[latest_12_months_by_buyer['合计金额'] > 0]

    top_5_buyers_df = latest_12_months_by_buyer.head(5)[['统一名称', '合计金额']]

    top_5_buyers_list = top_5_buyers_df['统一名称'].unique().tolist()

    total_amount = latest_12_months_by_buyer['合计金额'].sum()

    latest_12_months_by_buyer['金额占比'] = 0

    if total_amount > 0:
        latest_12_months_by_buyer['金额占比'] = np.round(latest_12_months_by_buyer['合计金额'] / (total_amount + 1.0),
                                                     4)

    top_10_percent_buyers_df = latest_12_months_by_buyer[latest_12_months_by_buyer['金额占比'] >= 0.1]

    top_10_percent_buyers_list = top_10_percent_buyers_df['统一名称'].unique().tolist()

    NS = [3, 6, 12]
    top_5_buyers_trend = {}
    top_10_percent_buyers_trend = {}

    for N in NS:
        if N == 12:
            top_5_buyers_temp = valid_latest_12[valid_latest_12['统一名称'].isin(top_5_buyers_list)]
            top_10_percent_buyers_temp = valid_latest_12[
                valid_latest_12['统一名称'].isin(top_10_percent_buyers_list)]

        elif N == 6:
            top_5_buyers_temp = valid_latest_6[valid_latest_6['统一名称'].isin(top_5_buyers_list)]
            top_10_percent_buyers_temp = valid_latest_6[valid_latest_6['统一名称'].isin(top_10_percent_buyers_list)]

        elif N == 3:
            top_5_buyers_temp = valid_latest_3[valid_latest_3['统一名称'].isin(top_5_buyers_list)]
            top_10_percent_buyers_temp = valid_latest_3[valid_latest_3['统一名称'].isin(top_10_percent_buyers_list)]

        top_5_buyers_by_month = DataQualityManager.groupByMultipleColumns(df=top_5_buyers_temp,
                                                                          sorted_columns_list=['年月'],
                                                                          aggregate_method='sum',
                                                                          as_index=False).sort_values(by='年月',
                                                                                                      ascending=True)

        top_10_percent_buyers_by_month = DataQualityManager.groupByMultipleColumns(
            df=top_10_percent_buyers_temp,
            sorted_columns_list=['年月'],
            aggregate_method='sum',
            as_index=False).sort_values(by='年月', ascending=True)

        top_5_buyers_trend[N] = DataQualityManager.identify_trend(top_5_buyers_by_month['合计金额'].tolist())

        top_10_percent_buyers_trend[N] = DataQualityManager.identify_trend(
            top_10_percent_buyers_by_month['合计金额'].tolist())

    longest_continuous_zero_invoice_months_latest_24 = DataQualityManager.longestContinuousZerosLen(valid_latest_24_sum_by_yearMonth_right['合计金额'].values)
    settings.R_CONTAINER["准入规则"]["E-连续性-3"]["企业实际表现"]["value"] = longest_continuous_zero_invoice_months_latest_24
    logging.info('{}: {}'.format(logcode(81), longest_continuous_zero_invoice_months_latest_12))

    latest_N_cancel_rate = {}
    latest_N_redCorrection_rate = {}
    latest_N_cancel_abs_sum = {}
    latest_N_redCorrection_abs_sum = {}
    latest_N_abs_sum = {}

    latest_N_cancel_12months_percent = {}

    latest_N_redCorrection_12months_percent = {}

    latest_N_cancel_12months_compare_rate = {}

    latest_N_redCorrection_12months_compare_rate = {}

    for N in month_freq_N:
        df1_valid_sum = DataQualityManager.selectByTimeRange(df1_valid_source, observeTime=observeTime,
                                                             time_column='年月',
                                                             to_observe_time_period=pd.Timedelta(N * 30, unit='D')
                                                             )["合计金额"].sum()

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

        latest_N_redCorrection_abs_sum[N] = df1_redCorrection_sum

        latest_N_cancel_abs_sum[N] = df1_cancel_absolute_sum

        latest_N_abs_sum[N] = df1_valid_absolute_sum

        latest_N_cancel_rate[N] = np.round(
            df1_cancel_absolute_sum / (np.abs(df1_valid_sum) + df1_cancel_absolute_sum + 0.0001), 4)
        if total_invoice_months < N:
            latest_N_cancel_rate[N] = np.nan
        latest_N_redCorrection_rate[N] = np.round(df1_redCorrection_sum / (df1_valid_absolute_sum + 0.0001), 4)
        if total_invoice_months < N:
            latest_N_redCorrection_rate[N] = np.nan
        logging.info("{}: {}".format(logcode(82,N), latest_N_cancel_rate[N]))
        logging.info("{}: {}".format(logcode(83,N), latest_N_redCorrection_rate[N]))

    settings.R_CONTAINER["准入规则"]["E-稳定性-1"]["企业实际表现"]["value"] = latest_N_cancel_rate[12]
    settings.R_CONTAINER["准入规则"]["E-稳定性-2"]["企业实际表现"]["value"] = latest_N_redCorrection_rate[12]
    settings.R_CONTAINER["信用评分"]["S-销售稳定性-1"]["企业实际表现"]["value"] = latest_N_redCorrection_rate[3]

    NS = [3, 6]

    for N in NS:
        latest_N_cancel_12months_percent[N] = np.round(
            latest_N_cancel_abs_sum[N] / (latest_N_cancel_abs_sum[12] + 1), 4)

        latest_N_redCorrection_12months_percent[N] = np.round(
            latest_N_redCorrection_abs_sum[N] / (latest_N_redCorrection_abs_sum[12] + 1), 4)

        latest_N_cancel_12months_compare_rate[N] = np.round(
            latest_N_cancel_rate[N] / (latest_N_cancel_rate[12] + 0.0001), 4)

        latest_N_redCorrection_12months_compare_rate[N] = np.round(
            latest_N_redCorrection_rate[N] * 1000 / (latest_N_redCorrection_rate[12] + 0.0001), 4)

    zero_freq_N = [12, 6, 3, 2, 1]
    zero_months_count = {}
    for N in zero_freq_N:
        df1_valid_sum_by_yearMonth_period = DataQualityManager.selectByTimeRange(df=df1_valid_sum_by_yearMonth,
                                                                                 time_column='年月',
                                                                                 observeTime=observeTime,
                                                                                 to_observe_time_period=pd.Timedelta(
                                                                                     N * 30, unit='D'))
        zero_months_count[N] = N - df1_valid_sum_by_yearMonth_period[
            df1_valid_sum_by_yearMonth_period['合计金额'] > 0].shape[0]
        if total_invoice_months < N:
            zero_months_count[N] = np.nan

        logging.info("{}:{}".format(logcode(84,N), zero_months_count[N]))

    settings.R_CONTAINER["准入规则"]["E-连续性-1"]["企业实际表现"]["value"] = zero_months_count[3]

    latest_12_big_buyers_rows = DataQualityManager.getBigBuyers(valid_df=df1_valid,
                                                                observeTime=observeTime,
                                                                to_observe_window=pd.Timedelta(365, unit='D'),
                                                                big_threshold=0.05)
    latest_12_big_buyers = latest_12_big_buyers_rows['统一名称'].unique()
    latest_12_big_buyers_size = len(latest_12_big_buyers)
    big_Buyer_yearMonth = df1_valid_12_by_yearMonth_buyer[
        df1_valid_12_by_yearMonth_buyer['统一名称'].isin(latest_12_big_buyers)]

    if save_source_data:
        big_Buyer_yearMonth.to_csv(
            os.path.join(client_outpath, "近12月TOP买方开票情况.csv"), encoding="utf-8-sig")

    average_zero_months = {}
    for N in month_freq_N:
        big_buyer_N_yearMonth = DataQualityManager.selectByTimeRange(df=big_Buyer_yearMonth,
                                                                     observeTime=observeTime,
                                                                     time_column='年月',
                                                                     to_observe_time_period=pd.Timedelta(N * 30,
                                                                                                         unit='D'))
        big_buyer_N_yearMonth_nonZero = big_buyer_N_yearMonth[np.round(big_buyer_N_yearMonth['合计金额'], 8) > 0]

        nonZero_sizes_big_buyer = DataQualityManager.groupByMultipleColumns(df=big_buyer_N_yearMonth_nonZero,
                                                                            sorted_columns_list=['统一名称'],
                                                                            aggregate_method='size')

        logging.info('{} : {}'.format(logcode(85,N), len(nonZero_sizes_big_buyer)))

        if latest_12_big_buyers_size > 0:
            nonZero_size = len(nonZero_sizes_big_buyer.values)
            if nonZero_size > 0:
                average_zero_months[N] = round(((N - np.mean(nonZero_sizes_big_buyer.values)) * nonZero_size +
                                                (
                                                        latest_12_big_buyers_size - nonZero_size) * N) / latest_12_big_buyers_size,
                                               4)
            else:
                average_zero_months[N] = N
        else:
            average_zero_months[N] = N

        logging.info("{}：{}".format(logcode(86,N), average_zero_months[N]))

    top10_buyer_similarity = {}

    top10_buyer_percent_similarity = {}

    giniIndexThis = {}

    churn_rate = {}

    customer_dev_rate = {}

    giniIndexLast = {}

    churn_rate_last = {}

    customer_dev_rate_last = {}

    _churn_rate_ = {}

    _customer_dev_rate_ = {}

    top_buyers_percent_recent_12_months = {}

    top_buyers_percent_recent_6_months = {}

    NS = [6, 12]

    for N in NS:
        top_50_buyers_this_year, total_buyers_count_this_year_, _giniIndexThis_, _churn_rate_[N], \
        _customer_dev_rate_[N], top_50_buyers_percent_this_year = DataQualityManager.getTopBuyers(
            valid_df=df1_valid,
            observeTime=observeTime,
            to_observe_window=pd.DateOffset(months=N * 2),
            top=50)

        top_10_buyers_this_year, total_buyers_count_this_year, giniIndexThis[N], churn_rate[N], customer_dev_rate[
            N], top_10_buyers_percent_this_year = DataQualityManager.getTopBuyers(
            valid_df=df1_valid,
            observeTime=observeTime,
            to_observe_window=pd.DateOffset(months=N),
            top=10)

        top_10_buyers_percent_this_year.reset_index(inplace=True)

        if N == 6:
            for k in (1, 3):
                top_buyers_percent_recent_6_months[k] = top_10_buyers_percent_this_year.loc[0:(k - 1), '金额占比'].sum()

        if N == 12:
            for k in (1, 3):
                top_buyers_percent_recent_12_months[k] = top_10_buyers_percent_this_year.loc[0:(k - 1), '金额占比'].sum()

        top_10_buyers_last_year, total_buyers_count_last_year, giniIndexLast[N], churn_rate_last[N], \
        customer_dev_rate_last[N], top_10_buyers_percent_last_year = DataQualityManager.getTopBuyers(
            valid_df=df1_valid,
            observeTime=observeTime - pd.DateOffset(months=N),
            to_observe_window=pd.DateOffset(months=N),
            top=10)

        top10_buyer_similarity[N] = DataQualityManager.orderedPairNamesListsSimiliarity(
            top_10_buyers_this_year.tolist(),
            top_10_buyers_last_year.tolist())

        logging.info("{}:{}".format(logcode(87, N), top10_buyer_similarity[N]))

        top10_buyer_percent_similarity[N] = DataQualityManager.pairNumberSimilarity(top_10_buyers_percent_this_year,
                                                                                    top_10_buyers_percent_last_year,
                                                                                    '统一名称', '金额占比')

        logging.info("{}:{}".format(logcode(88, N), top10_buyer_percent_similarity[N]))

        if N == 12:
            top_10_buyers_percent_this_year_report = top_10_buyers_percent_this_year[['统一名称', '合计金额', '金额占比']].rename(
                columns={'统一名称': '近12个月前十买家', '合计金额': '销售总额（万元）', '金额占比': '销售总额占比'}).reset_index(drop=True)
            top_10_buyers_percent_last_year_report = top_10_buyers_percent_last_year[['统一名称', '合计金额', '金额占比']].rename(
                columns={'统一名称': '远12个月前十买家', '合计金额': '销售总额（万元）', '金额占比': '销售总额占比'})\
                [['销售总额占比','销售总额（万元）','远12个月前十买家']].reset_index(drop=True)
            top_10_buyers_percent_this_last_year_report = pd.concat([top_10_buyers_percent_this_year_report,top_10_buyers_percent_last_year_report],axis=1)

            settings.R_CONTAINER["十大下游交易买方分析"]["十大买方开票占比对比"]['远近12个月前十买家']['value'] = top_10_buyers_percent_this_last_year_report

        top_10_buyers_percent_this_year['supplier_name'] = supplier_name
        top_10_buyers_percent_this_year['time'] = 'this ' + str(N) + ' months'
        top_10_buyers_percent_last_year['supplier_name'] = supplier_name
        top_10_buyers_percent_last_year['time'] = 'last ' + str(N) + ' months'

        top_buyers_profile = top_buyers_profile.append(top_10_buyers_percent_this_year, ignore_index=True)
        top_buyers_profile = top_buyers_profile.append(top_10_buyers_percent_last_year, ignore_index=True)
    settings.R_CONTAINER["信用评分"]["S-销售稳定性-2"]["企业实际表现"]["value"] = top10_buyer_percent_similarity[12]
    settings.R_CONTAINER["信用评分"]["S-销售成长性-1"]["企业实际表现"]["value"] = _churn_rate_[12]

    top_10_buyers_valid_year_month = DataQualityManager.getTopBuyersHistory(df=df1_valid,
                                                                            observeTime=observeTime,
                                                                            topBuyersList=top_10_buyers_this_year)

    top_50_buyers_valid_year_month = DataQualityManager.getTopBuyersHistory(df=df1_valid,
                                                                             observeTime=observeTime,
                                                                             topBuyersList=top_50_buyers_this_year)

    top_10_deal_table = top_10_buyers_valid_year_month.pivot(index='年月', columns='统一名称',
                                                             values='合计金额').sort_index(ascending=True)

    top_10_deal_table = DataQualityManager.completeMonths(top_10_deal_table,
                                                          start=observeTime - pd.Timedelta(365 * 2 - 1, unit='D'),
                                                          end=observeTime - pd.Timedelta(28, unit='D'),
                                                          set_index='年月')

    top_10_primitive_features = pd.DataFrame(index=top_10_deal_table.columns.values, columns=[
        '买方统一名称',
        '销方名称',
        '当月日期',
        '当月开票额',
        '近12月增长率',
        '近12月开票额离差率',
        '近12月开票额',
        '远12月开票额',
        '近12月占全量比',
        '远12月占全量比',
        '近6月零开票月数',
        '近3月零开票月数',
        '近12月最长连续零开票月数',
        '近12月零开票数',
        '远12月零开票数',
        '近12月红冲率',
        '远12月红冲率',
        '近12月作废率',
        '远12月作废率',
        '首次合作月',
        '最近一次合作月',
        '合作规律性'])

    top_50_deal_table = top_50_buyers_valid_year_month.pivot(index='年月', columns='统一名称', values='合计金额').sort_index(ascending=True)
    top_50_deal_table = DataQualityManager.completeMonths(top_50_deal_table, start=observeTime - pd.Timedelta(365 * 3 - 1, unit='D'),
                                                          end=observeTime - pd.Timedelta(28, unit='D'), set_index='年月')
    tmp_core_ents_dict = df1_valid[df1_valid['是否为核企']==True][['统一名称','核企级别']].drop_duplicates().set_index('统一名称').to_dict()['核企级别']
    top_50_deal_table_temp = top_50_deal_table.copy()[top_50_buyers_this_year]
    top_50_deal_table_temp['合计'] = top_50_deal_table_temp.sum(axis=1)
    top_50_deal_table_temp.index = top_50_deal_table_temp.index.map(lambda x: pd.to_datetime(x).strftime('%Y%m'))
    settings.R_CONTAINER["销项发票买方时序表"]['近36个开票月份']['value'] = top_50_deal_table_temp.index.to_list()
    top_50_deal_table_temp = top_50_deal_table_temp.T.reset_index()

    top_50_deal_table_temp.insert(1, '是否静态白名单', top_50_deal_table_temp['统一名称'].isin(tmp_core_ents_dict.keys()))
    top_50_deal_table_temp.insert(2, '静态白名单等级', top_50_deal_table_temp['统一名称'].map(tmp_core_ents_dict))
    top_50_deal_table_temp.insert(3, '动态白名单等级', top_50_deal_table_temp['统一名称'].map(buyer_class1_dict))
    settings.R_CONTAINER["销项发票买方时序表"]['TOP50买方时序表']['value'] = top_50_deal_table_temp

    ship_port_df = pd.concat([df1_valid_sum_by_yearMonth.set_index('年月')['合计金额'].to_frame(),
    shipping_revenue_sum_by_yearMonth.set_index('年月')['价税合计'].to_frame().rename(columns={'价税合计': '海运费收入-近36个开票月份开票量'}),
    port_charges_sum_by_yearMonth.set_index('年月')['价税合计'].to_frame().rename(columns={'价税合计': '港杂费收入-近36个开票月份开票量'})],axis=1)
    ship_port_df['其他收入-近36个开票月份开票量'] = ship_port_df['合计金额']-ship_port_df['海运费收入-近36个开票月份开票量']-ship_port_df['港杂费收入-近36个开票月份开票量']
    ship_port_df.drop(columns=['合计金额'],inplace=True)
    ship_port_df.index = ship_port_df.index.map(lambda x: pd.to_datetime(x).strftime('%Y%m'))

    amount_24_36 = pd.concat([ship_port_df.iloc[24:36].sum(axis=0).to_frame().rename(columns={0: '近12个月总额'}),
               ship_port_df.iloc[12:24].sum(axis=0).to_frame().rename(columns={0: '远12个月总额'})], axis=1).T

    settings.R_CONTAINER["销项发票分品类时序表"]['海运费收入-近远12个月总额']['value'] = amount_24_36['海运费收入-近36个开票月份开票量'].values.tolist()
    settings.R_CONTAINER["销项发票分品类时序表"]['港杂费收入-近远12个月总额']['value'] = amount_24_36['港杂费收入-近36个开票月份开票量'].values.tolist()
    settings.R_CONTAINER["销项发票分品类时序表"]['其他收入-近远12个月总额']['value'] = amount_24_36['其他收入-近36个开票月份开票量'].values.tolist()
    settings.R_CONTAINER["销项发票分品类时序表"]['近36个开票月份']['value'] = ship_port_df.index.to_list()
    settings.R_CONTAINER["销项发票分品类时序表"]['海运费收入-近36个开票月份开票量']['value'] = ship_port_df['海运费收入-近36个开票月份开票量'].values.tolist()
    settings.R_CONTAINER["销项发票分品类时序表"]['港杂费收入-近36个开票月份开票量']['value'] = ship_port_df['港杂费收入-近36个开票月份开票量'].values.tolist()
    settings.R_CONTAINER["销项发票分品类时序表"]['其他收入-近36个开票月份开票量']['value'] = ship_port_df['其他收入-近36个开票月份开票量'].values.tolist()

    total_ents_df = df1_valid_sum_by_yearMonth.sort_values(by='年月', ascending=False).iloc[:36][['年月', '合计金额', '发票张数']].set_index('年月').rename(columns={'合计金额': '全量买方开票金额(元）', '发票张数': '全量买方开票笔数'})
    static_ents_df = static_state_core_df.sort_values(by='年月', ascending=False).iloc[:36][['年月', '合计金额', '发票张数']].set_index('年月').rename(columns={'合计金额': '静态白名单买方开票金额(元)', '发票张数': '静态白名单买方开票笔数'})
    dynamic_ents_df = dynamic_ents_df.sort_values(by='年月', ascending=False).iloc[:36][['年月', '合计金额', '发票张数']].set_index('年月').rename(columns={'合计金额': '动态白名单买方开票金额(元)', '发票张数': '动态白名单买方开票笔数'})
    buyer_df_by_month = pd.concat([total_ents_df, static_ents_df, dynamic_ents_df], axis=1).reset_index().fillna(0)
    buyer_df_by_month['年月'] = buyer_df_by_month['年月'].dt.strftime("%Y%m")
    settings.R_CONTAINER["分类加总开票时序"]["月度-倒时序"]['value'] = buyer_df_by_month['年月'].values.tolist()
    settings.R_CONTAINER["分类加总开票时序"]["全量买方开票金额(元）-倒时序"]['value'] = buyer_df_by_month['全量买方开票金额(元）'].values.tolist()
    settings.R_CONTAINER["分类加总开票时序"]["动态白名单买方开票金额(元)-倒时序"]['value'] = buyer_df_by_month['动态白名单买方开票金额(元)'].values.tolist()
    settings.R_CONTAINER["分类加总开票时序"]["静态白名单买方开票金额(元)-倒时序"]['value'] = buyer_df_by_month['静态白名单买方开票金额(元)'].values.tolist()
    settings.R_CONTAINER["分类加总开票时序"]["全量买方开票笔数-倒时序"]['value'] = buyer_df_by_month['全量买方开票笔数'].values.tolist()
    settings.R_CONTAINER["分类加总开票时序"]["动态白名单买方开票笔数-倒时序"]['value'] = buyer_df_by_month['动态白名单买方开票笔数'].values.tolist()
    settings.R_CONTAINER["分类加总开票时序"]["静态白名单买方开票笔数-倒时序"]['value'] = buyer_df_by_month['静态白名单买方开票笔数'].values.tolist()

    if extract_quality_buyers:
        top_50_buyers_quality = DataQualityManager.qualityBuyerTagging(top_50_deal_table[top_50_buyers_this_year], latest_month=observeTime)
        top_50_buyers_quality_dict = top_50_buyers_quality['所满足标准'].to_dict()
        top_50_buyers_longest_possible = top_50_buyers_quality['近24月最大可能合作次数'].to_dict()
        top_50_buyers_latest_3_invoicing_month = top_50_buyers_quality['近3月购买月数'].to_dict()
        qualified_buyer_feature = DataQualityManager.characterizeQualityBuyers(df1, list(top_50_buyers_quality[top_50_buyers_quality['是否满足入池标准']].index.values), observeTime)

        for fi in ['X1', 'X2', 'X3', 'X6', 'X9', 'T1', 'T2', 'T3']:
            count = len(top_50_buyers_quality[top_50_buyers_quality['所满足标准'].str.contains(fi)].index)
            qualified_buyer_feature['满足' + fi + '的买方数量'] = count
        buyers_invoice_profile['稳定买方等级'] = buyers_invoice_profile['统一名称'].apply(
            lambda name: top_50_buyers_quality_dict[name] if name in top_50_buyers_quality_dict else '')
        buyers_invoice_profile['近24月最大可能合作次数'] = buyers_invoice_profile['统一名称'].apply(
            lambda name: top_50_buyers_longest_possible[
                name] if name in top_50_buyers_longest_possible else np.nan)
        buyers_invoice_profile['近3合作'] = buyers_invoice_profile['统一名称'].apply(
            lambda name: top_50_buyers_latest_3_invoicing_month[
                name] if name in top_50_buyers_latest_3_invoicing_month else np.nan)
        buyers_invoice_profile['近3同比远6有合作'] = buyers_invoice_profile['统一名称'].apply(lambda name:
                                                                                   np.nan if name not in top_50_buyers_quality_dict
                                                                                   else (True if 'T1' in
                                                                                                 top_50_buyers_quality_dict[
                                                                                                     name] else False))
        buyers_invoice_profile['近3有合作且总合作超过12含'] = buyers_invoice_profile['统一名称'].apply(lambda name:
                                                                                        np.nan if name not in top_50_buyers_quality_dict
                                                                                        else (True if 'T3' in
                                                                                                      top_50_buyers_quality_dict[
                                                                                                          name] else False))
        buyers_invoice_profile['近12月最长连续零开票低于3且近2月有合作'] = buyers_invoice_profile['统一名称'].apply(lambda name:
                                                                                               np.nan if name not in top_50_buyers_quality_dict
                                                                                               else (True if 'T2' in
                                                                                                             top_50_buyers_quality_dict[
                                                                                                                 name] else False))

    if not top_10_deal_table.empty:
        top_10_deal_table.fillna(value=0, inplace=True)

        this_year_start = observeTime - pd.Timedelta(365, unit='D')
        last_year_start = observeTime - pd.Timedelta(365 * 2, unit='D')
        top_10_deal_table_latest_12 = top_10_deal_table[this_year_start: observeTime]
        top_10_deal_table_last_12 = top_10_deal_table[last_year_start: this_year_start]

        top_10_primitive_features['当月日期'] = top_10_deal_table.last_valid_index()
        last_index = top_10_deal_table.last_valid_index()
        top_10_primitive_features['当月开票额'] = top_10_deal_table.loc[last_index].values

        top_10_primitive_features['近12月开票额离差率'] = (np.abs(top_10_deal_table_latest_12 - top_10_deal_table_latest_12.mean()).mean() / (top_10_deal_table_latest_12.mean() + 0.0001)).values
        top_10_primitive_features['近12月开票额'] = (top_10_deal_table_latest_12.sum()).values
        top_10_primitive_features['近12月占全量比'] = top_10_primitive_features['近12月开票额'] / (valid_latest_12_total + 0.0001)
        if total_invoice_months >= 12 + 12:
            top_10_primitive_features['近12月增长率'] = DataQualityManager.ceilTheOutlier(((
                                                                                              top_10_deal_table_latest_12.sum() - top_10_deal_table_last_12.sum()) / (
                                                                                              top_10_deal_table_last_12.sum() + 0.0001)).values,
                                                                                     lower_bound=-100,
                                                                                     upper_bound=100)
            top_10_primitive_features['远12月开票额'] = (top_10_deal_table_last_12.sum()).values
        valid_last_12_total = df1_valid_last_12_by_yearMonth_buyer['合计金额'].sum()
        if total_invoice_months >= 12 + 12:
            top_10_primitive_features['远12月占全量比'] = top_10_primitive_features['远12月开票额'] / (valid_last_12_total + 0.0001)
        if total_invoice_months >= 12:
            top_10_primitive_features['合作规律性'] = (top_10_deal_table_latest_12.apply(
                lambda column: DataQualityManager.sequenceRegularity(column.values))).values
        if total_invoice_months >= 6:
            top_10_primitive_features['近6月零开票月数'] = (top_10_deal_table_latest_12.tail(6).apply(
                lambda column: DataQualityManager.countZeros(column.values))).values
        if total_invoice_months >= 3:
            top_10_primitive_features['近3月零开票月数'] = (top_10_deal_table_latest_12.tail(3).apply(
                lambda column: DataQualityManager.countZeros(column.values))).values
        top_10_primitive_features['近12月最长连续零开票月数'] = (top_10_deal_table_latest_12.apply(
            lambda column: DataQualityManager.longestContinuousZerosLen(column.values))).values
        if total_invoice_months >= 12:
            top_10_primitive_features['近12月零开票数'] = (top_10_deal_table_latest_12.apply(
                lambda column: DataQualityManager.countZeros(column.values))).values
        if total_invoice_months >= 12 + 12:
            top_10_primitive_features['远12月零开票数'] = (top_10_deal_table_last_12.apply(
                lambda column: DataQualityManager.countZeros(column.values))).values
        top_10_primitive_features['首次合作月'] = (top_10_deal_table.apply(lambda column: column.index[column.to_numpy().nonzero()[0][0]])).values
        top_10_primitive_features['最近一次合作月'] = (top_10_deal_table.apply(lambda column: column.index[column.to_numpy().nonzero()[0][-1]])).values

        top_10_buyers_cancel_year_month = DataQualityManager.getTopBuyersHistory(df=df1_cancel,
                                                                                 observeTime=observeTime,
                                                                                 topBuyersList=top_10_buyers_this_year)

        top_10_deal_table_cancel = top_10_buyers_cancel_year_month.pivot(index='年月', columns='统一名称',
                                                                         values='合计金额').sort_index(ascending=True)
        top_10_deal_table_cancel = DataQualityManager.completeMonths(top_10_deal_table_cancel,
                                                                     # start=observeTime - pd.DateOffset(months=12 * 2),
                                                                     start=observeTime - pd.Timedelta(365 * 2 - 1,
                                                                                                      unit='D'),
                                                                     end=observeTime - pd.Timedelta(28,
                                                                                                    unit='D'),
                                                                     set_index='年月')
        top_10_deal_table_cancel.fillna(value=0, inplace=True)

        top_10_deal_table_latest_12_cancel = top_10_deal_table_cancel[this_year_start: observeTime]
        top_10_deal_table_last_12_cancel = top_10_deal_table_cancel[last_year_start: this_year_start]
        top_10_buyers_redCorrection_year_month = DataQualityManager.getTopBuyersHistory(df=df1_redCorrection,
                                                                                        observeTime=observeTime,
                                                                                        topBuyersList=top_10_buyers_this_year)

        top_10_deal_table_redCorrection = top_10_buyers_redCorrection_year_month.pivot(index='年月',
                                                                                       columns='统一名称',
                                                                                       values='合计金额').sort_index(ascending=True)
        top_10_deal_table_redCorrection = DataQualityManager.completeMonths(top_10_deal_table_redCorrection,
                                                                            start=observeTime - pd.Timedelta(365 * 2 - 1, unit='D'),
                                                                            end=observeTime - pd.Timedelta(28, unit='D'),
                                                                            set_index='年月')
        top_10_deal_table_redCorrection.fillna(value=0, inplace=True)

        top_10_deal_table_latest_12_redCorrection = top_10_deal_table_redCorrection[
                                                    this_year_start: observeTime]

        top_10_deal_table_last_12_redCorrection = top_10_deal_table_redCorrection[
                                                  last_year_start: this_year_start]

        if total_invoice_months >= 12:
            top_10_primitive_features['近12月红冲率'] = np.round((
                                                                    np.abs(
                                                                        top_10_deal_table_latest_12_redCorrection).sum() / np.abs(
                                                                top_10_deal_table_latest_12).sum()).replace(
                to_replace=np.nan, value=0).values, 4)
            top_10_primitive_features['近12月作废率'] = np.round((np.abs(top_10_deal_table_latest_12_cancel).sum() / (
                    np.abs(top_10_deal_table_latest_12).sum() + np.abs(
                top_10_deal_table_latest_12_cancel).sum())).replace(to_replace=np.nan, value=0).values, 4)
        if total_invoice_months >= 12 + 12:
            top_10_primitive_features['远12月红冲率'] = np.round((
                                                                    np.abs(
                                                                        top_10_deal_table_last_12_redCorrection).sum() / np.abs(
                                                                top_10_deal_table_last_12).sum()).replace(
                to_replace=np.nan, value=0).values, 4)
            top_10_primitive_features['远12月作废率'] = np.round((np.abs(top_10_deal_table_last_12_cancel).sum() / (
                    np.abs(top_10_deal_table_last_12).sum() + np.abs(
                top_10_deal_table_last_12_cancel).sum())).replace(to_replace=np.nan, value=0).values, 4)

    top_10_buyer_debut = DataQualityManager.groupByMultipleColumns(
        df=top_10_buyers_valid_year_month[['年月', '统一名称']],
        sorted_columns_list=['统一名称'],
        as_index=True,
        aggregate_method='min')
    top_10_buyer_debut['合作日长'] = observeTime - top_10_buyer_debut['年月']
    top_10_average_months = top_10_buyer_debut['合作日长'].mean() / np.timedelta64(1, 'M')
    settings.R_CONTAINER["信用评分"]["S-销售连续性-1"]["企业实际表现"]["value"] = top_10_average_months

    top_10_deviation_ratio = top_10_primitive_features['近12月开票额离差率'].sort_values(ascending=False).median()
    settings.R_CONTAINER["信用评分"]["S-销售连续性-2"]["企业实际表现"]["value"] = top_10_deviation_ratio
    top_10_average_zero_invoices_latest_6 = top_10_primitive_features['近6月零开票月数'].median()
    top_10_average_zero_invoices_latest_3 = top_10_primitive_features['近3月零开票月数'].median()
    top_10_average_longest_continuing_zeros_invoice = top_10_primitive_features['近12月最长连续零开票月数'].median()
    top_10_average_deviation_this_12 = top_10_primitive_features['近12月开票额离差率'].median()
    top_10_redCorrection_this_12 = top_10_primitive_features['近12月红冲率'].sort_values(ascending=False).head(
        5).median()
    top_10_redCorrection_last_12 = top_10_primitive_features['远12月红冲率'].sort_values(ascending=False).head(
        5).median()
    top_10_cancel_this_12 = top_10_primitive_features['近12月作废率'].sort_values(ascending=False).head(5).median()
    top_10_cancel_last_12 = top_10_primitive_features['远12月作废率'].sort_values(ascending=False).head(5).median()
    top_10_YoY_growth_12 = top_10_primitive_features['近12月增长率'].median()
    top_10_top_dominance_this = top_10_primitive_features['近12月占全量比'].sort_values(ascending=False).head(3).sum()
    top_10_average_zero_invoices_latest_12 = top_10_primitive_features['近12月零开票数'].median()

    top_10_top_dominance_last = top_10_primitive_features['远12月占全量比'].sort_values(ascending=False).head(3).sum()

    top_10_primitive_features['销方名称'] = supplier_name
    top_10_primitive_features['买方统一名称'] = top_10_primitive_features.index
    top_10_primitive_features.reset_index(drop=True, inplace=True)
    top_10_buyers_records = top_10_primitive_features.sort_values(by='近12月开票额', ascending=False)
    top_10_buyers_records['当月日期'] = top_10_buyers_records['当月日期'].apply(lambda x: pd.to_datetime(x).strftime('%Y%m%d'))
    top_10_buyers_records['首次合作月'] = top_10_buyers_records['首次合作月'].apply(lambda x: pd.to_datetime(x).strftime('%Y%m'))
    top_10_buyers_records['最近一次合作月'] = top_10_buyers_records['最近一次合作月'].apply(lambda x: pd.to_datetime(x).strftime('%Y%m'))
    top10_features_report=['买方统一名称','当月开票额','近3月零开票月数','近12月开票额','远12月开票额','近12月占全量比',\
                           '近12月增长率','近12月最长连续零开票月数','近12月零开票数','近12月红冲率','近12月作废率',\
                           '首次合作月','最近一次合作月']
    top_10_buyers_records = top_10_buyers_records[top10_features_report]
    settings.R_CONTAINER["十大下游交易买方分析"]["十大买方基础指标"]['value'] = top_10_buyers_records

    quarterly_deviation = {}

    retention_rate = {}

    NS = [3, 6, 12]

    for N in NS:
        valid_latest_N_months = DataQualityManager.selectByTimeRange(df=df1_valid, time_column='开票日期',
                                                                     observeTime=observeTime,
                                                                     to_observe_time_period=pd.DateOffset(months=N))

        valid_sum_by_month = DataQualityManager.groupByMultipleColumns(df=valid_latest_N_months,
                                                                       sorted_columns_list=['年月'],
                                                                       as_index=False,
                                                                       aggregate_method='sum',
                                                                       end=observeTime - pd.Timedelta(
                                                                           28, 'D'),
                                                                       start=observeTime - pd.DateOffset(
                                                                           months=N))
        tlist1 = []
        for i in range(2, len(valid_sum_by_month), 1):
            moving_avg_sum = valid_sum_by_month.loc[i - 1, '合计金额'] * 0.3 + valid_sum_by_month.loc[
                i - 2, '合计金额'] * 0.1 + valid_sum_by_month.loc[i, '合计金额'] * 0.6
            tlist1.append(moving_avg_sum)

        mean_ = np.mean(tlist1)

        tlist2 = [np.abs(i - mean_) for i in tlist1]

        mean_deviations = np.mean(tlist2)

        quarterly_deviation[N] = np.round(mean_deviations / (mean_ + 0.0001), 4)

        cust_sets_by_freq = valid_latest_N_months.set_index('年月').groupby(pd.Grouper(freq='2M'))['统一名称'].apply(
            lambda x: set(x) if not pd.isna(x).all() else np.nan)

        cust_sets_by_freq = pd.DataFrame(cust_sets_by_freq).reset_index()

        retention_rates = list()

        for i in range(1, len(cust_sets_by_freq), 1):
            set1 = cust_sets_by_freq.loc[i - 1, '统一名称']
            set2 = cust_sets_by_freq.loc[i, '统一名称']
            if isinstance(set1, set) and isinstance(set2, set) and len(set1) > 0 and len(set2) > 0:
                t1 = set1.intersection(set2)
                t2 = set1.union(set2)
                if len(t1) > 0 and len(t2) > 0:
                    similarity = len(set1.intersection(set2)) / len(set1.union(set2))
                else:
                    similarity = 0
            else:
                similarity = 0
            retention_rates.append(similarity)

        retention_rate[N] = np.mean(retention_rates)

    # credit_score = Scorer.match('forwarder_v1', valid_months=total_invoice_months, offset=0., factor=1., debug=True).scorer(dict(), left_clip=0, right_clip=100).get_score()
    default_profit_rate = settings.R_CONTAINER["信用测额"]["L-利润-1"]["可配参数"]["value"]
    # CreditRation.match('PYP').base_ration(buyers_invoice_profile, credit_score, shipping_revenue_latest_12_months, default_profit_rate).get_rations()
    # RiskEntry().decide_on_features(dict((f, settings.R_CONTAINER["准入规则"][f]["企业实际表现"]["value"]) for f in settings.R_CONTAINER["准入规则"]))

    end_run_time = pd.Timestamp.now()
    logging.info('end run :{}, total time cost: {}'.format(end_run_time, end_run_time - start_run_time))
    logging.info(logcode(92))

    if save_source_data:
        dump_model_to_file(settings.R_CONTAINER, os.path.join(client_outpath, "R_CONTAINER.pkl"))
        logging.info("settings持久化完成!")

    return top_10_buyers_records
