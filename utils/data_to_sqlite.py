import pandas as pd
import sqlite3
import os
import sys
from dataSource import DBconnector


class ToSqlite(object):
    def __init__(self, db_path):
        # self.conn = create_engine(r'sqlite:///{}'.format(db_path))
        self.conn = sqlite3.connect(db_path)

    def df_to_table(self, dataframe, tablename, if_exists='append'):
        """
        write dataframe to sqlite3 by tablename
        Parameters
        ----------
        dataframe df dataset
        tablename sqlite tablename
        if_exists If the table already exists, {‘fail’, ‘replace’, ‘append’}, default ‘fail’

        Returns
        -------

        """
        dataframe.to_sql(tablename, self.conn, if_exists=if_exists, index=False)

    def close(self):
        self.conn.close()


class DataToLocal(object):
    """
    发票项目中，包含以下几部分数据，需要从mysql中获取，因此发票项目本地化，需要先将如下数据集，获取下来，存储到本地sqlite3中
    后续发票项目只需要将连接方式改为从sqlite中查询即可
    """

    def __init__(self, sqlite_path):
        self.sqlutils = ToSqlite(sqlite_path)

    def core_ents_to_sqlite(self,if_exist='append'):
        """静态核心企业库存入sqlite"""
        WHITE_LIST_BUYER_CONN_PYP = DBconnector(DBType='mysql', host='172.16.90.84', port=23306, db='piao_invoicedb',
                                                user='bee', password='Llsbee!')
        query_core_brands = """SELECT distinct white_list_name,level,industry
                                from piao_invoicedb.tm_white_list
                                 where enabled = 1"""
        coreBrands_df = WHITE_LIST_BUYER_CONN_PYP.query(query=query_core_brands).as_pandas().getQueryResult()
        print("当前查询得到白名单数量:{}".format(coreBrands_df.shape[0]))
        self.sqlutils.df_to_table(coreBrands_df, 'tm_white_list',if_exists=if_exist)

    def multi_taxnos_to_sqlite(self,if_exist='append'):
        """多税号表存入sqlite"""
        SELLER_TAX_NO_CONN = DBconnector(DBType="mysql", host='172.16.90.84', port=23306, db='bigdata_fkdb',
                                         user='bee', password='Llsbee!')
        query_taxno = """
           SELECT seller_name,seller_taxno
           from fk_seller_taxno
        """
        taxno_df = SELLER_TAX_NO_CONN.query(query=query_taxno).as_pandas().typeConverter(
            {"税号": "string"}).getQueryResult()
        print('多税号表数量:{}'.format(taxno_df.shape[0]))
        self.sqlutils.df_to_table(taxno_df, 'fk_seller_taxno',if_exists=if_exist)

    def invoice_to_sqlite(self, taxnos, if_exist='append'):
        INVOICE_CONN = DBconnector(DBType="mysql", host='172.16.90.83', port=8066, db='csmsdb',
                                   user='bee', password='Llsbee!')
        queryInvoice = """  
          SELECT saleinvoice.invoiceid,
          saleinvoice.invoicetype,
          saleinvoice.invoicecode,
          saleinvoice.invoiceno,
          saleinvoice.buyername,
          saleinvoice.buyertaxno,
          saleinvoice.sellername,
          saleinvoice.sellertaxno,
          saleinvoice.totalamount as tatolamount,
          saleinvoice.taxrate,
          saleinvoice.totaltax,
          saleinvoice.invoicedate,
          saleinvoice.makeinvoicedeviceno,
          saleinvoice.makeinvoiceperson,
          saleinvoice.selleraddtel,
          saleinvoice.sellerbankno,
          saleinvoice.comments,
          saleinvoice.cancelflag
           FROM rz_invoice saleinvoice
           WHERE saleinvoice.sellertaxno IN
           (
             {}
           )
           ORDER BY invoicedate
           """.format(taxnos)
        invoice_df = INVOICE_CONN.query(query=queryInvoice).as_pandas().getQueryResult()
        print('税号[{}]的发票数量：{}'.format(taxnos, invoice_df.shape[0]))
        self.sqlutils.df_to_table(invoice_df, 'saleinvoice', if_exists=if_exist)

    def invoicedetaildata_to_sqlite(self, taxnos,if_exist='append'):
        INVOICE_CONN = DBconnector(DBType="mysql", host='172.16.90.83', port=8066, db='csmsdb',
                                   user='bee', password='Llsbee!')
        query_invoicedetail = """select
        c.invoiceid,
        c.sellertaxno,
        c.invoicedetailno,
        c.detaillistflag,
        c.WARESNAME,
        c.STANDARDTYPE,
        c.CALCUNIT,
        c.QUANTITY,
        c.NOTAXPRICE,
        c.AMOUNT,
        c.TAXRATE,
        c.TAXAMOUNT,
        c.TAXPRICE,
        c.TAXCODE
        from saleinvoicedetail c
        where c.sellertaxno in ({})
        """.format(taxnos)
        invoicedetail_df = INVOICE_CONN.query(query=query_invoicedetail).as_pandas().getQueryResult()
        print('税号[{}]的发票详情数量：{}'.format(taxnos, invoicedetail_df.shape[0]))
        self.sqlutils.df_to_table(invoicedetail_df, 'saleinvoicedetail',if_exists=if_exist)

    def related_ents_to_sqlite(self, if_exist='append'):
        """关联企业表存入sqlite"""
        RELATED_CORPS_CONN_PYP = DBconnector(DBType="mysql", host='172.16.90.84', port=23306, db='sme_cust',
                                             user='bee', password='Llsbee!')
        query_affliates_csmsdb = """
                                    select cri.relation_cust_name,cci.company_name
                                    from sme_cust.cust_relationcompany_info as cri,
                                    sme_cust.cust_company_info as cci
                                    where cci.id = cri.company_id
                                """
        related_ents_df = RELATED_CORPS_CONN_PYP.query(query_affliates_csmsdb).as_pandas().getQueryResult()
        print('关联企业表数量:{}'.format(related_ents_df.shape[0]))
        self.sqlutils.df_to_table(related_ents_df, 'cust_relationcompany_info',if_exists=if_exist)

    def close(self):
        self.sqlutils.close()


def config_db_export(if_exist='append'):
    # 准备核心企业数据表、多税号表、关联企业表
    SQLITE_PATH = os.path.join(BASE_PATH, '../../configs') + "/config_datas.db"
    print('sqlite path: [{}]'.format(SQLITE_PATH))
    dl = DataToLocal(SQLITE_PATH)
    dl.core_ents_to_sqlite(if_exist)
    dl.related_ents_to_sqlite(if_exist)
    dl.close()


def sample_invoice_export(taxnos, if_exist='append'):
    """
    Parameters
    ----------
    if_exists If the table already exists, {‘fail’, ‘replace’, ‘append’}, default ‘fail’
    Returns
    -------

    """
    # 准备客户的发票数据
    SQLITE_INVOICE_PATH = os.path.join(BASE_PATH, '../../configs') + "/client_invoice_datas.db"
    print('sqlite invoice path: [{}]'.format(SQLITE_INVOICE_PATH))
    dl = DataToLocal(SQLITE_INVOICE_PATH)
    dl.invoice_to_sqlite(taxnos,if_exist)
    dl.invoicedetaildata_to_sqlite(taxnos,if_exist)
    dl.close()


def test_sqlalchemy():
    demo_path = os.path.join(BASE_PATH, 'output') + "/demo.db"

    df = pd.DataFrame({'name': ['tom4', 'jerry4'], 'age': [23, 25]})

    sqlutils = ToSqlite(demo_path)
    sqlutils.df_to_table(df, 'demo01', if_exists='append')

    # from sqlalchemy import create_engine
    # engine = create_engine(r'sqlite:///{}'.format(demo_path))
    # df.to_sql('demo01',engine,if_exists='append',index=False)


if __name__ == '__main__':
    BASE_PATH = os.path.join(os.path.pardir, sys.path[0])  # 生产环境
    print('BASE PATH: ' + BASE_PATH)

    taxnos = """ '91310113685471496R' """
    sample_invoice_export(taxnos=taxnos,if_exist='replace')
    # config_db_export(if_exist='replace')
    # test_sqlalchemy()


