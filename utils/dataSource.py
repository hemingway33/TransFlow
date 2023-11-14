import pandas as pd
import numpy as np
import sqlite3
from lsh_utils import min_hash_enterprise_name
from datasketch import LeanMinHash, MinHashLSH
import pickle


class DBconnector(object):
    def __init__(self, DBType, host=None, port=None, db=None, user=None, password=None, url=None):
        self.DBType_ = DBType
        assert self.DBType_ == 'sqlite3', "离线版本不支持mysql数据库后端"
        self.host_ = host
        self.port_ = port
        self.user_ = user
        self.password_ = password
        self.db_ = db
        self.url_ = url
        self.cursor_ = self.connect()
        self.queryResult_ = None

    def connect(self):
        if self.DBType_ is 'sqlite3':
            conn = sqlite3.connect(self.host_)
            return conn

    def __str__(self):
        return self.host_

    def __repr__(self):
        return self.host_

    def query(self, query):
        if self.DBType_ is 'sqlite3':
            df_large = pd.read_sql_query(query, self.cursor_)
            self.queryResult_ = df_large
        return self

    def getQueryResult(self):
        return self.queryResult_

    def as_pandas(self):
        if self.DBType_ is 'sqlite3':
            self.queryResult_.replace(to_replace=np.nan, value='', inplace=True)
        return self

    def typeConverter(self, convertersDict):
        columns = self.queryResult_.columns.tolist()
        for column in convertersDict:
            if column in columns:
                if convertersDict[column] is 'datetime':
                    self.queryResult_[column] = pd.to_datetime(self.queryResult_[column], utc=True, infer_datetime_format=True).dt.tz_convert('Asia/Shanghai')
                if convertersDict[column] is 'string':
                    self.queryResult_[column] = self.queryResult_[column].astype(str)
                if convertersDict[column] is 'numeric':
                    self.queryResult_[column] = pd.to_numeric(self.queryResult_[column])
                if convertersDict[column] is 'time':
                    self.queryResult_[column] = pd.to_datetime(self.queryResult_[column], infer_datetime_format=True).dt.tz_localize('Asia/Shanghai')
                if convertersDict[column] is 'datetime1':
                    self.queryResult_[column] = pd.to_datetime(self.queryResult_[column], utc=True, unit='ms', infer_datetime_format=True).dt.tz_convert('Asia/Shanghai')
        return self

    def setIndex(self, target_column):
        self.queryResult_ = self.queryResult_.set_index(target_column)
        return self

    def writeTo(self, from_df, target_impala_table):
        pass

    def close(self):
        self.cursor_.close()


class LshDB(object):

    def __init__(self, num_perm, threshold, administrative_divisions_patt, organization_patt, stop_words_machine,
                 anonymize_lsh, anonymous_column="匿名化编号"):
        self.num_perm = num_perm
        self.threshold = threshold
        self.data_ = None
        self.anonymized_anchor_profile = None
        self.local_lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        self.local_lean_lsh_path = None
        self.administrative_divisions_patt = administrative_divisions_patt
        self.organization_patt = organization_patt
        self.stop_words_machine = stop_words_machine
        self.is_anonymized = anonymize_lsh
        self.anchor_dict = {}
        self.anonymous_column = anonymous_column

    def make(self):
        self.anonymized_anchor_profile = dict((self.anchor_dict[anchor][self.anonymous_column], self.anchor_dict[anchor]) for anchor in self.anchor_dict)
        return self

    def batch_minhash(self,anchor_dict):
        self.anchor_dict = anchor_dict
        if not self.is_anonymized:
            self.data_ = [(anchor, min_hash_enterprise_name(anchor, self.administrative_divisions_patt, self.organization_patt, self.stop_words_machine, self.num_perm))
                          for anchor in self.anchor_dict]
        else:
            self.data_ = [(self.anchor_dict[anchor][self.anonymous_column],
                           min_hash_enterprise_name(anchor, self.administrative_divisions_patt, self.organization_patt, self.stop_words_machine, self.num_perm)) for anchor in self.anchor_dict]
        self.make()
        return self

    def to_updatable_lean_lsh(self):
        for key, minhash in self.data_:
            lean_minhash = LeanMinHash(minhash)
            self.local_lsh.insert(key, lean_minhash, check_duplication=False)
        self.data_ = None
        return self

    def set_local_lean_lsh_path(self, lsh_file):
        self.local_lean_lsh_path = lsh_file
        return self

    def to_local_lean_lsh(self):
        """ 储存为本地序列化的 """
        assert self.local_lean_lsh_path is not None
        with open(self.local_lean_lsh_path, 'wb') as file:
            pickle.dump(self.local_lsh, file)
        return self

    def to_local_core_hash(self,path):
        """ 储存为本地序列化的 """
        with open(path, 'wb') as file:
            pickle.dump(self.anonymized_anchor_profile, file)
        return self

    def load_local_core_hash(self,path):
        """ 反序列化本地示例 """
        with open(path, 'rb') as file:
            anonymized_anchor_profile = pickle.load(file)
        self.anonymized_anchor_profile = anonymized_anchor_profile
        return anonymized_anchor_profile

    def load_local_lean_lsh(self):
        """ 反序列化本地示例 """
        assert self.local_lean_lsh_path is not None
        with open(self.local_lean_lsh_path, 'rb') as file:
            lean_lsh = pickle.load(file)
        self.local_lsh = lean_lsh
        return lean_lsh

    def query_one_local(self, name):
        hash_value = min_hash_enterprise_name(name, self.administrative_divisions_patt, self.organization_patt,
                                              self.stop_words_machine, self.num_perm)
        return self.local_lsh.query(hash_value)

    def make_hash(self, name):
        hash_value = min_hash_enterprise_name(name, self.administrative_divisions_patt, self.organization_patt,
                                              self.stop_words_machine, self.num_perm)
        return hash_value

    def query_many_local(self, names):
        return [self.query_one_local(name) for name in names]

