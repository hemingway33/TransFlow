import ahocorasick
from pandas import read_excel
import pickle


class TransFlowTagger:
    def __int__(self):
        self.key_words_scheme_path_ = "./流水标签规则文档v20220517.xlsx"
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


if __name__ == "__main__":
    TransFlowTagger().make_tag_machine()



