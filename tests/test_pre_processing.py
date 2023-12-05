from load_test_case import load_test_case
from preprocess.deduplicate.identify_names import identify_same_parties, rename_person_party, identify_self_trans
from preprocess.anchoring.identify_anchor import sketch_core_brands, load_anchor_lsh

test_name = "东莞市婉晶电子科技有限公司"
# test identify_same_parties
test_case_pd = load_test_case("东莞市婉晶电子科技有限公司-对公存款分户账明细记录.xlsx")
test_case_pd = identify_same_parties(test_case_pd, name_column="对方户名", threshold=0.85)
test_case_pd.to_csv("temp_identify_same_parties.csv", encoding="utf-8-sig")

# test_rename_person_party, identify_self_trans
test_case_pd = rename_person_party(test_case_pd, name_column="对方户名")
test_case_pd = identify_self_trans(test_case_pd, name_column="对方户名", self_name=test_name)
test_case_pd.to_csv("temp_rename_person_party.csv", encoding="utf-8-sig")

# test_core_parties
anchor_lsh = load_anchor_lsh()
test_case_pd = sketch_core_brands(test_case_pd, anchor_lsh)
test_case_pd.to_csv("temp_core_parties.csv", encoding="utf-8-sig")

