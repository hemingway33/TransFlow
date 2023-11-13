"""
    Created by hushiwei on 2021/10/12
"""

import re
from datasketch import MinHash

father_end_signal = ["集团", "公司", "店", "药房", "院", "站", "大学", "学校", "基金", "园", "电视台", "银行", "小学", "部队", "狱", "厂", "会",
                     "所", "政府", "局", "理处", "馆", "中心", "社"]
father_end_signal_patt = "|".join(father_end_signal)


def min_hash_enterprise_name(enterprise_name, administrative_divisions_pattern, organization_pattern,
                             stop_words_pattern, num_perm=256):
    vector = parse_and_quantize(enterprise_name, administrative_divisions_pattern, organization_pattern,
                                stop_words_pattern, father_end_signal_patt)
    # print("vectorized enterprise_name :{}".format(vector))
    m1 = MinHash(num_perm=num_perm)
    for d in vector:
        m1.update(d.encode('utf8'))
    return m1


def has_parentheses(text):
    text = re.sub(r"\【|\（|\[", "(", text)
    text = re.sub(r"\】|\）|\]", ")", text)

    parenthesized_contents = re.finditer(r'\((.*?)\)', text)
    parenthesized = []
    parenthesized_loc = []
    for match in parenthesized_contents:
        parent_ = text[match.start(): match.end()]
        parenthesized.append(parent_)
        parenthesized_loc.append((match.start(), match.end()))
    return len(parenthesized) > 0, parenthesized, parenthesized_loc


def has_administrative_division(text, administrative_divisions_pattern):
    divisions = []
    divisions_loc = []
    if len(text) < 2:
        return False, divisions, divisions_loc
    for match in re.finditer(administrative_divisions_pattern, text):
        divisions.append(match.group())
        divisions_loc.append(match.span())
    return len(divisions) > 0, divisions, divisions_loc


def has_organization_form(text, organization_pat):
    org_types = []
    org_types_loc = []
    if len(text) < 4:
        return False, org_types, org_types_loc
    for match in re.finditer(organization_pat, text):
        org_types.append(match.group())
        org_types_loc.append(match.span())
    return len(org_types) > 0, org_types, org_types_loc


def has_hierarchy_form(text, father_end_signal_pat):
    hierarchies = list(filter(None, re.split(father_end_signal_pat, text)))
    return len(hierarchies) > 1, hierarchies


def replace_stop_words_with_machine(text, stop_word_machine):
    for end_idx, (_source, _replace) in stop_word_machine.iter(text):
        # start_idx = end_idx - len(_source) + 1
        text = text.replace(_source, _replace)
    return text


def extract_key_body(truncated_text, stop_words_pattern):
    return replace_stop_words_with_machine(truncated_text, stop_words_pattern)


def n_gram_generator(text, n):
    n_grams = zip(*[text[i:] for i in range(n)])
    return ["".join(ngram) for ngram in n_grams]


def parse_and_quantize(enterprise_name, administrative_divisions_pattern, organization_pattern,
                       stop_words_pattern, father_end_signal_pattern):
    quantized_features = set()
    enterprise_name = enterprise_name.lower().strip()
    is_parenthesized, parenthesized, parenthesized_loc = has_parentheses(enterprise_name)
    if is_parenthesized:
        for par in parenthesized:
            quantized_features.add(par.replace("(", "").replace(")", ""))
            enterprise_name = enterprise_name.replace(par, "")

    has_division, divisions, divisions_loc = has_administrative_division(enterprise_name,
                                                                         administrative_divisions_pattern)
    if has_division:
        for div in set(divisions):
            quantized_features.add(div)
            enterprise_name = enterprise_name.replace(div, "")

    has_org, org_types, org_types_loc = has_organization_form(enterprise_name, organization_pattern)
    if has_org:
        key_body = extract_key_body(enterprise_name[:org_types_loc[0][0]], stop_words_pattern)
        for org in org_types:
            quantized_features.add(org)
    else:
        key_body = extract_key_body(enterprise_name, stop_words_pattern)

    has_hierarchies, hierarchies = has_hierarchy_form(enterprise_name, father_end_signal_pattern)
    if has_hierarchies:
        key_body = extract_key_body(hierarchies[0], stop_words_pattern)
        for div in set(divisions):
            if div not in key_body:
                quantized_features.remove(div)

    if key_body is not None:
        quantized_features.update(n_gram_generator(key_body, 2))
        quantized_features.update(n_gram_generator(key_body, 3))
    return quantized_features
