UNCERTAIN_CATEGORY = "无法确定"
WEIBO_CATEGORY_DICT = {
    "经济": 0,
    "健康": 1,
    "军事": 2,
    "科学": 3,
    "政治": 4,
    "国际": 5,
    "教育": 6,
    "娱乐": 7,
    "社会": 8,
}
WEIBO21_CATEGORY_DICT = {
    "科技": 0,
    "军事": 1,
    "教育考试": 2,
    "灾难事故": 3,
    "政治": 4,
    "医药健康": 5,
    "财经商业": 6,
    "文体娱乐": 7,
    "社会生活": 8,
}
FINEFAKE_CATEGORY_DICT = {
    "Business": 0,
    "Conflict": 1,
    "Entertainment": 2,
    "Health": 3,
    "Politics": 4,
    "Society": 5,
    "Uncategorized": 6,
}
DOMAIN_NAME_ALIASES = {
    "经济": "Economy",
    "健康": "Health",
    "军事": "Military",
    "科学": "Science",
    "政治": "Politics",
    "国际": "International",
    "教育": "Education",
    "娱乐": "Entertainment",
    "社会": "Society",
    "科技": "Technology",
    "教育考试": "EducationExam",
    "灾难事故": "DisasterAccident",
    "医药健康": "MedicalHealth",
    "财经商业": "FinanceBusiness",
    "文体娱乐": "CultureSportsEntertainment",
    "社会生活": "SocialLife",
}

def category_to_id(category, category_dict):
    if category_dict is None:
        raise ValueError("category_dict is required.")
    if category in category_dict:
        return category_dict[category]
    category_text = str(category).strip()
    if category_text in category_dict:
        return category_dict[category_text]
    try:
        category_id = int(category_text)
    except (TypeError, ValueError):
        category_id = None
    if category_id is not None and category_id in set(category_dict.values()):
        return category_id
    raise KeyError(f"Unknown category: {category}")

def display_domain_name(category_name):
    return DOMAIN_NAME_ALIASES.get(category_name, str(category_name))
