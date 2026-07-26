#!/usr/bin/env python3
"""Field vocabulary: the working words the corpus actually contains, labelled by taxonomy category.

Why this file exists. Of the 2,812 distinct terms mined from the real corpus, only 326 (11.6%) appear
anywhere in ICD-10-CM / PCS / LOINC / SNOMED. The other 88% is Taiwanese community-health working
vocabulary that no public code system publishes: heavily abbreviated hospital names, community care
sites named after their village, membership and case-lifecycle states, and phone-follow-up phrases.
Those categories carry only 3-8 seed exemplars in the taxonomy, and they are exactly the ones that
fail in the field. No amount of code-system training reaches them, because the vocabulary is not in
any code system -- it has to come from the files themselves.

These are labelled term TYPES (vocabulary), never row values, so no individual's record is
represented here. Terms were selected by corpus frequency, so the list reflects what operators
actually write rather than what an engineer imagines they write.

Seeds TEACH by meaning -- the embedding generalizes from them to unseen wording. This is training
supervision, not a runtime keyword list: nothing here is matched literally at inference. The tuple
contents below are Traditional Chinese data literals, not commentary.
"""

from __future__ import annotations

# Where care happened, or who provided it. Taiwanese practice abbreviates hospital names heavily (a
# four-character hospital name collapses to two), and names community sites after their village, so
# the surface form carries no clue that it denotes an institution. Both read as care context rather
# than as a person or a place of residence -- the distinction the department slot gets wrong today.
CARE_SOURCE = (
    "嘉基", "聖馬", "陽明", "長庚", "嘉榮", "嘉醫", "他院", "診所", "衛生所", "衛生局",
    "健康關懷站", "關懷站", "據點", "社區照顧關懷據點", "長照中心", "居家護理所",
    "骨科", "新代", "新陳代謝科", "心臟內科", "心內", "腎臟科", "神經內科", "胸腔內科",
    "家醫科", "復健科", "婦產科", "小兒科", "眼科", "耳鼻喉科", "皮膚科", "精神科",
    "牙科", "泌尿科", "腸胃科", "血液腫瘤科", "感染科", "風濕免疫科", "一般外科",
    "門診", "急診", "住院", "轉診單位", "承辦醫院",
)

# An intention or arrangement to be seen again, plus the contact attempts around it. These are
# follow-up ACTS, not conditions -- the class of phrase a regex alias was patched in to catch.
CARE_FOLLOWUP = (
    "回診", "複診", "追蹤", "定期追蹤", "持續追蹤", "定期回診", "規律回診", "定期門診",
    "未接", "無接聽電話", "電話未通", "已聯繫", "聯繫不上", "擇日再撥", "再追蹤",
    "提醒下次關懷時段", "提醒回診", "規律3個月抽血", "三個月後追蹤", "半年追蹤",
    "下次關懷", "關懷天氣變化大", "未繳糞便", "待補件", "已提醒",
)

# The administrative state of a case or a membership: a lifecycle fact about the RECORD, not a
# clinical fact about the person.
CASE_STATUS = (
    "永久有效", "失效", "永久會員", "常年會員", "正會員", "學生會員", "一般會員",
    "結案", "開案", "在案", "已結案", "暫停服務", "服務中", "符合", "不符合",
    "不適用", "待審", "已審核", "退件", "新案", "舊案",
)

# Willingness or arrangement to take a checkup -- an intent, which is distinct from having had one.
CHECKUP_INTENT = (
    "健檢意願", "願意受檢", "不願受檢", "願意參加", "不參加", "考慮中", "已預約",
    "待預約", "同意受檢", "拒絕受檢", "農民健檢", "成人健檢", "老人健檢", "四癌篩檢",
)

# Health-promotion contact: teaching given, advice delivered, a leaflet handed over.
HEALTH_EDUCATION = (
    "衛教", "衛教單張", "已衛教", "營養衛教", "用藥衛教", "運動衛教", "戒菸衛教",
    "飲食衛教", "口腔衛教", "跌倒預防衛教", "健康講座", "健康促進活動", "團體衛教",
)

# Substance exposure and cessation status -- a social-history fact. Cessation and current use are
# opposite states expressed by a one-character difference, so this category is also a hard-negative
# source in its own right.
SUBSTANCE_USE = (
    "抽菸", "吸菸", "戒菸", "已戒菸", "未戒菸", "嚼檳榔", "戒檳榔", "喝酒", "飲酒",
    "戒酒", "無抽菸", "不吸菸", "偶爾飲酒", "每日飲酒", "菸酒檳榔皆無",
)

# Named laboratory and examination items as the national insurance schedule writes them. These
# dominate the corpus by volume and are absent from the LOINC Chinese displays we ship, which is why
# the lab slot under-reads on real files.
LAB_VALUE = (
    "白血球表面標記", "白血球分類計數", "白血球酯脢", "全套血液檢查", "血液氣體分析",
    "胺基酸定量檢查", "核糖核酸類定性擴增試驗", "核糖核酸類定量擴增試驗",
    "細菌最低抑制濃度快速試驗", "試管抗藥性試驗", "代謝產物串聯質譜儀分析",
    "免疫病理檢查", "血小板抗體", "尿沉渣", "尿沈渣", "尿膽元", "膽紅素", "潛血",
    "潛血反應", "酸鹼度及酮體", "比重", "混濁度", "澱粉脢", "蛋白電泳分析",
    "皮質素免疫分析", "生長激素免疫分析", "胰島素免疫分析", "微白蛋白", "白蛋白",
    "尿糖試紙檢查", "尿一般檢查", "糞便一般檢查", "痰液一般檢查", "特殊血型",
    "血脂", "血糖", "血壓", "糖化血色素", "肝功能", "腎功能", "尿酸", "膽固醇",
)

# Contactable identity attributes as VALUES (what a phone number means), not as column headers.
CONTACT = (
    "手機", "市話", "聯絡電話", "緊急聯絡人", "家屬電話", "聯絡人",
)

# Strings that must land far from EVERY category. Two kinds occur in real files: administrative
# furniture (a blank-cell placeholder, a price, a form's own boilerplate, a column header appearing
# as a value) and content from documents that are not clinical at all -- the corpus includes an
# information-security audit checklist whose vocabulary would otherwise be forced into a clinical
# category by a classifier that has never been taught to refuse.
REJECTION = (
    "空白", "無", "尚無", "不詳", "未填", "其他", "備註", "說明", "小計", "合計",
    "總計", "編號", "序號", "項目", "名稱", "單位", "數量", "金額", "元",
    "姓名", "性別", "電話", "地址", "身分證", "身分證字號", "出生日期", "健檢編號",
    "郵件編號", "寄發日", "健檢日期", "寄件方式",
    "機器學習", "電腦視覺", "影像處理", "本系統經評定為普級", "資通系統", "防護基準",
    "教授", "副教授", "助理教授", "榮民", "學生",
    "嘉義市", "台北市", "台中市", "台南市", "新北市", "桃園市", "新竹市", "雲林縣",
    "西區", "東區", "大安區", "中壢區",
)

# Report-delivery options. These belong to the column-to-concept layer, not to the value taxonomy,
# and are kept separate so they train the concept side without polluting a clinical category. The
# corpus shows this concept is often enumerated IN the header, which is precisely why a label-only
# matcher misses it.
DELIVERY_METHOD = (
    "郵寄", "自取", "寄公司", "給公司", "親自領取", "掛號寄送", "農民健檢郵寄",
    "現場領取", "委託領取", "寄住家",
)

# Minimal-pair opposites drawn from field wording, as hard-negative TRAINING supervision.
#
# These exist because the code-system miner cannot reach this class: a bare polarity term is usually
# not itself a billable display (the code system carries the elaborated diagnosis, not the two-word
# form an operator types), so mining produced long procedure pairs the model already separates while
# the short clinical opposites that actually collide went untrained.
#
# Deliberately DISJOINT from the held-out antonym probes: training on the probe items would convert
# the measurement into memorization. Generalizing from these to those is the thing being tested.
POLARITY_PAIRS = (
    ("高血鈣", "低血鈣"),
    ("高血鈉", "低血鈉"),
    ("高尿酸", "低尿酸"),
    ("血壓偏高", "血壓偏低"),
    ("體溫過高", "體溫過低"),
    ("心跳過快", "心跳過慢"),
    ("視力良好", "視力不良"),
    ("聽力正常", "聽力異常"),
    ("已接種", "未接種"),
    ("已完成", "未完成"),
    ("有症狀", "無症狀"),
    ("規則服藥", "未規則服藥"),
    ("已轉診", "未轉診"),
    ("有運動習慣", "無運動習慣"),
    ("已受檢", "未受檢"),
    ("空腹", "飯後"),
    ("上升", "下降"),
    ("改善", "惡化"),
    ("增加", "減少"),
    ("正常", "異常"),
)

CATEGORIES: dict[str, tuple[str, ...]] = {
    "care_source": CARE_SOURCE,
    "care_followup": CARE_FOLLOWUP,
    "case_status": CASE_STATUS,
    "checkup_intent": CHECKUP_INTENT,
    "health_education": HEALTH_EDUCATION,
    "substance_use": SUBSTANCE_USE,
    "lab_value": LAB_VALUE,
    "contact": CONTACT,
}

CONCEPTS: dict[str, tuple[str, ...]] = {
    "delivery_method": DELIVERY_METHOD,
}


def rows() -> list[tuple[str, str]]:
    return [(term, category) for category, terms in CATEGORIES.items() for term in terms]


if __name__ == "__main__":
    total = sum(len(terms) for terms in CATEGORIES.values())
    print(f"{total} labelled field terms across {len(CATEGORIES)} categories")
    for category, terms in CATEGORIES.items():
        print(f"  {len(terms):4d}  {category}")
    print(f"  {len(REJECTION):4d}  (rejection)")
    print(f"  {len(DELIVERY_METHOD):4d}  delivery_method (concept layer)")
