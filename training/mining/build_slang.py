#!/usr/bin/env python3
"""Build the medical-slang table from the harvested public sources.
Two layers:
  spoken  -- 交班/晨會 verbal jargon (陳志金 blog / vocus). ASR-relevant.
  written -- clinical shorthand that appears in records (udn ptsafetyrm).
             import-relevant too.
Columns: slang | origin | meaning | layer | source
"""
import csv
import os

# layer=spoken: 陳志金「巷子內醫療用語」 + vocus 魔法密語 (same canonical corpus)
SPOKEN = [
    ("摸咪/摸咪挺", "morning meeting", "晨會"),
    ("挨毆", "I/O", "水份進出身體總和"),
    ("偷/偷挨毆", "Total I/O", "水份進出總和"),
    ("內ㄋㄟ多少", "Negative", "水份負平衡"),
    ("潑ㄆㄛ多少", "Positive", "水份正平衡"),
    ("掐/掐水", "fluid challenge", "快速輸液"),
    ("吸ㄟ", "Cancer", "癌症"),
    ("摳龍吸ㄟ", "Colon cancer", "大腸癌"),
    ("ㄎㄧˉ莫", "Chemo(therapy)", "化療"),
    ("咻", "Suture", "縫合"),
    ("妞胚", "New patient", "新病人"),
    ("很馬", "Malignant", "惡性腫瘤／態度惡劣"),
    ("英騰", "Intern", "實習醫師"),
    ("好ㄔㄨㄚ", "Trouble", "很忙／病情嚴重"),
    ("飛ㄇ哩", "Family", "病人家屬"),
    ("很ㄇ", "Murmur", "愛碎碎唸"),
    ("殺一下", "Suction", "抽痰"),
    ("阿浪", "Alarm", "監視器警報"),
    ("滂", "Puncture", "穿刺"),
    ("滂嘎死", "Puncture Gas", "抽動脈血"),
    ("賽奇", "Psychic", "精神疾病"),
    ("灰累", "Failure", "處置失敗"),
    ("AP", "Antepartum", "產前／懷孕中"),
    ("Portable", "Portable X-Ray", "移動式X光"),
    ("發漏一下", "Follow", "追蹤"),
    ("杯葛", "Bag", "點滴袋"),
    ("IV彿拉噓", "IV Flush", "導管沖液"),
    ("咖", "Culture", "培養"),
    ("U咖", "Urine culture", "尿液培養"),
    ("不辣咖", "Blood culture", "血液培養"),
    ("掰一下", "Biopsy", "切片"),
    ("Bonjour", "(台語)", "碎石術"),
    ("客羅特", "Clot", "檢體凝固"),
    ("牛肉西施", "Neurosis", "焦慮傾向"),
    ("史咖逼", "Scabies", "疥瘡"),
    ("漏屎", "Loss", "靜脈導管不通"),
    ("狹客", "Shock", "休克"),
    ("做浪吧", "Lumbar puncture", "腰椎穿刺"),
    ("尾錐欣", "Wet dressing", "濕敷"),
    ("歐卡", "OHCA", "到院前心跳停止"),
    ("豆一下EKG", "EKG", "接心電圖監視"),
    ("滴頭", "Ditto", "同前一次"),
    ("噢鼻屎", "OP site", "透明敷料"),
    ("妹塔", "Meta", "癌轉移／代謝"),
]

# layer=written: udn 詹廖明義「台灣醫療術語的溝通問題」 clinical shorthand
WRITTEN = [
    ("很Tra", "Troublesome", "難搞的病人"),
    ("New Pe", "New Patient", "新入院病人"),
    ("DOA", "Dead On Arrival", "到院前死亡"),
    ("OHCA", "Out-of-Hospital Cardiac Arrest", "到院前心跳停止"),
    ("EMT", "Emergency Medical Technician", "救護技術員"),
    ("CPR", "Cardiopulmonary Resuscitation", "心肺復甦"),
    ("Endo", "Endotracheal tube", "氣管內管"),
    ("BP", "Blood Pressure", "血壓"),
    ("DOPA", "Dopamine", "升壓藥"),
    ("IV Bag", "IV Burette", "點滴容器"),
    ("Anti", "Antibiotic", "抗生素"),
    ("DNR", "Do Not Resuscitate", "不施行心肺復甦"),
    ("AAD", "Against-Advice Discharge", "自動出院"),
    ("NG", "Nasogastric tube", "鼻胃管"),
    ("Bite", "Bite Block", "牙墊"),
    ("Negadon", "Nelaton Catheter", "導尿管(音譯訛用)"),
    ("Foley", "Foley catheter", "留置導尿管"),
    ("麻姐", "Anesthesiologist", "麻醉醫師"),
]


def main():
    out = os.path.join(os.path.dirname(__file__), "med_slang.tsv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["slang", "origin", "meaning", "layer", "source"])
        for s, o, m in SPOKEN:
            w.writerow([s, o, m, "spoken", "chenzhijin/vocus"])
        for s, o, m in WRITTEN:
            w.writerow([s, o, m, "written", "udn-ptsafetyrm"])
    print(f"med_slang.tsv: {len(SPOKEN)} spoken + {len(WRITTEN)} written "
          f"= {len(SPOKEN)+len(WRITTEN)} rows")


if __name__ == "__main__":
    main()
