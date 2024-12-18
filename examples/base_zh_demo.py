# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
This basic example loads a pre-trained model from the web and uses it get entities.
"""

import sys

sys.path.append('..')
from nerpy import NERModel

if __name__ == '__main__':
    # BertSoftmax中文实体识别模型: NERModel("bert", "shibing624/bert4ner-base-chinese")
    # BertSpan中文实体识别模型: NERModel("bertspan", "shibing624/bertspan4ner-base-chinese")
    model = NERModel("bert", "shibing624/bert4ner-base-chinese")
    sentences = [
        "常建良，男，1963年出生，工科学士，高级工程师，北京物资学院客座副教授",
        "中華民國民眾黨主席柯文哲涉政治獻金假帳案，調查局北機站清查金流發現，民眾黨利用「網紅帶貨」銷售手法，先藉由「學姐」黃瀞瑩等人高知名度，吸引選民捐贈政治獻金，再用「折扣碼」發放KP競選小物，進而從中抽佣分潤，抽佣的錢疑來自政治獻金，涉及違反政治獻金法，黃瀞瑩與「戰狼小姐姐」陳智菡、許甫、吳怡萱等4人恐由證人轉列被告偵辦",
        "李明在上海的騰訊公司擔任工程師"
    ]
    # set split_on_space=False if you use Chinese text
    predictions, raw_outputs, entities = model.predict(sentences, split_on_space=False)
    print(predictions, entities)

    # More detailed predictions
    for n, (preds, outs) in enumerate(zip(predictions, raw_outputs)):
        print("\n___________________________")
        print("Sentence: ", sentences[n])
        print("Entity: ", entities[n])
