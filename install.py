from deeppavlov import configs, build_model

ner_model = build_model(configs.ner.ner_ontonotes_bert_torch, download=True)
ner_model = build_model(configs.ner.ner_ontonotes_bert_mult_torch, download=True)
