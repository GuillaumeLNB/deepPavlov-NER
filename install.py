from deeppavlov import configs, build_model

# new way
ner_model = build_model(configs.ner.ner_ontonotes_bert, download=True)
ner_model = build_model(configs.ner.ner_ontonotes_bert_mult, download=True)

# old way
# ner_model = build_model(configs.ner.ner_ontonotes_bert_torch, download=True)
# ner_model = build_model(configs.ner.ner_ontonotes_bert_mult_torch, download=True)
