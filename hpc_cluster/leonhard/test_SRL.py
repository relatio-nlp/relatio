from narrativeNLP.semantic_role_labeling import SRL

srl = SRL(
    "https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz",
    cuda_device=0,
)

res = srl(["This is a test."])
print(res)
