import matplotlib.pyplot as plt

from torchtext.legacy import data
from torchtext.legacy import datasets
TEXT = data.Field(lower=True)
UD_TAGS = data.Field(unk_token=None)
fields = (("text", TEXT), ("udtags", UD_TAGS))
train_data, valid_data, test_data = datasets.UDPOS.splits(fields)
def visualizeSentenceWithTags(example):
    print("Token" + "".join([" "]*(15)) + "POS Tag")
    print(" ---------------------------------")
    for w, t in zip(example['text'], example['udtags']):
        print(w + "". join([" "]*(20 - len (w))) + t)

visualizeSentenceWithTags(vars(train_data . examples[997]))
# import pdb; pdb.set_trace()
data_dump = '\n'.join(entry.text )
# for i in range(1000):
#     print(f'{i}: ' + ' '.join(train_data.examples[i].text))
#     print(f'{i}: ' + ' '.join(train_data.examples[i].udtags))

pos_tags = {}
for i in range(len(train_data.examples)):
    for pos in train_data.examples[i].udtags:
        if pos in pos_tags:
            pos_tags[pos] += 1
        else:
            pos_tags[pos] = 1

# import pdb; pdb.set_trace()
# plt.hist(pos_tags)
# plt.bar(pos_tags.keys(), pos_tags.values())
plt.pie(pos_tags.values(), labels=pos_tags.keys())
plt.show()
