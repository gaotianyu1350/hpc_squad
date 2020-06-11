import json
import transformers
from transformers import BertTokenizer
from tqdm import tqdm

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
break_cnt = 0
data = json.load(open('./squad_data/train-v1.1.json'))
f = open('squad_train_text.txt', 'w')
for item in tqdm(data['data']):
    for subitem in item['paragraphs']:
        context = subitem['context']
        f.write(context.strip() + '\n')
        tot_word = len(tokenizer.tokenize(context))
        if tot_word > 512:
            break_cnt += 1
print(break_cnt)

