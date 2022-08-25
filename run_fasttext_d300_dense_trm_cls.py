import torch
import numpy as np
import pandas as pd
import matchzoo as mz
import random
from pytorch_transformers import AdamW, WarmupLinearSchedule
print('start run_fasttext_d300_dense_trm ......')

SEED = 6666
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

ranking_task = mz.tasks.Ranking(losses=mz.losses.RankCrossEntropyLoss(num_neg=5))
ranking_task.metrics = [
    mz.metrics.MeanReciprocalRank(maxRank=100),
    mz.metrics.NormalizedDiscountedCumulativeGain(k=5)
]
print("`ranking_task` initialized with metrics", ranking_task.metrics)

device = [1, 0]
batch_size = 512

# load
preprocessor = mz.load_preprocessor('./full_data_results/basic_data_full/save/preprocessor')
train_pack_processed = mz.load_data_pack('./full_data_results/basic_data_full/save/train')
dev_pack_processed = mz.load_data_pack('./full_data_results/basic_data_full/save/dev')

print(train_pack_processed[:1].unpack()[0]['text_left'])
print(train_pack_processed[:1].unpack()[0]['text_right'])
print('load done!')

# dataset
trainset = mz.dataloader.Dataset(
    data_pack=train_pack_processed,
    mode='pair',
    num_dup=10,
    num_neg=5,
    batch_size=batch_size,
    resample=True,
    sort=False,
    shuffle=True,
)
devset = mz.dataloader.Dataset(
    data_pack=dev_pack_processed,
    mode='point',
    batch_size=batch_size,
    resample=False,
    sort=False,
    shuffle=True,
)

# dataloader
padding_callback = mz.models.Dense_Transformer_CLS.get_default_padding_callback(
    fixed_length_left=30,
    fixed_length_right=30,
    pad_word_mode='post'
)
trainloader = mz.dataloader.DataLoader(
    dataset=trainset,
    device=device,
    stage='train',
    num_workers=4,
    callback=padding_callback
)
devloader = mz.dataloader.DataLoader(
    dataset=devset,
    device=device,
    stage='dev',
    num_workers=4,
    callback=padding_callback
)

fasttext_embedding = mz.embedding.load_from_file(
    file_path='/data/users/caiyinqiong/.matchzoo/datasets/fasttext/wiki.qqp.en.vec',
    mode='fasttext')
term_index = preprocessor.context['vocab_unit'].state['term_index']
embedding_matrix = fasttext_embedding.build_matrix(term_index)
l2_norm = np.sqrt((embedding_matrix * embedding_matrix).sum(axis=1))
embedding_matrix = embedding_matrix / l2_norm[:, np.newaxis]

# model
model = mz.models.Dense_Transformer_CLS()
model.params['task'] = ranking_task
model.params['embedding'] = embedding_matrix
model.params['embedding_freeze'] = False

model.params['left_length'] = 30
model.params['right_length'] = 30

model.params['pos_embedding_indim'] = 30
model.params['pos_embedding_outdim'] = 300

model.params['nhead'] = 6
model.params['dim_ffl'] = 256
model.params['num_layers'] = 1   # #################

model.params['dropout_rate'] = 0.1
model.build()
print(model, sum(p.numel() for p in model.parameters() if p.requires_grad))
print('#################################################')

no_decay = ['bias', 'norm']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 5e-5},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
for group in optimizer_grouped_parameters:
    print(len(group['params']), group['weight_decay'])
print('#################################################')

optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5, betas=(0.9, 0.98), eps=1e-8)
scheduler = WarmupLinearSchedule(optimizer, warmup_steps=100, t_total=-1)
trainer = mz.trainers.Trainer(
    model=model,
    device=device,
    optimizer=optimizer,
    scheduler=scheduler,
    trainloader=trainloader,
    validloader=devloader,
    validate_interval=None,
    epochs=60,
    save_dir='./full_data_results/ablation_layer1/full-d300-Add-6666',
    save_all=True
)
trainer.run()
