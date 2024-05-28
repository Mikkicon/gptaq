import numpy as np
import torch


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_wikitext2(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    from transformers import AutoTokenizer 
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_ptb(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')

    from transformers import AutoTokenizer 
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    import os
    train_file_path = os.path.join('data', 'c4-train.00000-of-01024.json.gz')
    val_file_path = os.path.join('data', 'c4-validation.00000-of-00008.json.gz')

    # traindata = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    # valdata = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
    traindata = load_dataset('json', data_files={'train': train_file_path}, split='train')
    valdata = load_dataset('json', data_files={'validation': val_file_path}, split='validation')

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    valenc = get_cached_valenc(tokenizer=tokenizer, valdata=valdata, seqlen=seqlen)
    valenc = torch.hstack(valenc)
    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc 

def get_ptb_new(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4_new(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model=''
):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model)
    if 'ptb' in name:
        if 'new' in name:
            return get_ptb_new(nsamples, seed, seqlen, model)
        return get_ptb(nsamples, seed, seqlen, model)
    if 'c4' in name:
        if 'new' in name:
            return get_c4_new(nsamples, seed, seqlen, model)
        return get_c4(nsamples, seed, seqlen, model)

def get_cached_valenc(tokenizer, valdata, seqlen):
    import random
    import os
    import pickle

    random.seed(0)
    valenc = []
    str_tokenizer = str(tokenizer)
    cache_file = f"models/facebook/{str_tokenizer[0:str_tokenizer.index('(')]}_cache.pkl"

    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
    else:
        cache = {}

    def get_tokenized_input(index):
        if index in cache:
            return cache[index]
        else:
            tmp = tokenizer(valdata[index]['text'], return_tensors='pt')
            cache[index] = tmp
            return tmp

    try:
        for _ in range(256):
            while True:
                i = random.randint(0, len(valdata) - 1)
                tmp = get_tokenized_input(i)
                if tmp.input_ids.shape[1] >= seqlen:
                    break
            i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            valenc.append(tmp.input_ids[:, i:j])
    finally:
        with open(cache_file, 'wb') as f:
            pickle.dump(cache, f)
    print(f"Processed {len(valenc)} sequences.")
    return valenc
