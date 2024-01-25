import argparse
import pickle

import numpy as np
from mindspore import context
from mindspore.common.tensor import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.gpt import GPT
from src.pre_process import Tokenizer
from src.utils import GPTConfig

SEQ_LEN = 128
VOCAB_SIZE = 1928

# card: Ascend910
# context: The context of mindspore, used to configure the current execution environment, includes the execution mode, execution backend and other feature switches.

context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend", device_id=0)


def generate(prompt: str, tokenizer: Tokenizer, gpt: GPT, max_length=2000):
    """
    Text generation with pre-trained gpt model.
    
    Use only last SEQ_LEN input chars to generate the following words.
    And with constrain that input + output <= max_length.
    
    To make the output stable, use upper/lowwer threshold while "randomly" choosing the next output.
        p[0] > upper thres: choose
        p[chosen] < lowwer thres: retry until 10 times of bad luck 
            -> to avoid a very low probability of dead loop.
    """
    TOPK = 5
    upper_threshold = 0.5
    lowwer_threshold = 0.1
    input_ids = [tokenizer.stoi[ch] for ch in prompt]

    valid_length = len(input_ids)
    while valid_length < max_length:
        # print("Starting an inference loop.")
        # Use only last SEQ_LEN input chars to generate the following words.
        if len(input_ids) > SEQ_LEN:
            input_ids = input_ids[-SEQ_LEN:]
        inputs = Tensor(np.array([input_ids], dtype=np.int32))
        logits = gpt(inputs)
        # print(len(logits)) : 2
        # output: logits, loss.
        logits = logits[0].asnumpy() # get output and transform to numpy array
        probs = logits[-1, :] # get the probs of last word
        
        # sort and get the top 10 probs
        p_args = probs.argsort()[::-1][:TOPK]
        p = probs[p_args]
        p = np.exp(p) / np.sum(np.exp(p))
        # for i in range(len(p)):
        #     prod = int(p_args[i])
        #     print(tokenizer.itos[prod], end="", flush=True)
        #     print(p[i], end="", flush=True)
        # print("")
        
        target_index = 0
        if p[target_index] < upper_threshold:
            target_index = np.random.choice(len(p), p=p)
            cnt = 0
            while p[target_index] < lowwer_threshold and cnt <= 10:
                target_index = np.random.choice(len(p), p=p)
                cnt += 1
        
        prod = int(p_args[target_index])

        input_ids.append(prod)
        print(tokenizer.itos[prod], end="", flush=True)

        valid_length += 1
    print("")


def continuation(tokenizer: Tokenizer, gpt: GPT, max_length=1024):
    """Using GPT for fiction continuation.

    Args:
        gpt (nn.Cell): GPT model
        max_length (int): max generating length
    """
    print('Continuing the text in the style of Jin Yong\'s novels. Press "Ctrl+D" to exit.')
    
    while True:
        try:
            print("输入一个开头：", end="")
            prompt = input()
            generate(prompt, tokenizer, gpt)

        except EOFError:
            print("\nBye!")
            break


def main():
    local_config = {
        "ckpt_path": "GPT_baseline.ckpt"
    }

    ckpt_path = local_config["ckpt_path"]

    config = GPTConfig(
        batch_size=1,
        seq_length=SEQ_LEN,
        vocab_size=VOCAB_SIZE, # from the pre-process result
        embedding_size=128,
        num_layers=4,
        num_heads=8,
        expand_ratio=4,
        post_layernorm_residual=False,
        dropout_rate=1.0,
        use_past=False,
    )
    ckpt_dict = load_checkpoint(ckpt_path)

    gpt = GPT(config)

    gpt.set_train(False)
    load_param_into_net(gpt, ckpt_dict)

    with open("./dataset/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    # do inference: continue a novel with Jin Yong style
    continuation(tokenizer=tokenizer, gpt=gpt)


if __name__ == "__main__":
    main()
