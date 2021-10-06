from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

def createTokenizer(files, vocab_size, min_freq, max_len, save_path):
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=files, vocab_size=vocab_size, min_frequency=min_freq, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
    ])
    tokenizer.save_model(save_path)
    tokenizer = ByteLevelBPETokenizer(save_path+"vocab.json", save_path+"merges.txt", )
    tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>")),
    )
    tokenizer.enable_truncation(max_length=max_len)

    tokenizer.save(save_path+"tokenizer.json")
    