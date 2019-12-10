import re

def process_sentence(sentence):
    """ Preprocess sentences
    Convert everything to lower case letters.
    Add space before punctuation.
    Remove special characters.
    Remove trailing and leading spaces.
    """
    # sentence = re.sub("[0-9]", "", sentence)
    # sentence = re.sub("endoftext", "", sentence)
    # sentence = unicode_to_ascii(sentence.lower().strip())
    sentence = re.sub("ä", "ae", sentence)
    sentence = re.sub("ö", "oe", sentence)
    sentence = re.sub("ü", "ue", sentence)
    sentence = re.sub("ß", "ss", sentence)
    sentence = re.sub(r"([,.!?])", r" \1", sentence)
    sentence = re.sub("[' ']{2,}", " ", sentence)
    sentence = re.sub("[^A-Za-z,.!?]+", " ", sentence)
    sentence = sentence.strip()
    return sentence

def load_de_en_translations(input_file, target_file, max_samples=None):
    inputs = []
    targets = []
    with open(input_file, "r", errors="ignore", encoding="utf-8") as fin_de:
        with open(target_file, "r", errors="ignore", encoding="utf-8") as fin_en:
            for inp, tar in zip(fin_de, fin_en):
                # inp = process_sentence(inp)
                # tar = process_sentence(tar)
                inputs.append(inp)
                targets.append(tar)
                if max_samples is not None and len(inputs) >= max_samples:
                    break
    return inputs, targets