import unicodedata
import torch

#-\'.ü!"#%&()*+,/:;?
Alphabets = {
    #'!#&():;?*%'
    'all': '` ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789\'-"/,.+_!#&():;?*',
    'iam_word': '` ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789\'-"/,.+_!#&():;?*',
    'iam_line': '` ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789\'-"/,.+_!#&():;?',
    'cvl_word': '` ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789\'-"/,.+_!#&():;?',
    'custom': '` ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789\'-"/,.+_!#&():;?',
    # 'cvl_word': '` ABDEFGHILNPRSTUVWYZabcdefghiklmnopqrstuvwxyz\'-_159', # n_class: 52
    'rimes_word': '` ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789%\'-/Éàâçèéêëîïôùû',
    'hindi_word': '',  # Hindi uses JSON vocabulary file instead
}


def _load_hindi_alphabet():
    """
    Build Hindi alphabet string from data/hindi_char2idx.json.
    Returns a string like '` ' + all_unique_hindi_chars.
    """
    import json
    import os

    json_path = os.path.join('data', 'hindi_char2idx.json')
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            char2idx = json.load(f)
        # Sort by index to keep consistent order
        chars_sorted = [ch for ch, idx in sorted(char2idx.items(), key=lambda x: x[1])]
        return '` ' + ''.join(chars_sorted)

    # Fallback
    raise RuntimeError("Could not find Hindi vocab files in ./data (hindi_char2idx.json).")

class strLabelConverter(object):
    """Convert between str and label.

    NOTE:
    Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet_key, ignore_case=False):
        # Special handling for Hindi - load from JSON
        if alphabet_key == 'hindi_word':
            alphabet = _load_hindi_alphabet()
        else:
            alphabet = Alphabets[alphabet_key]

        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()

        self.alphabet = alphabet
        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i

    def encode(self, text, max_len=None):
        """Support batch or single str."""
        if len(text) == 1:
            text = text[0]

        if isinstance(text, str):
            text = [
                self.dict[char.lower() if self._ignore_case else char]
                for char in text
            ]
            return text

        length = []
        results = []
        for item in text:
            length.append(len(item))
            result = [self.dict[char] for char in item]
            results.append(result)

        labels = torch.nn.utils.rnn.pad_sequence(
            [torch.LongTensor(text) for text in results],
            batch_first=True
        )
        lengths = torch.IntTensor(length)

        if max_len is not None and max_len > labels.size(-1):
            pad_labels = torch.zeros((labels.size(0), max_len)).long()
            pad_labels[:, :labels.size(-1)] = labels
            labels = pad_labels

        return labels, lengths

    def decode(self, t, length=None, raw=True):
        """Decode encoded texts back into strs."""
        import numpy as np

        def nonzero_count(x):
            if isinstance(x, torch.Tensor):
                return (x != 0).sum().item()
            return len([i for i in x if i != 0])

        if isinstance(t, list):
            t = torch.IntTensor(t)
            length = torch.IntTensor([len(t)])
        elif length is None:
            length = torch.IntTensor([nonzero_count(t)])

        # Convert tensor to numpy array
        if isinstance(t, torch.Tensor):
            t_array = t.cpu().numpy() if t.is_cuda else t.numpy()
        else:
            t_array = np.array(t)

        # Handle batch dimension
        if len(t_array.shape) == 2:
            # Batch mode: decode multiple sequences
            if isinstance(length, torch.Tensor):
                length = length.cpu().numpy() if length.is_cuda else length.numpy()
            elif not isinstance(length, (list, np.ndarray)):
                length = [length] * len(t_array)

            texts = []
            for i in range(len(t_array)):
                seq = t_array[i]
                seq_len = int(length[i]) if i < len(length) else len(seq)

                if raw:
                    text = ''.join([self.alphabet[int(idx)] for idx in seq[:seq_len]])
                else:
                    char_list = []
                    for j in range(seq_len):
                        idx = int(seq[j])
                        if idx != 0 and (j == 0 or seq[j-1] != seq[j]):
                            char_list.append(self.alphabet[idx])
                    text = ''.join(char_list)
                texts.append(text)

            return texts

        else:
            # Single sequence
            if isinstance(length, torch.Tensor) and length.numel() == 1:
                length = length.item()

            if raw:
                return ''.join([self.alphabet[int(i)] for i in t_array])
            else:
                char_list = []
                for i in range(int(length)):
                    idx = int(t_array[i])
                    if idx != 0 and (i == 0 or t_array[i - 1] != t_array[i]):
                        char_list.append(self.alphabet[idx])
                return ''.join(char_list)





def get_true_alphabet(name):
    """
    Get alphabet for dataset.
    For Hindi, loads characters dynamically from vocabulary file.
    """
    tag = '_'.join(name.split('_')[:2])

    if tag == 'hindi_word':
        # Load Hindi alphabet from vocabulary JSON
        import json
        try:
            vocab_path = './data/hindi_char2idx.json'
            with open(vocab_path, 'r', encoding='utf-8') as f:
                char2idx = json.load(f)

            # Extract actual characters (exclude special tokens)
            special_tokens = {'<PAD>', '<SOS>', '<EOS>', '<UNK>', '<BLANK>'}
            chars = [ch for ch in char2idx.keys() if ch not in special_tokens]
            alphabet = ''.join(sorted(chars))

            print(f"✓ Loaded Hindi alphabet: {len(chars)} characters")
            return alphabet
        except FileNotFoundError:
            print(f"Warning: {vocab_path} not found, returning empty alphabet")
            return ''
        except Exception as e:
            print(f"Error loading Hindi alphabet: {e}")
            return ''

    # For other datasets, use predefined alphabets
    return Alphabets[tag]



def get_lexicon(path, true_alphabet, max_length=20, ignore_case=True):
    """
    Load lexicon from file.
    For Hindi, skip alphabet filtering since JSON vocab handles it.
    """
    words = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                if len(line) < 1:  # Changed from < 2 for Hindi single-char words
                    continue

                # For Hindi, skip alphabet filtering if true_alphabet is empty
                if true_alphabet == '':
                    # Hindi mode: accept all characters from file
                    word = line
                else:
                    # Original mode: filter by alphabet
                    word = ''.join(ch for ch in line if ch in true_alphabet)
                    if len(word) != len(line):
                        continue

                if len(word) >= max_length:
                    continue

                # Only lowercase for Latin scripts
                if ignore_case and true_alphabet and ord(true_alphabet[0]) < 256:
                    word = word.lower()

                words.append(word)

        print(f"✓ Loaded lexicon: {len(words)} words from {path}")
    except FileNotFoundError as e:
        print(f"Error loading lexicon: {e}")
    return words



def word_capitalize(word):
    word = list(word)
    word[0] = unicodedata.normalize('NFKD', word[0].upper()).encode('ascii', 'ignore').decode("utf-8")
    word = ''.join(word)
    return word
