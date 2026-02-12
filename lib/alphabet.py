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




class strLabelConverter(object):
    """Convert between str and label.
    NOTE:
        Insert `blank` to the alphabet for CTC.
    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet_key, ignore_case=False):
        alphabet = Alphabets[alphabet_key]
        # print(alphabet)
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i

    def encode(self, text, max_len=None):
        """Support batch or single str.
        Args:
            text (str or list of str): texts to convert.
        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        if len(text) == 1:
            text = text[0]

        if isinstance(text, str):
            text = [
                self.dict[char.lower() if self._ignore_case else char]
                for char in text
            ]
            return text

        length = []
        result = []
        results = []
        for item in text:
            # item = item.decode('utf-8', 'strict')
            length.append(len(item))
            for char in item:
                index = self.dict[char]
                result.append(index)
            results.append(result)
            result = []

        labels = torch.nn.utils.rnn.pad_sequence([torch.LongTensor(text) for text in results], batch_first=True)
        lengths = torch.IntTensor(length)

        if max_len is not None and max_len > labels.size(-1):
            pad_labels = torch.zeros((labels.size(0), max_len)).long()
            pad_labels[:, :labels.size(-1)] = labels
            labels = pad_labels

        return labels, lengths

    def decode(self, t, length=None, raw=True):
        """Decode encoded texts back into strs.
        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        Raises:
            AssertionError: when the texts and its length does not match.
        Returns:
            text (str or list of str): texts to convert.
        """
        def nonzero_count(x):
            return len(x.nonzero())

        if isinstance(t, list):
            t = torch.IntTensor(t)
            length = torch.IntTensor([len(t)])
        elif length is None:
            length = torch.IntTensor([nonzero_count(t)])

        if length.numel() == 1:
            length = length[0]
            assert nonzero_count(t) == length, "{} text with length: {} does not match declared length: {}".\
                                                format(t, nonzero_count(t), length)
            if raw:
                return ''.join([self.alphabet[i] for i in t])
            else:
                char_list = []
                if t.dim() == 2:
                    t = t[0]
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i]])
                return ''.join(char_list)
        else:
            # batch mode
            assert nonzero_count(t) == length.sum(), "texts with length: {} does not match declared length: {}".\
                                                      format(nonzero_count(t), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[i, :l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts


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
