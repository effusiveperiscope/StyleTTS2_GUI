import gruut
from gruut.const import default_split_words
from pathlib import Path
import re
import eng_to_ipa as ipa
from log import logger
import string
CMU_IPA_MAPPING = {
    "AA": "ɑ",
    "AE": "æ",
    "AH": "ə",
    "AO": "ɔ",
    "AW": "aʊ",
    "AY": "aɪ",
    "B": "b",
    "CH": "ʧ",
    "D": "d",
    "DH": "ð",
    "EH": "ɛ",
    "ER": "ɝ",
    "EY": "eɪ",
    "F": "f",
    "G": "ɡ",
    "HH": "h",
    "IH": "ɪ",
    "IY": "i",
    "JH": "dʒ",
    "K": "k",
    "L": "l",
    "M": "m",
    "N": "n",
    "NG": "ŋ",
    "OW": "oʊ",
    "OY": "ɔɪ",
    "P": "p",
    "R": "r",
    "S": "s",
    "SH": "ʃ",
    "T": "t",
    "TH": "θ",
    "UH": "ʊ",
    "UW": "u",
    "V": "v",
    "W": "w",
    "Y": "j",
    "Z": "z",
    "ZH": "ʒ"
}
CMU_VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY",
    "EH", "ER", "EY", "IH", "IY",
    "OW", "OY", "UH", "UW"}
HORSEWORDS_DICTIONARY = 'horsewords.clean'

def is_cmu_vowel(token):
    return token in CMU_VOWELS

def is_cmu_consonant(token):
    return not is_cmu_vowel(token)

def load_dictionary(dict_path, remove_nums=True):
    arpadict = dict()
    with open(dict_path, "r") as f:
        for line in f.readlines():
            word = line.split("  ")
            assert len(word) == 2
            maps_to = word[1].strip()
            if remove_nums:
                maps_to = re.sub(r'\d+', '', maps_to)
            arpadict[word[0].strip().upper()] = maps_to
    return arpadict

exceptions_dictionary = load_dictionary(HORSEWORDS_DICTIONARY)

def arpabet_to_ipa(arpabet : str):
    return ''.join(CMU_IPA_MAPPING[token] for token in arpabet.split(' '))

# Only stresses up to 2 are allowed.
arpa_pattern = re.compile(r'([A-Z]+)([0-2])?')
def arpabet_toks_to_ipa(arpabet : list):
    max_stress = None
    max_stress_i = None
    sec_stress = None
    sec_stress_i = None
    
    tokens_info = []
    for i, token in enumerate(arpabet):
        re_data = arpa_pattern.match(token)
        if not re_data:
            raise ValueError(f"Bad ARPABET pattern {token}")
        (arpa_key, arpa_stress) = (re_data[1], re_data[2])
        tokens_info.append({'stress': 0, 'key': arpa_key})
        if arpa_stress is not None:
            arpa_stress = int(arpa_stress)
            if arpa_stress == 1:
                max_stress = token
                max_stress_i = i
            if arpa_stress == 2:
                sec_stress = token
                sec_stress_i = i

    if max_stress is not None:
        i = max_stress_i
        # seek backwards to the nearest consonant
        while is_cmu_vowel(arpabet[i]) and i != 0:
            i -= 1
        tokens_info[i]['stress'] = 1 # primary stress
    if sec_stress is not None:
        i = sec_stress_i
        # seek backwards to the nearest consonant
        while is_cmu_vowel(arpabet[i]) and i != 0:
            i -= 1
        tokens_info[i]['stress'] = 2 # secondary stress

    ipa_out = ""
    for tok in tokens_info:
        if tok['stress'] == 1:
            ipa_out += 'ˈ'
        elif tok['stress'] == 2:
            ipa_out += 'ˌ'
        ipa_out += CMU_IPA_MAPPING[tok['key']]
        #ipa_out += ' '
    
    return ipa_out

def _chunk_conv_to_ipa(text : str):
    output = []
    for word in default_split_words(text):
        #if word.upper() in exceptions_dictionary:
        #    ipa_sub = arpabet_to_ipa(exceptions_dictionary[word.upper()])
        #    output.append(ipa_sub)
        sentence = gruut.sentences(word)
        for sent in sentence:
            for word in sent.words:
                if not word.is_spoken:
                    output.append(word.text)
                elif word.phonemes:
                    word_str = ''.join(word.phonemes)
                    output.append(word_str)
        return ''.join(output)

def conv_to_ipa(text : str, print_unhandled : bool = False,
    normalize : bool = False):
    output = []
    # 1. Check if all words are in CMU. If so, just use gruut
    phoneme_join = ' ' if normalize else ''
    if ipa.isin_cmu(text):
        sentence = gruut.sentences(text)
        for sent in sentence:
            for word in sent.words:
                if not word.is_spoken:
                    output.append(word.text)
                elif word.phonemes:
                    word_str = ''.join(word.phonemes)
                    if normalize:
                        word_str = phoneme_join.join(word_str)+' ▁'
                    output.append(word_str)
        return ' '.join(output)

    # 2. If not, handle each word individually
    for word in default_split_words(text):
        if word.upper() in exceptions_dictionary:
            ipa_sub = arpabet_to_ipa(exceptions_dictionary[word.upper()])
            if normalize:
                ipa_sub = phoneme_join.join(ipa_sub)
            output.append(ipa_sub)
        else:
            if print_unhandled and not ipa.isin_cmu(text):
                print("Found word not in dictionary "+word)
                print("Using gruut fallback")
            sentence = gruut.sentences(word)
            for sent in sentence:
                for word in sent.words:
                    if not word.is_spoken:
                        output.append(word.text)
                    elif word.phonemes:
                        word_str = ''.join(word.phonemes)
                        if normalize:
                            word_str = phoneme_join.join(word_str)
                        output.append(word_str)
    output = ' '.join(output)
    return output

# for dirtier sources; implements arpabet escapes
from nltk import sent_tokenize, word_tokenize
def conv_to_ipa2(text : str,
    normalize : bool = False,
    do_arpa_escapes : bool = True):
    transcription = []
    sentences = sent_tokenize(text)
    total_ipa = ''
    for sentence in sentences:
        chunks = []
        words = word_tokenize(sentence)
        cur_chunk = []
        arpabet_escape = False
        arpabet_chunk = []
        for w in words:
            if w == '{' and do_arpa_escapes:
                if len(cur_chunk):
                    chunks.append(('cmu', ' '.join(cur_chunk)))
                    cur_chunk = []
                arpabet_escape = True
                continue
            elif w == '}' and do_arpa_escapes:
                arpabet_escape = False
                chunks.append(('ipa',arpabet_toks_to_ipa(arpabet_chunk)))
                arpabet_chunk = []
                continue
            elif not w.isalnum():
                if len(cur_chunk):
                    chunks.append(('cmu', ' '.join(cur_chunk)))
                    cur_chunk = []
                if all(c in string.punctuation for c in w):
                    chunks.append(('sym', w))
                else:
                    chunks.append(('exc', w))
                continue
            if arpabet_escape and w != '{' and do_arpa_escapes:
                arpabet_chunk.append(w)
                continue
            if not ipa.isin_cmu(w):
                if len(cur_chunk):
                    chunks.append(('cmu', ' '.join(cur_chunk)))
                    cur_chunk = []
                if w.upper() in exceptions_dictionary:
                    ipa_sub = arpabet_to_ipa(exceptions_dictionary[w.upper()])
                    chunks.append(('ipa', ipa_sub))
                else: # eventually acronym handling will have to go here
                    chunks.append(('exc', w.lower()))
            else: # this word is known to be in the CMU dictionary
                cur_chunk.append(w)
        if len(cur_chunk):
            chunks.append(('cmu', ' '.join(cur_chunk)))
            cur_chunk = []

        output = []
        phoneme_join = ' ' if normalize else ''
        for chunk in chunks:
            if chunk[0] == 'cmu' or chunk[0] == 'exc':
                sentence = gruut.sentences(chunk[1].lower())
                for sent in sentence:
                    for word in sent.words:
                        if not word.is_spoken:
                            output.append(word.text)
                        elif word.phonemes:
                            word_str = ''.join(word.phonemes)
                            if normalize:
                                word_str = phoneme_join.join(word_str)+' ▁'
                            output.append(word_str)
            elif chunk[0] == 'sym' or chunk[0] == 'ipa':
                output.append(chunk[1]+' ')
        sentence_ipa = ' '.join(output)+' '
        #print(sentence_ipa)
        total_ipa += sentence_ipa
    return total_ipa

from enum import Enum
class TokenizerMode(Enum):
    DEFAULT = 0,
    IPA = 1,
    ARPABET = 2

# nltk's built-in word tokenizer is broken too
def conv_to_ipa3(text):
    chunk = ""
    output = ""
    word_count = 0

    mode = TokenizerMode.DEFAULT
    for i in range(len(text)):
        c = text[i]

        if mode == TokenizerMode.DEFAULT:
            if c in '.!?':
                if len(chunk):
                    output += _chunk_conv_to_ipa(chunk)
                    word_count += 1
                    chunk = ""
                output += ' '+c
                continue
            elif c in string.whitespace:
                if len(chunk):
                    output += _chunk_conv_to_ipa(chunk)
                    word_count += 1
                    chunk = ""
                output += c
                continue
            elif c == '<':
                if len(chunk):
                    output += _chunk_conv_to_ipa(chunk)
                    word_count += 1
                    chunk = ""
                mode = TokenizerMode.IPA
                continue
            elif c == '{':
                if len(chunk):
                    output += _chunk_conv_to_ipa(chunk)
                    word_count += 1
                    chunk = ""
                mode = TokenizerMode.ARPABET
                continue
            else:
                chunk += c
        elif mode == TokenizerMode.IPA:
            if c == '>':
                mode = TokenizerMode.DEFAULT
                word_count += 1
                chunk = ""
                continue
            output += c
        elif mode == TokenizerMode.ARPABET:
            if c == '}':
                mode = TokenizerMode.DEFAULT
                output += arpabet_toks_to_ipa(chunk.split(' '))
                word_count += 1
                chunk = ""
                continue
            chunk += c
    if mode != TokenizerMode.DEFAULT:
        if mode == TokenizerMode.IPA:
            add = "Unmatched IPA escape bracket <"
        elif mode == TokenizerMode.ARPABET:
            add = "Unmatched ARPABET escape bracket {"
        raise ValueError(f"Failed IPA conversion, {add}")
    if len(chunk):
        output += _chunk_conv_to_ipa(chunk)
    #print(word_count)
    return output, word_count