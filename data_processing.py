import re
from typing import List, Union
from tqdm.auto import tqdm

def split_batch(iterable, n: int):
    return [
        iterable[ndx:ndx+n] for ndx in range(0, len(iterable), n)
    ]

def mask_idx(text: str, idx: Union[int, List[int]]) -> str:
    text = list(text)

    if type(idx) == int: idx = [idx]
    for i in idx:
        text[i] = '[MASK]'

    return ''.join(text)
        
def mask_homo(sents_origin: List[str]):
    sents_masked, m_idx = [], []
    for sent in tqdm(sents_origin, desc='Masking homoglyph chararcters'):
        m_id =[]
        sent = get_clean_text(sent.lower())
        for i in range(len(sent)):
            if ord(sent[i]) not in range(97, 123) and ord(sent[i]) != 32 :
                m_id.append(i)

        sents_masked.append(mask_idx(sent, m_id))
        m_idx.append(m_id)
    return sents_masked


def get_clean_text(s):
    s = str(s)
#    print(s)
    s = re.sub(r'[\u200a-\u200f\u202a-\u202d\u2068]', '', s) # invisible space
    s = re.sub(r'[\ua7d4\ufeff]', '', s)                     # invisible space
    s = s.replace('b\xad','')
    s = s.replace('\\xad|\\xad\\xad','')
    s = re.sub(r'(bc1|[13])[a-zA-HJ-NP-Z0-9]{25,39}', '', s) # bitcoin address
    s = re.sub(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)', '', s) # url    
    s = re.sub(r'(?:[a-z0-9!#$%&\'*+/=?^_`{|}~-]+(?:\.[a-z0-9!$%&\'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])', '', s) # email address
    s = re.sub(r'(?:[a-z0-9!#$%&\'*+/=?^_`{|}~-]+(?:\.[a-z0-9!$%&\'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")_at_(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])', '', s) # email address
    #\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b
    #[0-9a-zA-Z-_.]+@[0-9a-zA-Z_.]+\.\w+
    #[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+
    
    # s = re.sub(r'[^\x00-\x7F]', ' ', s) # out of ASCII - homoglyph까지 날아감 
    # s = re.sub(r'[^\u0041-\u007A]', ' ', s) # out of ASCII - homoglyph까지 날아감 
    s = re.sub(r'[\u2060-\u2069]', '', s) # simbol
    s = re.sub(r'[\u005B-\u0060]', '', s) # simbol
    s = re.sub(r'[\u02B9-\u0371\u0378-\u0385\u0387]', '', s) # simbol
    s = re.sub(r'[\u0588-\u05F0\u05F2-\u109F]', '', s) # simbol
    s = re.sub(r'[\u2120-\u2122\u2136\u2137\u213B-\u2144\u214A-\u216B]', '', s) # simbol
    s = re.sub(r'[\u2000-\u208F\u2100\u2101]', '', s) # simbol
    s = re.sub(r'[\u2187-\u22F1]', '', s) # simbol
    s = re.sub(r'[\u2300-\u249B\u2500-\u2C5F\u2CBA]', '', s) # simbol
    s = re.sub(r'[🏆🏦💋💯📈😂😉😎😜😤😪🙃🙈🤣🤦🦴]', '', s) # simbol
    # s = re.sub(r'[\u0000-\u0040\u005B-\u0060\u0080-\u0089\u008B-\u008D\u008F-\u0099\u009B-\u009D\u00A0\u00A4\u00A6-\u00A9]', '', s) # Unicode of nonASCII
    # s = re.sub(r'[\u00AB-\u00B4\u00B6-\u00BF\u002A0-\u0371\u0374-\u0375\u0378-\u0385\u0386\u038A\u0386D\u038F\u0504-\u0509]', '', s) # Unicode of nonASCII
    # s = re.sub(r'[\u0512-\u0519\u0520-\u0530\u0557-\u0560\u0588-\u05F0\u05F2-\u109F]', '', s) # Unicode of nonASCII
    
#     s = re.sub(r'[^\u00C0-\u029F\u0372-\u0373\u0376-\u0377\u0386\u0388-\u038A\u038C\u038E\u0390-\u0503\u050A-\u0511]', '[MASK]', s) ### Unicode of Homoglyph
#     s = re.sub(r'[^\u051A-\u051F\u0531-\u0556\u0561-\u0587\u05F0\u05F1\u0388-\u038A\u038C\u038E\u0390-\u0503\u050A-\u0511]', '[MASK]', s) ### Unicode of Homoglyph
    
    s = re.sub(r'[\u2D60-\uA725\uA800-\uFF20\uFF5B-\uFFFF\u1780-\u1C86\u2010-\u208F]', '', s) # Unicode of etc
    s = re.sub(r'[\u1100-\u11FF\u3131-\u318F\uAC00-\uD800]', '', s) # Unicode of Korean
    #s = re.sub(r'[\u3040-\u309F\u30A0-\u30FF\u31F0-\u31FF]', '', s) # Unicode of Japanese
    s = re.sub(r'[\x00-\x40]', ' ', s) # non alphabet ASCII
    s = re.sub(r'[\x5e-\x60]', ' ', s) # non alphabet ASCII
    s = re.sub(r'[\x7b-\x7f]', ' ', s) # non alphabet ASCII
    s = re.sub(r'[\x81-\x9f]', ' ', s) # non alphabet ASCII
    s = re.sub(r'[\s]', ' ', s) # 공백문자
    s = re.sub(r'[\t\n\r]', ' ', s) #tab, newline, return 위로 가야하나..?
    s = re.sub(r'[¢£¤¥¦§¨©ª«¬®¯°±²³´¶·¸¹º»¼½¾¿]', '', s)
    s = re.sub(r'[Зლ₩€Ｉ𝟐𝟒𝟕]', '', s)
    s = re.sub(r'[\u1D7D0]', '', s)
    
    
    
#     s = re.sub(r'[\u2E80-\u2EFF\u3400-\u4DBF\u4E00-\u9FBF\uF900-\uFAFF\u20000-\u2A6DF\u2F800-\u2FA1F]','', s) # Unicode of Chinese
    
    #s = re.sub(r'[\d]', '', s) # 숫자는 공백없이
    s = s.split()
    s = ' '.join(s)
    return s