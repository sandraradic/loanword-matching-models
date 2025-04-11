import pandas as pd
import nltk 
import string
from rapidfuzz import process, fuzz

# Broad English Dictionary (not helpful in this study, used local dictionary instead)
nltk.download('punkt')

# Load the locally downloaded English dictionary (one word per line)
# English Dictionary used: en_US Hunspell Dictionary from SCOWL (Atkinson, 2020)
with open("en_US_clean.txt", "r", encoding="utf-8") as f:
    english_words = set(word.strip().lower() for word in f if word.strip())

# Load the locally downloaded Serbian dictionary (one word per line)
# Serbian dictionary used: public Serbian Latin dictionary uploaded by Github user titoBouzout (Mihajlović, 2015)
with open("serbian_words.txt", "r", encoding="utf-8") as f:
    serbian_words = set(word.strip().lower() for word in f if word.strip())

# List of common English acronyms that Model 2 will reference
acronyms = {"lol", "wtf", "omg", "btw", "rip", "lmao", "lmfao", "rofl", "idk", "brb", "gtg", "imo", "fyi", "smh", "tbh", "tfw", "wyd", "yt", "pm"}

# MODEL 1: Raw Anglicisms - Dictionary-Based English Tokens (Excludes Acronyms)
def disambiguate_token(token, tokens, index, window=1):
    """
    Classify a token as 'english' or 'serbian'.
    If token appears only in one dictionary, return that classification.
    If ambiguous (appears in both), examine neighboring tokens to decide.
    """
    token_clean = token.lower().strip(string.punctuation)
    if token_clean in english_words and token_clean not in serbian_words:
        return 'english'
    elif token_clean in serbian_words and token_clean not in english_words:
        return 'serbian'
    elif token_clean in english_words and token_clean in serbian_words:
        start = max(0, index - window)
        end = min(len(tokens), index + window + 1)
        english_count = 0
        serbian_count = 0
        for i in range(start, end):
            if i == index:
                continue
            neighbor = tokens[i].lower().strip(string.punctuation)
            if neighbor in english_words and neighbor not in serbian_words:
                english_count += 1
            elif neighbor in serbian_words and neighbor not in english_words:
                serbian_count += 1
        return 'english' if english_count > serbian_count else 'serbian'
    else:
        return None

def extract_english_tokens(tweet, window=1):
    """
    Tokenize the tweet and return list of classified English tokens.
    Exclude acronyms.
    """
    tweet = str(tweet)
    tokens = nltk.word_tokenize(tweet)
    identified = []
    for i, token in enumerate(tokens):
        token_clean = token.strip(string.punctuation)
        # Exclude tokens that are acronyms so that "btw" isn't flagged here.
        if token_clean.lower() in acronyms:
            continue
        if token_clean.isalpha() and len(token_clean) > 1:
            classification = disambiguate_token(token, tokens, i, window=window)
            if classification == 'english':
                identified.append(token_clean.lower())
    return identified

# MODEL 2: Raw Acronym Extraction 
def extract_acronyms(tweet):
    tweet = str(tweet)
    tokens = nltk.word_tokenize(tweet)
    # Return acronyms as comma-separated string
    return ", ".join([token.lower() for token in tokens if token.lower() in acronyms])

def contains_acronym(tweet):
    tweet = str(tweet)
    tokens = nltk.word_tokenize(tweet)
    return any(token.lower() in acronyms for token in tokens)

# MODEL 3: Fuzzy Matching

# Transliteration Preparation - Defined mapping from Serbian letters to English phonetic equivalents
serbian_to_english = {
    'č': 'ch',
    'ć': 'ch',
    'c': 'ts',
    'đ': 'dj',
    'j': 'y',
    'š': 'sh',
    'ž': 'zh'
}

# Transliteration function
def transliterate(word):
    """
    Replace Serbian characters in 'word' with their English phonetic equivalents.
    """
    for serb, eng in serbian_to_english.items():
        word = word.replace(serb, eng).replace(serb.upper(), eng.upper())
    return word


def extract_fuzzy_english(tweet, threshold=92): 
    # threshold can be changed, I found n = 92 to give the best results on this dataset
    """
    Tokenize the tweet and return tokens (3+ characters with 1+ vowel) that,
    after transliteration, fuzzy-match an English word in our dictionary with a score >= threshold.
    Skips tokens with a match in the Serbian dictionary. 
    """
    tweet = str(tweet)
    tokens = nltk.word_tokenize(tweet)
    identified = []
    vowels = set("aeiou")
    for token in tokens:
        token_clean = token.strip(string.punctuation).lower()
        # Only consider tokens that are 3+ characters long, have at least one vowel, and are not present in the Serbian dictionary.
        if token_clean.isalpha() and len(token_clean) >= 3 and any(v in token_clean for v in vowels) and token_clean not in serbian_words:
            # Transliterate the token
            token_translit = transliterate(token_clean)
            match = process.extractOne(token_translit, list(english_words), scorer=fuzz.ratio)
            if match is not None:
                best_match, score, _ = match
                if score >= threshold:
                    identified.append(token_clean)
    return identified


# Read in dataset (Ljubešić et al, 2023)
df = pd.read_csv('output/tweets_all_raw.txt', header=None, names=['tweet'], encoding='utf-8')

# Create dataframe for Model 1 (raw anglicisms)
df['identified_english'] = df['tweet'].apply(lambda tweet: ", ".join(extract_english_tokens(tweet, window=1)))
df['has_english'] = df['tweet'].apply(lambda tweet: len(extract_english_tokens(tweet, window=1)) > 0)

# Create dataframe for Model 2 (raw acronyms)
df['identified_acronym'] = df['tweet'].apply(lambda tweet: extract_acronyms(tweet))
df['has_acronym'] = df['tweet'].apply(lambda tweet: contains_acronym(tweet))

# Create dataframe for Model 3 (fuzzy matching)
df['identified_fuzzy'] = df['tweet'].apply(lambda tweet: ", ".join(extract_fuzzy_english(tweet, threshold=90)))
df['has_fuzzy'] = df['tweet'].apply(lambda tweet: len(extract_fuzzy_english(tweet, threshold=90)) > 0)

# Output tweets into excel sheet, with non-overlapping data
# Priority order:
# Model 1 / Excel tab 1: Tweets with dictionary-based English tokens.
# Model 2 / Excel tab 2: Tweets with acronyms that are not already in Subset 1.
# Model 3 / Excel tab 3: Tweets with fuzzy matches that are not in Subset 1 or 2.
# Excel tab 4: Remaining tweets that were not flagged by any of the 3 possible models.
subset1 = df[df['has_english']].copy()
subset2 = df[(df['has_acronym']) & (~df['has_english'])].copy()
subset3 = df[(df['has_fuzzy']) & (~df['has_english']) & (~df['has_acronym'])].copy()
subset4 = df[~(df['has_english'] | df['has_acronym'] | df['has_fuzzy'])].copy()

# For Model 3, copy the fuzzy matching tokens into the "identified_english" column to compare with Model 1's identified english
subset3.loc[:, 'identified_english'] = subset3['identified_fuzzy']

# Write All Subsets to the Same Excel File with Separate Sheets for easy analysis
with pd.ExcelWriter("V2whole_analysis.xlsx", engine="openpyxl") as writer:
    subset1.to_excel(writer, sheet_name="English Tweets", index=False)
    subset2.to_excel(writer, sheet_name="Acronyms", index=False)
    subset3.to_excel(writer, sheet_name="Fuzzy English", index=False)
    subset4.to_excel(writer, sheet_name="Remaining Tweets", index=False)

# Print summary in terminal of stats overviews (eg: flagged 100 English tweets, 30 acronyms, etc)
print("Saved subsets:")
print("English Tweets: {} tweets".format(len(subset1)))
print("Acronyms: {} tweets".format(len(subset2)))
print("Fuzzy English: {} tweets".format(len(subset3)))
print("Remaining Tweets: {} tweets".format(len(subset4)))
