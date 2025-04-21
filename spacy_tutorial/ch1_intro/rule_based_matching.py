import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)


def print_matches(matches):
    for match_id, start, end in matches:
        print("Match found: ", doc[start:end].text)


# Example
pattern = [{"TEXT": "iPhone"}, {"TEXT": "X"}]
matcher.add("IPHONE_PATTERN", [pattern])

doc = nlp("Upcoming iPhone X release date leaked")

matches = matcher(doc)
print_matches(matches)

# Ex.1
matcher = Matcher(nlp.vocab)
doc = nlp(
    "After making the iOS update you won't notice a radical system-wide "
    "redesign: nothing like the aesthetic upheaval we got with iOS 7. Most of "
    "iOS 11's furniture remains the same as in iOS 10. But you will discover "
    "some tweaks once you delve a little deeper."
)

pattern = [{"TEXT": "iOS"}, {"IS_DIGIT": True}]
matcher.add("IOS_VERSION_PATTERN", [pattern])
matches = matcher(doc)
print_matches(matches)

# Ex.2
matcher = Matcher(nlp.vocab)
doc = nlp(
    "i downloaded Fortnite on my laptop and can't open the game at all. Help? "
    "so when I was downloading Minecraft, I got the Windows version where it "
    "is the '.zip' folder and I used the default program to unpack it... do "
    "I also need to download Winzip?"
)

pattern = [{"LEMMA": "download"}, {"POS": "PROPN"}]

matcher.add("DOWNLOAD_THING_PATTERN", [pattern])
matches = matcher(doc)
print_matches(matches)

# Ex.3
matcher = Matcher(nlp.vocab)
doc = nlp(
    "Features of the app include a beautiful design, smart search, automatic "
    "labels and optional voice responses."
)

pattern = [{"POS": "ADJ"}, {"POS": "NOUN"}, {"POS": "NOUN", "OP": "?"}]
matcher.add("ADJ_NOUN_PATTERN", [pattern])
matches = matcher(doc)
print_matches(matches)
