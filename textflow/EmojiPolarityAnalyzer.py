from typing import Optional
from textflow.Analyzer import Analyzer
import emoji

class EmojiPolarityAnalyzer(Analyzer):
    """
    A class that provides methods to analyze the polarity of the text of a sequence.

    Attributes:
        positiveEmoji: a list with the positive emojis 
        negativeEmoji: a list with the negative emojis
    """

    def __init__(self, positiveEmojiPath = "textflow\emojis\emojiPositive.txt", negativeEmojiPath = 'textflow\emojis\emojiNegative.txt'):
        """
        Create a text emoji polarity analyzer from an input object.
        Args:
            positiveEmojiPath = the path of the file that contains the positive text emojis.
            negativeEmojiPath = the path of the file that contains the negative text emojis.
        """
        positiveEmoji = []
        negativeEmoji = []
        with open(positiveEmojiPath) as archivo:
            positiveEmoji = archivo.readlines()
        with open(negativeEmojiPath) as archivo:
            negativeEmoji = archivo.readlines()

        self.positiveEmoji = []
        self.negativeEmoji = []
        for posEmoji in positiveEmoji:
            self.positiveEmoji.append(chr(int(posEmoji.replace("\n", "").replace(" ", "")[2:], 16)).encode('unicode-escape').decode('unicode-escape'))
        for negEmoji in negativeEmoji:
            self.negativeEmoji.append(chr(int(negEmoji.replace("\n", "").replace(" ", "")[2:], 16)).encode('unicode-escape').decode('unicode-escape'))
    

    def analyze(self, sequence, tag, levelOfAnalyzer, levelOfResult:Optional[str] = ""): 
        """
        Analyze a sequence with a text emoji polarity function.

        Args:
            sequence: the Sequence we want to analyze.
            tag: the label to store the analysis result.
            levelOfAnalyzer: the path of the sequence level to analyze inside of the result.
            levelOfResult: the path of the sequence level to store the result.
        """
        super().analyze(self.emojiPolarity,sequence, tag, levelOfAnalyzer, levelOfResult, True)

    def emojiPolarity(self, arrayText):
        """
        Function that analyzes the polarity of a list of texts.

        Args:
            arrayText: list that contains the texts that we want to analyze
        Returns:
            A list with the dictionaries. Each dictionary contains the result
            of the analysis of the corresponding text.
        """
        arrayResults =[]
        for text in arrayText:
            dicNumEmoji = {"positive": 0, "negative": 0}
            emojiFounded = {"positive": set(), "negative": set()}
            
            listEmojis=emoji.emoji_lis(text)
            for dictEmoji in listEmojis:
                if dictEmoji['emoji'] in self.positiveEmoji:
                    dicNumEmoji['positive'] += 1
                    emojiFounded['positive'].add(dictEmoji['emoji'])
                elif dictEmoji['emoji'] in self.negativeEmoji:
                    dicNumEmoji['negative'] += 1
                    emojiFounded['negative'].add(dictEmoji['emoji'])

            if(dicNumEmoji['positive'] > 0 or dicNumEmoji['negative'] >0):
                percentagePolarity = {"positive": dicNumEmoji['positive']/(dicNumEmoji['positive']+dicNumEmoji['negative']), "negative": dicNumEmoji['negative']/(dicNumEmoji['positive']+dicNumEmoji['negative'])}
            else:
                percentagePolarity = {"positive": 0, "negative": 0}
            
            tep = {
                "numEmojisPolarity": dicNumEmoji,
                "percentageEmojisPolarity": percentagePolarity,
                "distinctEmojiFounded": emojiFounded
            }
            arrayResults.append(tep)
        
        return arrayResults

