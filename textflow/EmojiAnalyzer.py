from typing import Optional
import emoji
from textflow.Analyzer import Analyzer
import re

class EmojiAnalyzer(Analyzer):
    """
    A class that provides methods to analyze the different emojis of the text of a sequence.

    Attributes:
        language: the languague of text to analyze the emojis.
        textEmojis: the text with the words of the emojis instead of emojis.
        freqEmoji: a dictionary with the different emojis and their frequence.
        nEmoji: the number of emojis that appear in the text to analyze.
    """

    def __init__(self, language='es'):
        """
        Create an emoji analyzer.

        Args:
            language: the languague of text to analyze the emojis.
        """
        self.language=language
        
    
    def analyze(self, sequence, tag, levelOfAnalyzer, levelOfResult:Optional[str] = ""): 
        """
        Analyze a sequence with a emoji function.

        Args:
            sequence: the Sequence we want to analyze.
            tag: the label to store the analysis result.
            levelOfAnalyzer: the path of the sequence level to analyze inside of the result.
            levelOfResult: the path of the sequence level to store the result.
        """
        super().analyze(self.emojiResult,sequence, tag, levelOfAnalyzer, levelOfResult, True)

    def emojiResult(self, arrayText):
        """
        Function that analyzes the emojis of a list of texts.

        Args:
            arrayText: list that contains the texts that we want to analyze
        Returns:
            A list with the dictionaries. Each dictionary contains the result
            of the analysis of the corresponding text.
        """
        resultsList = []
        for t in arrayText:
            t.lower()
            self.countEmoji(t)
            result={
                "TextWithoutEmoji": self.textEmojis,
                "FreqEmoji": self.freqEmoji,
                "NumEmojis": self.nEmoji
            }
            resultsList.append(result)
        return resultsList

    def countEmoji(self, text):
        """
        Function that counts the number of emojis that appear in the text, their frequency and
        changes the corresponding emoji for its meaning in words

        Args:
            text: the text that we want to analyze
        """
        self.freqEmoji={}
        textNoEmoji = emoji.demojize(text, language=self.language)
        emojis = re.findall(r':[\w][_\w.]*:',textNoEmoji)
        self.nEmoji = len(emojis)
        for emo in emojis:
            if emo in self.freqEmoji:
                self.freqEmoji[emo] += 1
            else:
                self.freqEmoji[emo] = 1
        self.textEmojis=re.sub(r':[\w][_\w.]*:', self.repl_func, textNoEmoji)

    def repl_func(self, match):
        """
        Function that replace the match of a string.

        Args:
            match: the match object with the pattern

        Returns:
            The match with ' ' instead of ':' and with ' ' instead of '_'
        """
        return match[0].replace(":"," ").replace("_"," ")
        


