# TextFlow class

This class provides methods to create a sequence from directories, documents, strings… To calculate different metrics from these sequences. It consists in a Python library for calculating different metrics from plain texts.

# Files

 - **[INSTALL.md](https://gitlab.ujaen.es/jcollado/textflow/blob/master/INSTALL.md):** A guide to make this project work on your local environment.

### ./textFlow

- **[Analyzer.py](https://gitlab.ujaen.es/jcollado/textflow/blob/master/textflow/Analyzer.py):** This module provides an abstract class with methods for the calculation of different metrics on a sequence.

- **[ComplexityAnalyzer.py](https://gitlab.ujaen.es/jcollado/textflow/blob/master/textflow/ComplexityAnalyzer.py):** This module provides a class methods for the calculation of different complexity metrics on a sequence. This class inherits from Analyzer.py

- **[CREA_5000.txt](https://gitlab.ujaen.es/jcollado/textflow/blob/master/textflow/Crea-5000.txt):** A dataset of 5000 spanish words ordered by its absolute frequency.

- **[EmotionAnalyzer.py](https://gitlab.ujaen.es/jcollado/textflow/blob/master/textflow/EmotionAnalyzer.py):** This module provides a class methods for the calculation of the emotions on a sequence. This class inherits from Analyzer.py

- **[LemmaAnalyzer.py](https://gitlab.ujaen.es/jcollado/textflow/blob/master/textflow/LemmaAnalyzer.py):** This module provides a class methods for the calculation of different lemma metrics on a sequence. This class inherits from Analyzer.py

- **[POSAnalyzer.py](https://gitlab.ujaen.es/jcollado/textflow/blob/master/textflow/POSAnalyzer.py):** This module provides a class methods for the calculation of different Part-of-speech metrics on a sequence. This class inherits from Analyzer.py

- **[PolarityAnalyzer.py](https://gitlab.ujaen.es/jcollado/textflow/blob/master/textflow/PolarityAnalyzer.py):** This module provides a class methods for the calculation of the polarity on a sequence. This class inherits from Analyzer.py

- **[Sequence.py](https://gitlab.ujaen.es/jcollado/textflow/blob/master/textflow/Sequence.py):** This module provides an abstract class with methods for creating sequences from dfferent sources. A sequence contains 2 dictionaries, one for metadata and other for the subsequence of this sequence.

- **[SequenceDirectory.py](https://gitlab.ujaen.es/jcollado/textflow/blob/master/textflow/SequenceDirectory.py):** This module provides methods for creating a sequence from a directory.

- **[SequenceFile.py](https://gitlab.ujaen.es/jcollado/textflow/blob/master/textflow/SequenceFile.py):** This module provides methods for creating a sequence from a file or a document.

- **[SequenceString.py](https://gitlab.ujaen.es/jcollado/textflow/blob/master/textflow/SequenceString.py):** This module provides methods for creating a sequence from a string.

- **[SequenceToken.py](https://gitlab.ujaen.es/jcollado/textflow/blob/master/textflow/SequenceToken.py):** This module provides methods for creating a sequence from a token.

- **[StylometryAnalyzer.py](https://gitlab.ujaen.es/jcollado/textflow/blob/master/textflow/StylometryAnalyzer.py):** This module provides a class methods for the calculation of different stylometry metrics on a sequence. This class inherits from Analyzer.py

- **[VolumetryAnalyzer.py](https://gitlab.ujaen.es/jcollado/textflow/blob/master/textflow/VolumetryAnalyzer.py):** This module provides a class methods for the calculation of different volumetry metrics on a sequence. This class inherits from Analyzer.py

- **[IronityAnalyzer.py](https://gitlab.ujaen.es/jcollado/textflow/blob/master/textflow/IronityAnalyzer.py):** This module provides a class methods for the calculation of the ironity on a sequence. This class inherits from Analyzer.py

- **[NERAnalyzer.py](https://gitlab.ujaen.es/jcollado/textflow/blob/master/textflow/NERAnalyzer.py):** This module provides a class methods for the search of different NER on a sequence. This class inherits from Analyzer.py

- **[NGramsAnalyzer.py](https://gitlab.ujaen.es/jcollado/textflow/blob/master/textflow/NGramsAnalyzer.py):** This module provides a class methods for the calculation of n-grams and their frequence on a sequence. This class inherits from Analyzer.py

- **[EmojiAnalyzer.py](https://gitlab.ujaen.es/jcollado/textflow/blob/master/textflow/EmojiAnalyzer.py.py):** This module provides a class methods for the calculation of different emojis metrics on a sequence. This class inherits from Analyzer.py

**Note:** All of the analyzers implemented by default are applied to plain text.

### ./examples

- **[example_text.txt](https://gitlab.ujaen.es/jcollado/textflow/blob/master/Examples/ExampleDirectory/Documento%20sin%20t%C3%ADtulo.txt):** Simple .txt file to test the library.

- **[example.ipynb](https://gitlab.ujaen.es/jcollado/textflow/blob/master/Examples/Example.ipynb):** Colab notebook that shows how to use the different methods of textFlow.

# Metrics

In this section, we introduce the different metrics offered in this Python library. These metrics are returned by the corresponding analyzer and store in the corresponding dictionary (metadata or children) of a sequence.


- **Volumetry:** Here it calculates the number of words, number of unique words, number of characters and average word length for text.

- **Lemmas:** Number of the lemmas that are uniques in the text, a list of lemma words and the average lemma length for text.

- **Part-of-speech (POS)**: a list with the POS of the text and the frequency of the different POS labels.

- **Complexity:** Number of sentences, complex sentences, punctuation marks, words, rare words,chars, syllabes, average sentence length, index of low frequency words, index of lexical distribution, lexical complexity, spaulding score, index of sentence complexity, auto readability, readability of Fernandez-Huerta, perspicuity of Flesh-Szigriszt , polaini compressibility, mu legibility, SOL readability,some indicators of age (min age, crawford) and different indicators of the embeddings depth.

- **Stylometry:** Number of different words, different lexical index (TTR,RTT, Herdan, Mass, Somers, Dugast, Honore), frequency of stopwords, frequency of punctuation marks, frequency of words.

- **Polarity:** Polarity score of a text.

- **Emotions:** Emotions score of a text.

- **Emojis:** Number of emojis of the text, their frequence and the text with the words of the emojis instead of the emoji.

- **NER:** the frequence of different entities, the entities grouped by each category and the text with entities instead of the words.

- **N-Grams:** the different n-grams of the text and their frequence.

- **Ironity:** Ironity score of a text.

# Dependencies

- **ComplexityAnalyzer.py, POSAnalyzer, LemmaAnalyzer and NERAnalyzer:** In these classes, spacy is used to calculate the different metrics of the analyzers. If do you want to use other package, you should implements the methods nlp, sents, pos_, lemma_ and text.

- **IronityAnalizer.py, EmotionAnalyzer.py and PolarityAnalyzer.py:** These classes use models and pipelines of transformers, you can use different models to inference the emotion or the polarity of a text.

- **EmojiAnalizer.py:** This class use emoji library.
