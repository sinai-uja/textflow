# TextFlow class

This class provides methods to create a sequence from directories, documents, strings… To calculate different metrics from these sequences. It consists of a Python library for calculating different metrics from plain texts.

# Introduction:

In this library, we have sequences and analyzers.

+ **Sequences:** are the main element of the library. A sequence has three main attributes:

    + **Format:** is a string with the origin format of a sequence. This format can be a string, a file(.txt), a directory, a token, etc.
    + **Metadata:** This is a dictionary where we store the metadata of a sequence, like the source text of a sequence (if the origin of the sequence is a file of text), or the path of the directory (if the origin of the sequence is a directory). Different analyzers store the result of the analysis inside this dictionary if the result of the analysis is metadata (the number of words, the label with the emotion of a text, the text source replacing words...).
    + **Children:** This is a dictionary where we store a list of sequences that came from the actual sequence. For example, if we have a text, we can split this text in phrases. "Phrases" will be the key in the dictionary and each phrase of the text will be a sequence inside the list of sequences of the key of the children's dictionary. Each phrase can be split into words too, that we will store in the children's dictionary of the phrase sequences. So, inside the original sequence(text) we have a sequence of phrases and inside of them sequences of words. This forms the different levels of a sequence.

        <ul> 
            <li> The level in a sequence is used like a path in a directory, to access the different subsequences in analyzers or filter functions. In our example, we have:
            <ul>
                <li> Text
                <ul> 
                    <li> Phrases
                        <ul> 
                            <li> Words 
                        </ul>
                </ul>
            </ul>
        </ul>

        So, to access children of level Words we can use "Phrases/Words" in filter or analyze. In the same mode, we can use "Phrases/Words/text" to access a text(value of metadata dictionary) at the Words level in functions like filterMetadata or analyze.    

+ **Analyzers:** The analyzers provide methods to analyze sequences and store the result in a sequence. These analyzers can analyze the metadata of a sequence or the children of a sequence. And can store the result in any of these dictionaries (metadata or children).


# Files

 - **[INSTALL.md](https://github.com/sinai-uja/textflow/blob/master/INSTALL.md):** A guide to make this project work in your local environment.

### ./textFlow

- **[Analyzer.py](https://github.com/sinai-uja/textflow/blob/master/textflow/Analyzer.py):** This module provides an abstract class with methods for the calculation of different metrics on a sequence.

- **[ComplexityAnalyzer.py](https://github.com/sinai-uja/textflow/blob/master/textflow/ComplexityAnalyzer.py):** This module provides class methods for the calculation of different complexity metrics on a sequence. This class inherits from Analyzer.py

- **[CREA_5000.txt](https://github.com/sinai-uja/textflow/blob/master/textflow/Crea-5000.txt):** A dataset of 5000 Spanish words ordered by its absolute frequency.

- **[EmotionAnalyzer.py](https://github.com/sinai-uja/textflow/blob/master/textflow/EmotionAnalyzer.py):** This module provides class methods for the calculation of the emotions on a sequence. This class inherits from Analyzer.py

- **[LemmaAnalyzer.py](https://github.com/sinai-uja/textflow/blob/master/textflow/LemmaAnalyzer.py):** This module provides class methods for the calculation of different lemma metrics on a sequence. This class inherits from Analyzer.py

- **[LexicalDiversityAnalyzer.py](https://github.com/sinai-uja/textflow/blob/master/textflow/LexicalDiversityAnalyzer.py):** This module provides a class methods for the calculation of different lexical diversity measures. This class inherits from Analyzer.py

- **[POSAnalyzer.py](https://github.com/sinai-uja/textflow/blob/master/textflow/POSAnalyzer.py):** This module provides a class methods for the calculation of different Part-of-speech metrics on a sequence. This class inherits from Analyzer.py

- **[PolarityAnalyzer.py](https://github.com/sinai-uja/textflow/blob/master/textflow/PolarityAnalyzer.py):** This module provides class methods for the calculation of the polarity on a sequence. This class inherits from Analyzer.py

- **[Sequence.py](https://github.com/sinai-uja/textflow/blob/master/textflow/Sequence.py):** This module provides an abstract class with methods for creating sequences from different sources. A sequence contains 2 dictionaries, one for metadata and the other for the subsequence of this sequence.

- **[SequenceDirectory.py](https://github.com/sinai-uja/textflow/blob/master/textflow/SequenceDirectory.py):** This module provides methods for creating a sequence from a directory.

- **[SequenceFile.py](https://github.com/sinai-uja/textflow/blob/master/textflow/SequenceFile.py):** This module provides methods for creating a sequence from a file or a document.

- **[SequenceString.py](https://github.com/sinai-uja/textflow/blob/master/textflow/SequenceString.py):** This module provides methods for creating a sequence from a string.

- **[SequenceToken.py](https://github.com/sinai-uja/textflow/blob/master/textflow/SequenceToken.py):** This module provides methods for creating a sequence from a token.

- **[StylometryAnalyzer.py](https://github.com/sinai-uja/textflow/blob/master/textflow/StylometryAnalyzer.py):** This module provides class methods for the calculation of different stylometry metrics on a sequence. This class inherits from Analyzer.py

- **[VolumetryAnalyzer.py](https://github.com/sinai-uja/textflow/blob/master/textflow/VolumetryAnalyzer.py):** This module provides class methods for the calculation of different volumetry metrics on a sequence. This class inherits from Analyzer.py

- **[IronityAnalyzer.py](https://github.com/sinai-uja/textflow/blob/master/textflow/IronityAnalyzer.py):** This module provides class methods for the calculation of the ironity on a sequence. This class inherits from Analyzer.py

- **[NERAnalyzer.py](https://github.com/sinai-uja/textflow/blob/master/textflow/NERAnalyzer.py):** This module provides class methods for the search of different NER on a sequence. This class inherits from Analyzer.py

- **[NGramsAnalyzer.py](https://github.com/sinai-uja/textflow/blob/master/textflow/NGramsAnalyzer.py):** This module provides class methods for the calculation of n-grams and their frequence on a sequence. This class inherits from Analyzer.py

- **[EmojiAnalyzer.py](https://github.com/sinai-uja/textflow/blob/master/textflow/EmojiAnalyzer.py):** This module provides class methods for the calculation of different emojis metrics on a sequence. This class inherits from Analyzer.py

- **[TextEmojiPolarityAnalyzer.py](https://github.com/sinai-uja/textflow/blob/master/textflow/TextEmojiPolarityAnalyzer.py):** This module provides class methods to calculate the polarity of a text related to text emojis on a sequence. This class inherits from Analyzer.py

- **[EmojiPolarityAnalyzer.py](https://github.com/sinai-uja/textflow/blob/master/textflow/EmojiPolarityAnalyzer.py):** This module provides class methods to calculate the polarity of a text related to emojis on a sequence. This class inherits from Analyzer.py

- **[PerplexityAnalyzer.py](https://github.com/sinai-uja/textflow/blob/master/textflow/PerplexityAnalyzer.py):** This module provides class methods to calculate the perplexity of a text. This class inherits from Analyzer.py

**Note:** All of the analyzers implemented by default are applied to plain text.

### ./examples

- **[example_text.txt](https://github.com/sinai-uja/textflow/blob/master/Examples/ExampleDirectory/Documento%20sin%20t%C3%ADtulo.txt):** Simple .txt file to test the library.

- **[example.ipynb](https://github.com/sinai-uja/textflow/blob/master/Examples/Example.ipynb):** Colab notebook that shows how to use the different methods of textFlow.

- **[AnalyzeADataframe.ipynb](https://github.com/sinai-uja/textflow/blob/master/Examples/AnalyzeADataframe.ipynb.ipynb):** Colab notebook that shows how to use textFlow with a Dataframe of pandas.

# Metrics

In this section, we introduce the different metrics offered in this Python library. These metrics are returned by the corresponding analyzer and stored in the corresponding dictionary (metadata or children) of a sequence.


- **Volumetry:** Here it calculates different metrics that are stored in a metadata dictionary:

    + **words:** The number of words in the text.
    + **uniqueWords:** The number of unique words in the text.
    + **chars:** The number of characters of the text.
    + **avgWordsLen:** The average word length for text

- **Lemmas:** It calculates different metrics that are stored in a metadata dictionary:
    
    + **srclemmas:** A list with the words of the text lemmatized.
    + **uniqueLemmas:** The number of unique lemmas of the text.
    + **avgLemmas:** The average lemma length for text.

- **Part-of-speech (POS)**: It calculates different metrics that are stored in a metadata dictionary: 
    
    + **srcPOS:** A list with the POS of the words of the text
    + **FreqPOS:** The frequency of the different POS labels.

- **Complexity:**  It calculates different metrics that are stored in a metadata dictionary: 

    + **nSentences:** The number of sentences.
    + **nComplexSentence:** The number of complex sentences.
    + **avglenSentence:**  The average sentence length.
    + **nPuntuationMarks:** The number of punctuation marks.
    + **nWords:** The number of words.
    + **nRareWords:** The number of rare words.
    + **nSyllabes:** The number of syllabes.
    + **nChar:** The number of characters.
    + **ILFW:** The index of low-frequency words.                   
    + **LDI:** The index of lexical distribution.                     
    + **LC:** The lexical complexity.
    + **SSR:** The spaulding score.
    + **SCI:** The index of sentence complexity
    + **ARI:** The auto readability index.                    
    + **huerta:** The readability of Fernandez-Huerta.   
    + **IFSZ:** The perspicuity of Flesh-Szigriszt.     
    + **polini:** The polaini compressibility. 
    + **mu:** The mu legibility.       
    + **minage:** An indicator of minimum age.          
    + **SOL:** The SOL readability         
    + **crawford:** An indicator of Crawford's age 
    + **min_depth:** minimum of maximum tree depths
    + **max_depth:** maximum of maximum tree depths
    + **mean_depth:** mean of maximum tree depths 

- **Stylometry:** It calculates different metrics that are stored in a metadata dictionary:
    + **uniqueWords:** The number of different words. 
    + **TTR:** The lexical index TTR
    + **RTTR:** The lexical index RTTR
    + **Herdan:** The lexical index Herdan
    + **Mass:** The lexical index Mass
    + **Somers:** The lexical index Somers
    + **Dugast:** The lexical index Dugast
    + **Honore:** The lexical index Honore
    + **FreqStopWords:** The frequency of stopwords
    + **FreqPuntuationMarks:** The frequency of punctuation marks
    + **FreqWords:** The frequency of words

- **LexicalDiversity** It calculates different metrics on lexical diversity:
    + **SimpleTTR:** Simple Token-Type Ratio.
    + **RootTTR:** Root Token-Type Ratio.
    + **LogTTR:** Log Token-Type Ratio.
    + **MaasTTR:** Maas Token-Type Ratio.
    + **MSTTR:** Mean segmental Token-Type Ratio. The segment size is 50 words. 
    + **MATTR:** Moving average Token-Type Ratio. The window size is 50 words.
    + **HDD:** Hypergeometric distribution D. A more straightforward and reliable implementation of vocD (Malvern, Richards, Chipere, & Duran, 2004) as per McCarthy and Jarvis (2007, 2010).
    + **MTLD:** Measure of lexical textual diversity. Calculates MTLD based on McCarthy and Jarvis (2010).
    + **MTLDMAWrap:** Measure of lexical textual diversity (moving average, wrap). Calculates MTLD using a moving window approach. Instead of calculating partial factors, it wraps to the beginning of the text to complete the last factors.
    + **MTLDMABi:** Measure of lexical textual diversity (moving average, bi-directional). Calculates the average MTLD score by calculating MTLD in each direction using a moving window approach.

- **Polarity:** Polarity score of a text.

    + **label:** the label that predicts the polarity model.
    + **score:** the score to assign the label to the text.

- **Emotions:** Emotions score of a text.

    + **label:** the label that predicts the polarity model.
    + **score:** the score to assign the label to the text.

- **Emojis:** It calculates different metrics that are stored in a metadata dictionary:

    + **TextWithoutEmoji:** A string with the words of emojis instead of the emoji.
    + **FreqEmoji:** The frequence of emojis
    + **NumEmojis:** The number of emojis.

- **NER:** It calculates different metrics that are stored in a metadata dictionary:

    + **srcNER:** The text with entities instead of the words
    + **entidades:** The entities grouped by each category
    + **freqEntidades:** The frequency of different entities. 

- **N-Grams:** It calculates different metrics that are stored in a metadata dictionary:

    + **n-grams:** The different n-grams of the text
    + **freqN-Grams:** The frequency of different n-grams

- **Ironity:** Ironity score of a text.
    
    + **label:** the label that predicts the polarity model.
    + **score:** the score to assign the label to the text.

- **TextEmojiPolarity:** It calculates a text's polarity based on text emojis. (Martínez-Cámara, E., Jiménez-Zafra, S. M., Martin-Valdivia, M. T., & Lopez, L. A. U. (2014, August). SINAI: voting system for twitter sentiment analysis. *In Proceedings of the 8th International Workshop on Semantic Evaluation (SemEval 2014)* (pp. 572-577))
    
    + **numEmojisPolarity:** the number of positive and negative text emojis.
    + **percentageEmojisPolarity:** the score to assign the label positive or negative to the text related to the text emojis that appear in the text.
    + **distinctEmojiFounded:** the different positive and negative text emojis.

- **EmojiPolarity:** It calculates a text's polarity based on emojis.
    
    + **numEmojisPolarity:** the number of positive and negative emojis.
    + **percentageEmojisPolarity:** the score to assign the label positive or negative to the text related to the emojis that appear in the text.
    + **distinctEmojiFounded:** the different positive and negative emojis.

- **Perplexity:** It calculates the perplexity of a text.
    
    + **perplexity:** the perplexity of a text. Perplexity is defined as the exponentiated average negative log-likelihood of a sequence.

# Dependencies

- **ComplexityAnalyzer.py, POSAnalyzer, LemmaAnalyzer, and NERAnalyzer:** In these classes, spacy is used to calculate the different metrics of the analyzers. If you want to use other packages, you should implement the methods nlp, sents, pos_, lemma_, and text.

- **IronityAnalizer.py, EmotionAnalyzer.py, PolarityAnalyzer, and PerplexityAnalyzer.py:** These classes use models and pipelines of transformers, you can use different models to infer the emotion, the polarity, or the perplexity of a text.

- **EmojiAnalizer.py, EmojiPolarityAnalyzer.py:** This class use emoji library.

# How to create a Sequence?

If you want to create a class to initialize a sequence without a class for it, you can create your own. 

- **We must create a class that inherits from Sequence.py**

- **Inside of this class, we have to create the following functions, because these functions implement the basic functionality of a Sequence**. The basic functionality is like iterating in the sequence, filtering in subsequences(children), printing the sequence...:


                def initializeSequence(self, format):
                    '''
                    Initializes the attributes of a sequence.

                    Args:
                        format: a string with the origin format of the sequence.
                    '''
                    super().initializeSequence(format)

                def __str__(self):
                    '''
                    Convert a Sequence to a string
                    
                    Returns:
                    A string that contains the text of a Sequence  
                    '''
                    return super().__str__()

                def __repr__(self):
                    '''
                    Convert a Sequence to a string
                    
                    Returns:
                    A string with the formal representation of a Sequence  
                    '''
                    return super().__repr__()

                def __len__(self):
                    '''
                    Calculate the length of a Sequence.
                    The length of a Sequence is the length of the children.

                    Returns:
                        A number with the length of the Sequence
                    '''
                    return super().__len__()

                def __iter__(self):
                    '''
                    Iterate in a Sequence
                    To do this, we iterates througth the children dictionary 
                    Returns:
                        A Sequence Iterator  
                    '''
                    return super().__iter__()

                def __getitem__(self, idx):
                    '''
                    Get the value of a key from the dictionary of children 

                    Args:
                        idx: a string that represent the key of the children dictionary
                            or an integer that represent the position of the key in children dictionary keys

                    Returns:
                        A List of Sequences 
                    '''
                    return super().__getitem__(idx)

                def __eq__(self, other):
                    '''
                    Check if a sequence it is the same that the current one.

                    Args:
                        other: a sequence to check if it is the same that the current one.
                    Returns:
                        True if the sequences are equals.
                        False in others cases.
                    '''
                    return super().__eq__(other)

                def depth(self,dictionaryList: Optional[list] = None):
                    '''
                    Calculate the maximum depth of a Sequence

                    Args:
                        diccionaryList: the inicial list to calculate the depth.

                    Returns:
                        A tuple that contains a number (the depth of a Sequence) and a list (the route of the max depth) 
                    '''
                    return super().depth(dictionaryList)

                def filter(self, level, criteria):
                    '''
                    Filter the children of a Sequence according to a criteria

                    Args:
                        level: the route of the level as string, separating each level with "/" 
                        criteria: the filter function

                    Returns:
                        A generator with the result of the filter
                    '''
                    return super().filter(level,criteria)

                def filterMetadata(self, level, criteria):
                    '''
                    Filter the children of a Sequence according to a criteria

                    Args:
                        level: the route of the level as string, separating each level with "/" 
                        criteria: the filter function

                    Returns:
                        A generator with the result of the filter
                    '''
                    return super().filterMetadata(level,criteria)

- **Now, we have to create the __init__ function in the class:**

        def __init__(self):
            pass

    - **First of all, we have to call self.initializeSequence("format"), this function initialize the metadata, and children dictionary of a Sequence and put the format of a sequence.**

        We are going to see, how to initialize a sequence from a directory.

            def __init__(self):
                    
                    # Initializes the attributes of a sequence.
                    self.initializeSequence("directory") 

    - **Then, we have to think about the metadata that we will have and how the children sequences are going to be built in that initializer.** 

        In the example we have like metadata the name of the files of the directory and the path of the diferent subdirectories.
        
            def __init__(self,src):
                '''
                Initialize a Sequence from a directory path
    
                By default, create subsequences for any directories and files in the source directory 
                and for each file, create subsequence, splitting the text of the file into words.
    
                Args:
                    src: the path of the directory
                '''
                # Initializes the attributes of a sequence.
                self.initializeSequence("directory") 
    
                # Create the metadata and children of a Sequence
                self.metadata["nameFiles"] = []
                self.metadata["directoriesPath"] = []
                contenido = os.listdir(src)
                for file in contenido:
                    if os.path.isfile(src+"/"+file):
                        self.metadata["nameFiles"].append(file)
                    
                    else:
                        self.metadata["directoriesPath"].append(src+"/"+file)

    - **Finally,we can create the sequences down to the lowest level by calling other sequence initializers and what labels they will have in the children dictionary.** 

        Here, we can see how we add new parameters to create more sublevels in the original sequence.

            def __init__(self,src,listLabel = ["directories","files","tokens"],listClasses=[SequenceFile,SequenceString],listTokenizer=[WhitespaceTokenizer()]):
                '''
                Initialize a Sequence from a directory path
    
                By default, create subsequences for any directories and files in the source directory 
                and for each file, create subsequence, splitting the text of the file into words.
    
                Args:
                    src: the path of the directory
                    listLabel: a list with different labels to create new levels in the children dictionary
                    listClasses: a list with different classes that inicialize a sequence with sublevels
                    listTokenizer: a list with the tokenizer to inicialize the different subsequences
    
                '''
                # Initializes the attributes of a sequence.
                self.initializeSequence("directory") 
    
                # Create the metadata and children of a Sequence
                self.metadata["nameFiles"] = []
                self.metadata["directoriesPath"] = []
                if not listTokenizer or listTokenizer == None:
                        listTokenizer = [WhitespaceTokenizer()]
                contenido = os.listdir(src)
                for file in contenido:
                    if os.path.isfile(src+"/"+file):
                        self.metadata["nameFiles"].append(file)
                        if listLabel and listClasses:
                            if listLabel[1] in self.children:
                            #Create a sublevel of sequence
                                self.children[listLabel[1]].append(listClasses[0](src+"/"+file,listLabel[1:],listClasses[1:],listTokenizer[1:])) 
                            else:
                            #Create a sublevel of sequence
                                self.children[listLabel[1]] = [listClasses[0](src+"/"+file,listLabel[1:],listClasses[1:],listTokenizer[1:])]
                    
                    else:
                        self.metadata["directoriesPath"].append(src+"/"+file)
                        if listLabel[0] in self.children:
                            self.children[listLabel[0]].append(SequenceDirectory(src+"/"+file,listLabel,listClasses,listTokenizer ))
                        else:
                            self.children[listLabel[0]] = [SequenceDirectory(src+"/"+file,listLabel,listClasses,listTokenizer)]

- **The result of the new Initializer of a Sequence from directory is look like:**

        class SequenceDirectory(Sequence):
            """
            A class that provides methods to create a sequence from a directory

            Attributes:
                format: a string with the origin format of a sequence.
                metadata: a dictionary with the metadata of a sequence.
                children: a dictionary with the subsequence of a sequence.
            """


            def __init__(self,src,listLabel = ["directories","files","tokens"],listClasses=[SequenceFile,SequenceString],listTokenizer=[WhitespaceTokenizer()]):
                '''
                Initialize a Sequence from a directory path

                By default, create subsequences for any directories and files in the source directory 
                and for each file, create subsequence, splitting the text of the file into words.

                Args:
                    src: the path of the directory
                    listLabel: a list with different labels to create new levels in the children dictionary
                    listClasses: a list with different classes that inicialize a sequence with sublevels
                    listTokenizer: a list with the tokenizer to inicialize the different subsequences

                '''
                self.initializeSequence("directory")
                self.metadata["nameFiles"] = []
                self.metadata["directoriesPath"] = []
                if not listTokenizer or listTokenizer == None:
                        listTokenizer = [WhitespaceTokenizer()]
                contenido = os.listdir(src)
                for file in contenido:
                    if os.path.isfile(src+"/"+file):
                        self.metadata["nameFiles"].append(file)
                        if listLabel and listClasses:
                            if listLabel[1] in self.children:
                                self.children[listLabel[1]].append(listClasses[0](src+"/"+file,listLabel[1:],listClasses[1:],listTokenizer[1:]))
                            else:
                                self.children[listLabel[1]] = [listClasses[0](src+"/"+file,listLabel[1:],listClasses[1:],listTokenizer[1:])]
                    
                    else:
                        self.metadata["directoriesPath"].append(src+"/"+file)
                        if listLabel[0] in self.children:
                            self.children[listLabel[0]].append(SequenceDirectory(src+"/"+file,listLabel,listClasses,listTokenizer ))
                        else:
                            self.children[listLabel[0]] = [SequenceDirectory(src+"/"+file,listLabel,listClasses,listTokenizer)]

                

            def initializeSequence(self, format):
                '''
                Initializes the attributes of a sequence.

                Args:
                    format: a string with the origin format of the sequence.
                '''
                super().initializeSequence(format)

            def __str__(self):
                '''
                Convert a Sequence to a string
                
                Returns:
                A string that contains the text of a Sequence  
                '''
                return super().__str__()
            

            def __repr__(self):
                '''
                Convert a Sequence to a string
                
                Returns:
                A string with the formal representation of a Sequence  
                '''
                return super().__repr__()

            def __len__(self):
                '''
                Calculate the length of a Sequence.
                The length of a Sequence is the length of the children.

                Returns:
                    A number with the length of the Sequence
                '''
                return super().__len__()

            def __iter__(self):
                '''
                Iterate in a Sequence
                To do this, we iterates througth the children dictionary 
                Returns:
                    A Sequence Iterator  
                '''
                return super().__iter__()
            
            def __getitem__(self, idx):
                '''
                Get the value of a key from the dictionary of children 

                Args:
                    idx: a string that represent the key of the children dictionary
                        or an integer that represent the position of the key in children dictionary keys

                Returns:
                    A List of Sequences 
                '''
                return super().__getitem__(idx)

            def __eq__(self, other):
                '''
                Check if a sequence it is the same that the current one.

                Args:
                    other: a sequence to check if it is the same that the current one.
                Returns:
                    True if the sequences are equals.
                    False in others cases.
                '''
                return super().__eq__(other)

            def depth(self,dictionaryList: Optional[list] = None):
                '''
                Calculate the maximum depth of a Sequence

                Args:
                    diccionaryList: the inicial list to calculate the depth.

                Returns:
                    A tuple that contains a number (the depth of a Sequence) and a list (the route of the max depth) 
                '''
                return super().depth(dictionaryList)
            
            def filter(self, level, criteria):
                '''
                Filter the children of a Sequence according to a criteria

                Args:
                    level: the route of the level as string, separating each level with "/" 
                    criteria: the filter function

                Returns:
                    A generator with the result of the filter
                '''
                return super().filter(level,criteria)
            
            def filterMetadata(self, level, criteria):
                '''
                Filter the children of a Sequence according to a criteria

                Args:
                    level: the route of the level as string, separating each level with "/" 
                    criteria: the filter function

                Returns:
                    A generator with the result of the filter
                '''
                return super().filterMetadata(level,criteria)

# How to create an Analyzer?

The steps to create an analyzer are:

- **Create a class that inherits of Analyzer.py**

- **Create the __init__ function of this class with the params to configurate this class**

    + For example, if we want to do a ironity analyzer, we use a pipeline that need a task, a model, a maximum of Embedding and if return all the scores of only the label with the maximum score of the text. The analyzer must be as flexible as possible, so all of the parameters that need the pipeline are passed like params at the init function.

        + Look that inside the init function we modified the labels of the model, this is because the model by defect dont have the labels clearly defined. (NI = Non-Ironic and I = Ironic)   



                def __init__(self, task = "text-classification",modelIronity = 'dtomas/roberta-base-bne-irony', allScores = True, maxEmbedding = 514):
                
                """
                Create an ironic analyzer.
    
                Args:
                    task: the task defining which pipeline will be returned.
                    model: the model that will be used by the pipeline to make predictions.
                    allScores: True, if we want that the classifier returns all scores. False, in other case.
                    maxEmbedding: The number of max_position_embeddings in the config.json of the model selected.
                """
                
                if modelIronity == 'dtomas/roberta-base-bne-irony':
                    model = AutoModelForSequenceClassification.from_pretrained(modelIronity)
                    model.config.id2label = {0: 'NI', 1: 'I'}
                    model.config.label2id = {'NI': 0, 'I': 1}
                    tokenizer = AutoTokenizer.from_pretrained(modelIronity)
                    self.ironityClassifier = pipeline(task,model= model, tokenizer=tokenizer,return_all_scores=allScores, truncation=True)
                else:
                    self.ironityClassifier = pipeline(task,model= modelIronity, return_all_scores=allScores)
                self.maxEmbedding = maxEmbedding

- **Create a function that analyze a list of things that we want to analyze and return a list with the result of each text.**


    + For example, we want to create an analyzer of text, so we need a function that receives a list of texts and applies what it needs to each text. This function have to return a list with the result of the analisys.

    + We can see an example of a ironity function, that receives a list of text, for each text apply the ironity classifier and put the result in a list. This list is returned by the function:


        
            def ironity(self, arrayText):
                """
                Function that analyzes the ironity of a list of texts.
    
                Args:
                    arrayText: list that contains the texts that we want to analyze
                Returns:
                    A list with the dictionaries. Each dictionary contains the result
                    of the analysis of the corresponding text.
                """
                arrayResults =[]
                for text in arrayText:
                    prediction = self.ironityClassifier(text[:self.maxEmbedding])
                    arrayResults.append(prediction)
                return arrayResults


- **Create the analyze function (inherit function of Analyzer.py):**

    + Replace self.analyzeFunction with the analyzer function that we have implemented in the last point.
    + The parameter "True" of the super().analyze is because this analyzer is a metadata analyzer. If do you want to create an analyzer of sequence, this parameter must be "False".



            def analyze(self, sequence, tag, levelOfAnalyzer, levelOfResult:Optional[str] = ""): 
                    """
                    Analyze a sequence with a ironic function.
    
                    Args:
                        sequence: the Sequence we want to analyze.
                        tag: the label to store the analysis result.
                        levelOfAnalyzer: the path of the sequence level to analyze inside of the result.
                        levelOfResult: the path of the sequence level to store the result.
                    """
                    super().analyze(self.analyzeFunction,sequence, tag, levelOfAnalyzer, levelOfResult, True)

        In our example this function looks like:


            def analyze(self, sequence, tag, levelOfAnalyzer, levelOfResult:Optional[str] = ""): 
                    """
                    Analyze a sequence with a ironic function.
    
                    Args:
                        sequence: the Sequence we want to analyze.
                        tag: the label to store the analysis result.
                        levelOfAnalyzer: the path of the sequence level to analyze inside of the result.
                        levelOfResult: the path of the sequence level to store the result.
                    """
                    super().analyze(self.ironity,sequence, tag, levelOfAnalyzer, levelOfResult, True)

- **The result of the new analyzer is:**


    class IronityAnalyzer(Analyzer):
    """
    A class that provides methods to analyze the ironity of the text of a sequence.

    Attributes:
       ironityClassifier: a pipeline that uses a model for inference the ironity of the text of a sequence. 
                          By default, the label 'NI' is non-ironic and 'I' ironic.
        maxEmbedding: The number of max_position_embeddings in the config.json of the model selected.
    """

    def __init__(self, task = "text-classification",modelIronity = 'dtomas/roberta-base-bne-irony', allScores = True, maxEmbedding = 514):
        """
        Create an ironic analyzer.

        Args:
            task: the task defining which pipeline will be returned.
            model: the model that will be used by the pipeline to make predictions.
            allScores: True, if we want that the classifier returns all scores. False, in other case.
            maxEmbedding: The number of max_position_embeddings in the config.json of the model selected.
        """
        if modelIronity == 'dtomas/roberta-base-bne-irony':
            model = AutoModelForSequenceClassification.from_pretrained(modelIronity)
            model.config.id2label = {0: 'NI', 1: 'I'}
            model.config.label2id = {'NI': 0, 'I': 1}
            tokenizer = AutoTokenizer.from_pretrained(modelIronity)
            self.ironityClassifier = pipeline(task,model= model, tokenizer=tokenizer,return_all_scores=allScores, truncation=True)
        else:
            self.ironityClassifier = pipeline(task,model= modelIronity, return_all_scores=allScores)
        self.maxEmbedding = maxEmbedding
        

    
    def analyze(self, sequence, tag, levelOfAnalyzer, levelOfResult:Optional[str] = ""): 
        """
        Analyze a sequence with a ironic function.

        Args:
            sequence: the Sequence we want to analyze.
            tag: the label to store the analysis result.
            levelOfAnalyzer: the path of the sequence level to analyze inside of the result.
            levelOfResult: the path of the sequence level to store the result.
        """
        super().analyze(self.ironity,sequence, tag, levelOfAnalyzer, levelOfResult, True)

    def ironity(self, arrayText):
        """
        Function that analyzes the ironity of a list of texts.

        Args:
            arrayText: list that contains the texts that we want to analyze
        Returns:
            A list with the dictionaries. Each dictionary contains the result
            of the analysis of the corresponding text.
        """
        arrayResults =[]
        for text in arrayText:
            prediction = self.ironityClassifier(text[:self.maxEmbedding])
            arrayResults.append(prediction)
        return arrayResults

# Contact:
If you have any questions about the use of the library, you can contact with Estrella Vallecillo (mevallec@ujaen.es).

# Acknowledgements:

This work has been partially supported by the CONSENSO (PID2021-122263OB-C21), MODERATES (TED2021-130145B-I00), SocialTOX (PDC2022-133146-C21) projects financed by the Plan Nacional I+D+i of the Government of Spain, and the PRECOM (SUBV-00016) project financed by the Ministry of Consumer Affairs of the Government of Spain.
