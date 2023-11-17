from typing import Optional
from textflow.Analyzer import Analyzer
import re
import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

class PerplexityAnalyzer(Analyzer):
    """
    A class that provides methods to analyze the perplexity of the text of a sequence.

    Attributes:
        language: the languague of text to analyze the emojis.
        textEmojis: the text with the words of the emojis instead of emojis.
        freqEmoji: a dictionary with the different emojis and their frequence.
        nEmoji: the number of emojis that appear in the text to analyze.
    """

    def __init__(self, model = GPT2LMHeadModel.from_pretrained("PlanTL-GOB-ES/gpt2-large-bne"),tokenizer = GPT2TokenizerFast.from_pretrained("PlanTL-GOB-ES/gpt2-large-bne"), device="cuda"):
        """
        Create an emoji analyzer.

        Args:
            language: the languague of text to analyze the emojis.
        """
        self.device=device
        self.model = model.to(device)
        self.tokenizer = tokenizer
        
    
    def analyze(self, sequence, tag, levelOfAnalyzer, levelOfResult:Optional[str] = ""): 
        """
        Analyze a sequence with a emoji function.

        Args:
            sequence: the Sequence we want to analyze.
            tag: the label to store the analysis result.
            levelOfAnalyzer: the path of the sequence level to analyze inside of the result.
            levelOfResult: the path of the sequence level to store the result.
        """
        super().analyze(self.perplexityResult,sequence, tag, levelOfAnalyzer, levelOfResult, True)

    def perplexityResult(self, arrayText):
        """
        Function that analyzes the emojis of a list of texts.

        Args:
            arrayText: list that contains the texts that we want to analyze
        Returns:
            A float list. Each float value of the list represent the result
            of the perplexity of the corresponding text.
        """
        resultsList = []
        for t in arrayText:
            resultsList.append(self.perplexity(t))
        return resultsList

    def perplexity(self, text):
        """
        Function that calculate the perplexity of a text

        Args:
            text: the text that we want to analyze
        """
        encodings = self.tokenizer(text, return_tensors="pt")
        max_length = self.model.config.n_positions
        stride = 512
        seq_len = encodings.input_ids.size(1)
        nlls = []
        prev_end_loc = 0
        for begin_loc in tqdm(range(0, seq_len, stride)):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)

                # loss is calculated using CrossEntropyLoss which averages over valid labels
                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                # to the left by 1.
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        ppl = torch.exp(torch.stack(nlls).mean())

        return ppl.item()
        


