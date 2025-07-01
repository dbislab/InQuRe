"""
Test the embedding via transformers (these tests need an internet connection for nltk wordnet)
"""
import time
import unittest

import nltk
import numpy as np
from nltk.corpus import wordnet
from torch import Tensor

from query_rewriting.distance_measures.vector_embedding import model_embedding, model_similarity, set_up_model, \
    model_similarity_for_lists, model_embedding_list, embedding_gpt, calculate_cosine_sim, \
    similarity_spacy_en_core_web_lg, load_en_core_web_lg


# Sources for wordnet: https://www.nltk.org/howto/wordnet.html

class TestVectorEmbedding(unittest.TestCase):

    def setUp(self):
        """
        Download the NLTK Wordnet library (if needed), needs an internet connection
        """
        # quiet parameter: suppress the message if nltk wordnet is downloaded in the console
        nltk.download('wordnet', quiet=True)

    def test_model_similarity(self):
        """
        Test if the all-mpnet-base-v2 embedder works for two very similar sentences and time the execution.
        """
        start_time_model_setup: float = time.time()
        set_up_model('sentence-transformers/all-mpnet-base-v2')
        end_time_model_setup: float = time.time()
        start_time1: float = time.time()
        tensor1: Tensor = model_embedding("This is a sentence.")
        end_time1: float = time.time()
        start_time2: float = time.time()
        tensor2: Tensor = model_embedding("This is also a sentence.")
        end_time2: float = time.time()
        start_time3: float = time.time()
        similarity: float = model_similarity(tensor1, tensor2)
        end_time3: float = time.time()
        print(
            f"\nFirst embedding time: {end_time1 - start_time1}, second embedding time: {end_time2 - start_time2}, "
            f"similarity calculation time: {end_time3 - start_time3}")
        print(f"Model setup time for all-mpnet-base-v2: {end_time_model_setup - start_time_model_setup}\n")
        # print(f"Similarity with mpnet: {similarity}")
        self.assertGreater(similarity, 0.89637)  # Similarity with mpnet: 0.8963762521743774


    def test_play_with_different_inputs(self):
        """
        Test to play around with examples.
        """
        set_up_model('sentence-transformers/all-mpnet-base-v2')
        sentence1: str = "rent"
        sentence2: str = "income"
        tensor1: Tensor = model_embedding(sentence1)
        tensor2: Tensor = model_embedding(sentence2)
        similarity: float = model_similarity(tensor1, tensor2)
        print(f"\nSimilarity between:  \n'{sentence1}'  \nand  \n'{sentence2}'  \nis {similarity}.")

    @unittest.skip
    def test_play_with_different_inputs_GPT(self):
        """
        Test to play around with examples with the GPT embedders.
        """
        word1: str = "rent"
        word2: str = "income"
        vec1: list[float] = embedding_gpt(word1, "text-embedding-3-small")
        vec2: list[float] = embedding_gpt(word2, "text-embedding-3-small")
        sim: float = calculate_cosine_sim(vec1, vec2)
        print(f"Similarity of  \n'{word1}'  \nand  \n'{word2}'  \nis: {sim}")

    def test_accuracy_on_synonyms(self):
        """
        Test if all-mpnet-base-v2 is good for synonyms (i.e. they have a high similarity).
        """
        set_up_model('sentence-transformers/all-mpnet-base-v2')
        test_words: list = ['car', 'house', 'dog']
        for word in test_words:
            syns: list[list] = wordnet.synonyms(word)
            syn_list: list = [word for sublist in syns for word in sublist]
            print(f"\nSynonyms of {word}: {syn_list}")
            tensor1: Tensor = model_embedding(word)
            tensor2: np.ndarray = model_embedding_list(syn_list)
            similarity: np.ndarray = model_similarity_for_lists(tensor1, tensor2).numpy()
            # print(similarity.shape)
            # print(similarity)
            list_sim: list = similarity.flatten().tolist()
            print(f"Similarity of synonyms: {list_sim}\n")
            for i, synonym in enumerate(syn_list):
                print(f"Word: {word}, Synonym: {synonym}, Similarity: {list_sim[i]}")

    def test_accuracy_on_antonyms(self):
        """
        Test if all-mpnet-base-v2 is good for antonyms (i.e. they have a low similarity).
        """
        set_up_model('sentence-transformers/all-mpnet-base-v2')
        test_words: list = ['good']
        for word in test_words:
            antonyms = []
            for syn in wordnet.synsets(word):
                for lemmas in syn.lemmas():
                    if lemmas.antonyms():
                        antonyms.append(lemmas.antonyms()[0].name())
            print(f"\nAntonyms of {word}: {antonyms}")
            tensor1: Tensor = model_embedding(word)
            tensor2: np.ndarray = model_embedding_list(antonyms)
            similarity: np.ndarray = model_similarity_for_lists(tensor1, tensor2).numpy()
            # print(similarity.shape)
            # print(similarity)
            list_sim: list = similarity.flatten().tolist()
            print(f"Similarity of antonyms: {list_sim}\n")
            for i, antonym in enumerate(antonyms):
                print(f"Word: {word}, Antonym: {antonym}, Similarity: {list_sim[i]}")

    def test_spacy_similarity(self):
        string_in1: str = "rent"
        string_in2: str = "apartment"
        start_time: float = time.time()
        sim: float = similarity_spacy_en_core_web_lg(string_in1, string_in2, load_en_core_web_lg())
        end_time: float = time.time()
        print(f"\nTime needed in total for spacy: {end_time - start_time}")
        print(f"Similarity of  \n'{string_in1}'  \nand  \n'{string_in2}'  \nis: {sim}")

    def test_cosine_sim(self):
        vec1: list[float] = [1, 1, 1, 1]
        vec2: list[float] = [1, 1, 1, 1]
        self.assertEqual(1, calculate_cosine_sim(vec1, vec2))
        vec1: list[float] = [-1, -1, -1, -1]
        vec2: list[float] = [1, 1, 1, 1]
        self.assertEqual(-1, calculate_cosine_sim(vec1, vec2))
        vec1: list[float] = [3, 5, 1, 1]
        vec2: list[float] = [1, 1, 1, 1]
        self.assertEqual(10 / 12, calculate_cosine_sim(vec1, vec2))

