"""Dataset loader and data utilities.

Author:
    Shrey Desai and Yasumasa Onoe
"""

import collections
import itertools
import torch
import spacy
import json

from torch.utils.data import Dataset
from random import shuffle
from utils import cuda, load_dataset, load_tags, load_tag_file


PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'


class Vocabulary:
    """
    This class creates two dictionaries mapping:
        1) words --> indices,
        2) indices --> words.

    Args:
        samples: A list of training examples stored in `QADataset.samples`.
        vocab_size: Int. The number of top words to be used.

    Attributes:
        words: A list of top words (string) sorted by frequency. `PAD_TOKEN`
            (at position 0) and `UNK_TOKEN` (at position 1) are prepended.
            All words will be lowercased.
        encoding: A dictionary mapping words (string) to indices (int).
        decoding: A dictionary mapping indices (int) to words (string).
    """
    def __init__(self, samples, vocab_size, is_list):
        self.words = self._initialize_list(samples) if is_list else self._initialize(samples, vocab_size)
        self.encoding = {word: index for (index, word) in enumerate(self.words)}
        self.decoding = {index: word for (index, word) in enumerate(self.words)}

    def _initialize_list(self, samples):
        vocab = collections.defaultdict(int)
        for token in samples:
            vocab[token.lower()] += 1
        top_words = [word for word in vocab]
        words = [PAD_TOKEN, UNK_TOKEN] + top_words
        return words

    def _initialize(self, samples, vocab_size):
        """
        Counts and sorts all tokens in the data, then it returns a vocab
        list. `PAD_TOKEN and `UNK_TOKEN` are added at the beginning of the
        list. All words are lowercased.

        Args:
            samples: A list of training examples stored in `QADataset.samples`.
            vocab_size: Int. The number of top words to be used.

        Returns:
            A list of top words (string) sorted by frequency. `PAD_TOKEN`
            (at position 0) and `UNK_TOKEN` (at position 1) are prepended.
        """
        vocab = collections.defaultdict(int)
        for (_, passage, question, _, _, _, _, _, _) in samples:
            for token in itertools.chain(passage, question):
                vocab[token.lower()] += 1
        top_words = [
            word for (word, _) in
            sorted(vocab.items(), key=lambda x: x[1], reverse=True)
        ][:vocab_size]
        words = [PAD_TOKEN, UNK_TOKEN] + top_words
        return words
    
    def __len__(self):
        return len(self.words)


class Tokenizer:
    """
    This class provides two methods converting:
        1) List of words --> List of indices,
        2) List of indices --> List of words.

    Args:
        vocabulary: An instantiated `Vocabulary` object.

    Attributes:
        vocabulary: A list of top words (string) sorted by frequency.
            `PAD_TOKEN` (at position 0) and `UNK_TOKEN` (at position 1) are
            prepended.
        pad_token_id: Index of `PAD_TOKEN` (int).
        unk_token_id: Index of `UNK_TOKEN` (int).
    """
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.pad_token_id = self.vocabulary.encoding[PAD_TOKEN]
        self.unk_token_id = self.vocabulary.encoding[UNK_TOKEN]

    def convert_tokens_to_ids(self, tokens):
        """
        Converts words to corresponding indices.

        Args:
            tokens: A list of words (string).

        Returns:
            A list of indices (int).
        """
        return [
            self.vocabulary.encoding.get(token, self.unk_token_id)
            for token in tokens
        ]

    def convert_ids_to_tokens(self, token_ids):
        """
        Converts indices to corresponding words.

        Args:
            token_ids: A list of indices (int).

        Returns:
            A list of words (string).
        """
        return [
            self.vocabulary.decoding.get(token_id, UNK_TOKEN)
            for token_id in token_ids
        ]


class QADataset(Dataset):
    """
    This class creates a data generator.

    Args:
        args: `argparse` object.
        path: Path to a data file (.gz), e.g. "datasets/squad_dev.jsonl.gz".

    Attributes:
        args: `argparse` object.
        meta: Dataset metadata (e.g. dataset name, split).
        elems: A list of raw examples (jsonl).
        samples: A list of preprocessed examples (tuple). Passages and
            questions are shortened to max sequence length.
        tokenizer: `Tokenizer` object.
        batch_size: Int. The number of example in a mini batch.
    """
    def __init__(self, args, dataset_path, tags_path):
        self.args = args
        self.meta, self.elems = load_dataset(dataset_path)
        self.tags_path = tags_path
        self.pos, self.dep = load_tags(tags_path)
        self.samples = self._create_samples()
        self.tokenizers = None
        self.batch_size = args.batch_size if 'batch_size' in args else 1
        self.pad_token_id = self.tokenizers.pad_token_id \
            if self.tokenizers is not None else 0

    def _create_samples(self):
        """
        Formats raw examples to desired form. Any passages/questions longer
        than max sequence length will be truncated.

        Returns:
            A list of words (string).
        """
        # nlp = spacy.load("en")

        # # pos tags
        # pos_tags = load_tag_file(self.args.pos_tag_path)
        # pos_vocabulary = Vocabulary(pos_tags, len(pos_tags), True)
        # pos_tokenizer = Tokenizer(pos_vocabulary)
        # # dep tags
        # dep_tags = load_tag_file(self.args.dep_tag_path)
        # dep_vocabulary = Vocabulary(dep_tags, len(dep_tags), True)
        # dep_tokenizer = Tokenizer(dep_vocabulary)

        samples = []
        pos_tags =[]
        dep_tags = []
        index = 0
        for elem in self.elems:
            # Unpack the context paragraph. Shorten to max sequence length.
            passage = [
                token.lower() for (token, offset) in elem['context_tokens']
            ][:self.args.max_context_length]
            passage_pos = self.pos[index]
            passage_dep = self.dep[index]
            if (len(passage_pos) != len(passage)):
                print("UH OH IN THE PASSAGES")
            index += 1
            # write out the tags
            # Each passage has several questions associated with it.
            # Additionally, each question has multiple possible answer spans.
            
            # passage_doc = nlp(" ".join(passage))
            # if (len(passage_doc) > len(passage)):
            #     i = 0
            #     passage_doc = [token for token in passage_doc]
            #     while i < len(passage_doc) and i < len(passage):
            #         if (passage_doc[i].text[0] != passage[i][0]):
            #             del passage_doc[i]
            #             i -= 1
            #         i += 1
            #     passage_doc = passage_doc[:i]
            # pos_tags.append(pos_tokenizer.convert_tokens_to_ids([token.tag_.lower() for token in passage_doc]))
            # dep_tags.append(dep_tokenizer.convert_tokens_to_ids([token.dep_.lower() for token in passage_doc]))
            
            for qa in elem['qas']:
                qid = qa['qid']
                question = [
                    token.lower() for (token, offset) in qa['question_tokens']
                ][:self.args.max_question_length]

                question_pos = self.pos[index]
                question_dep = self.dep[index]
                if (len(question_pos) != len(question)):
                    print("UH OH IN THE PASSAGES")

                index += 1

                # question_doc = nlp(" ".join(question))
                # if (len(question_doc) > len(question)):
                #     i = 0 
                #     question_doc = [token for token in question_doc]
                #     while i < len(question_doc) and i < len(question):
                #         if (question_doc[i].text[0] != question[i][0]):
                #             del question_doc[i]
                #             i -= 1
                #         i += 1
                #     question_doc = question_doc[:i]
                # pos_tags.append(pos_tokenizer.convert_tokens_to_ids([token.tag_.lower() for token in question_doc]))
                # dep_tags.append(dep_tokenizer.convert_tokens_to_ids([token.dep_.lower() for token in question_doc]))

                # Select the first answer span, which is formatted as
                # (start_position, end_position), where the end_position
                # is inclusive.
                answers = qa['detected_answers']
                answer_start, answer_end = answers[0]['token_spans'][0]
                samples.append(
                    (qid, passage, question, answer_start, answer_end, passage_pos, passage_dep, question_pos, question_dep)
                )
                # index += 1
                # print("Index is now " + str(index))
        # with open(self.tags_path, "w") as file:
        #     final_dict = {'pos': pos_tags, 'dep': dep_tags}
        #     file.write(json.dumps(final_dict))
            
        return samples

    def _create_data_generator(self, shuffle_examples=False):
        """
        Converts preprocessed text data to Torch tensors and returns a
        generator.

        Args:
            shuffle_examples: If `True`, shuffle examples. Default: `False`

        Returns:
            A generator that iterates through all examples one by one.
            (Tuple of tensors)
        """
        if self.tokenizers is None:
            raise RuntimeError('error: no tokenizer registered')

        example_idxs = list(range(len(self.samples)))
        if shuffle_examples:
            shuffle(example_idxs)

        passages = []
        questions = []
        passages_pos = []
        passages_dep = []
        questions_pos = []
        questions_dep = []
        start_positions = []
        end_positions = []
        for idx in example_idxs:
            # Unpack QA sample and tokenize passage/question.
            qid, passage, question, answer_start, answer_end, passage_pos, passage_dep, question_pos, question_dep = self.samples[idx]
            # Convert words to tensor.
            passage_ids = torch.tensor(
                self.tokenizers.convert_tokens_to_ids(passage)
            )
            question_ids = torch.tensor(
                self.tokenizers.convert_tokens_to_ids(question)
            )

            passage_pos_tags = torch.tensor(passage_pos)
            passage_dep_tags = torch.tensor(passage_dep)
            question_pos_tags = torch.tensor(question_pos)
            question_dep_tags = torch.tensor(question_dep)

            if len(passage) != len(passage_pos):
                print(passage)
                print([token for token in passage_pos])

            if len(question) != len(question_pos):
                print(question)
                print([token for token in question_pos])

            answer_start_ids = torch.tensor(answer_start)
            answer_end_ids = torch.tensor(answer_end)

            # Store each part in an independent list.
            passages.append(passage_ids)
            questions.append(question_ids)
            passages_pos.append(passage_pos_tags)
            questions_pos.append(question_pos_tags)
            passages_dep.append(passage_dep_tags)
            questions_dep.append(question_dep_tags)
            start_positions.append(answer_start_ids)
            end_positions.append(answer_end_ids)

        return zip(passages, questions, passages_pos, questions_pos, 
                passages_dep, questions_dep, start_positions, end_positions)

    def _create_batches(self, generator, batch_size):
        """
        This is a generator that gives one batch at a time. Tensors are
        converted to "cuda" if necessary.

        Args:
            generator: A data generator created by `_create_data_generator`.
            batch_size: Int. The number of example in a mini batch.

        Yields:
            A dictionary of tensors containing a single batch.
        """
        current_batch = [None] * batch_size
        no_more_data = False
        # Loop through all examples.
        while True:
            bsz = batch_size
            # Get examples from generator
            for i in range(batch_size):
                try:
                    current_batch[i] = list(next(generator))
                except StopIteration:  # Run out examples
                    no_more_data = True
                    bsz = i  # The size of the last batch.
                    break
            # Stop if there's no leftover examples
            if no_more_data and bsz == 0:
                break

            passages_words = []
            questions_words = []
            passages_pos = []
            questions_pos = []
            passages_dep = []
            questions_dep = []
            start_positions = torch.zeros(bsz)
            end_positions = torch.zeros(bsz)
            max_passage_length = 0
            max_question_length = 0
            # Check max lengths for both passages and questions
            for ii in range(bsz):
                passages_words.append(current_batch[ii][0])
                questions_words.append(current_batch[ii][1])
                passages_pos.append(current_batch[ii][2])
                questions_pos.append(current_batch[ii][3])
                passages_dep.append(current_batch[ii][4])
                questions_dep.append(current_batch[ii][5])
                start_positions[ii] = current_batch[ii][6]
                end_positions[ii] = current_batch[ii][7]
                max_passage_length = max(
                    max_passage_length, len(current_batch[ii][0])
                )
                max_question_length = max(
                    max_question_length, len(current_batch[ii][1])
                )

            # Assume pad token index is 0. Need to change here if pad token
            # index is other than 0.
            padded_passages_words = torch.zeros(bsz, max_passage_length)
            padded_questions_words = torch.zeros(bsz, max_question_length)
            padded_passages_pos = torch.zeros(bsz, max_passage_length)
            padded_questions_pos = torch.zeros(bsz, max_question_length)
            padded_passages_dep = torch.zeros(bsz, max_passage_length)
            padded_questions_dep = torch.zeros(bsz, max_question_length)
            # Pad passages and questions
            for iii, passage_question in enumerate(zip(passages_words, questions_words,
                    passages_pos, questions_pos, passages_dep, questions_dep)):
                p_words, q_words, p_pos, q_pos, p_dep, q_dep = passage_question
                padded_passages_words[iii][:len(p_words)] = p_words
                padded_questions_words[iii][:len(q_words)] = q_words
                padded_passages_pos[iii][:len(p_pos)] = p_pos
                padded_questions_pos[iii][:len(q_pos)] = q_pos
                padded_passages_dep[iii][:len(p_dep)] = p_dep
                padded_questions_pos[iii][:len(q_dep)] = q_dep
                

            # Create an input dictionary
            batch_dict = {
                'passages_words': cuda(self.args, padded_passages_words).long(),
                'questions_words': cuda(self.args, padded_questions_words).long(),
                'passages_pos': cuda(self.args, padded_passages_pos).long(),
                'questions_pos': cuda(self.args, padded_questions_pos).long(),
                'passages_dep': cuda(self.args, padded_passages_dep).long(),
                'questions_dep': cuda(self.args, padded_questions_dep).long(),
                'start_positions': cuda(self.args, start_positions).long(),
                'end_positions': cuda(self.args, end_positions).long()
            }

            if no_more_data:
                if bsz > 0:
                    # This is the last batch (smaller than `batch_size`)
                    yield batch_dict
                break
            yield batch_dict

    def get_batch(self, shuffle_examples=False):
        """
        Returns a data generator that supports mini-batch.

        Args:
            shuffle_examples: If `True`, shuffle examples. Default: `False`

        Returns:
            A data generator that iterates though all batches.
        """
        return self._create_batches(
            self._create_data_generator(shuffle_examples=shuffle_examples),
            self.batch_size
        )

    def register_tokenizers(self, word_tokenizer):
        """
        Stores `Tokenizer` object as an instance variable.

        Args:
            tokenizer: If `True`, shuffle examples. Default: `False`
        """
        self.tokenizers = word_tokenizer
    
    def __len__(self):
        return len(self.samples)
