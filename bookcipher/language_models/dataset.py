# Taken from our spell corrector

import argparse
import codecs
import os
# from sentence_splitter import sentence_splitter

def train_dataset_source_target_lines(train_file):
    with codecs.open(train_file, "r", "utf-8") as train:
        for line in train:
            source, target = line.rstrip().split('\t')
            yield (source, target)

def get_source_target_lines(input_files):
    for input_file in input_files:
        with codecs.open(input_file.format("sources"), "r", "utf-8") as source, codecs.open(input_file.format("targets"), "r", "utf-8") as target:
            for (source_sentence, source_target) in zip(source, target):
                yield (source_sentence.rstrip(), source_target.rstrip())

def get_source_target_prediction_lines(input_files, prediction_file):
    for input_file in input_files:
        with codecs.open(input_file.format("sources"), "r", "utf-8") as source, codecs.open(input_file.format("targets"), "r", "utf-8") as target, codecs.open(prediction_file, "r", "utf-8") as prediction:
            for (source_sentence, target_sentence, prediction_sentence) in zip(source, target, prediction):
                yield (source_sentence.rstrip(), target_sentence.rstrip(), prediction_sentence.rstrip())

# def make_dataset(source_target_lines, output_dir, dataset_type, max_lines, max_length, include_negatives):
#     i = 0
#     output_sentences = 0
#     output_source_file = os.path.join(output_dir, "{}_sources.txt".format(dataset_type))
#     output_target_file = os.path.join(output_dir, "{}_targets.txt".format(dataset_type))
#     output_target_file_detection = os.path.join(output_dir, "{}_targets_detection.txt".format(dataset_type))
#     with codecs.open(output_source_file, "w", "utf-8") as output_source, codecs.open(output_target_file, "w", "utf-8") as output_target, codecs.open(output_target_file_detection, "w", "utf-8") as output_target_detection:
#         for (source_line, target_line) in source_target_lines:
#             i += 1
#             if not i % 10000: print(i)
#             if max_lines is not None and i > max_lines: break
#             source_split = sentence_splitter(source_line)
#             target_split = sentence_splitter(target_line)
#             for (source_sentence, target_sentence) in zip(source_split, target_split):
#                 if len(source_sentence) > max_length or len(source_sentence) <= 1:
#                     continue
#                 if not include_negatives and source_sentence == target_sentence:
#                     # Skip over unchanged sentences
#                     continue
#                 output_sentences += 1
#                 for (source_char, target_char) in zip(source_sentence, target_sentence):
#                     output_target_detection.write('0' if source_char == target_char else '1')
#                 output_source.write(source_sentence + '\n')
#                 output_target.write(target_sentence + '\n')
#                 output_target_detection.write('\n')
#     print("Wrote {} output sentences".format(output_sentences))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate datasets for spell check.")
    parser.add_argument("--train_file", type=str)
    parser.add_argument("--source_file", type=str)
    parser.add_argument("--target_file", type=str)
    parser.add_argument("--output_dir", type=str, default="datasets")
    parser.add_argument("--dataset_type", type=str, default="train")
    parser.add_argument("--max_lines", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--include_negatives", default=False, action='store_true')
    args = parser.parse_args()

    if args.train_file and (args.source_file or args.target_file):
        print("Must specify either a training file path or a source and a target path")
        exit()
    elif args.train_file:
        source_target_lines = train_dataset_source_target_lines(args.train_file)
    elif args.source_file and args.target_file:
        source_target_lines = get_source_target_lines(args.source_file, args.target_file)
    else:
        print("Must specify either a training file path or a source and a target path")
        exit()

    make_dataset(source_target_lines, args.output_dir, args.dataset_type, args.max_lines, args.max_length, args.include_negatives)
