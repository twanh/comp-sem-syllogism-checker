import argparse
import json
import random
from dataclasses import asdict
from dataclasses import dataclass


@dataclass
class SyllogismDatasetItem:
    syllogism: str
    validity: bool
    plausibility: bool


def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description='Create combined syllogism dataset.',
    )


    parser.add_argument(
        'bertolazzi_file_believable',
        type=str,
        help='Path to the Bertolazzi et al. (2024) believable dataset file (JSON format).',  # noqa: E501
    )

    parser.add_argument(
        'bertolazzi_file_unbelievable',
        type=str,
        help='Path to the Bertolazzi et al. (2024) unbelievable dataset file (JSON format).',  # noqa: E501
    )

    parser.add_argument(
        'output_file',
        type=str,
        help='Path to the output combined dataset file (JSON format).',
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=32,
        help='Random seed for shuffling the dataset and choosing answer options.',  # noqa: E501
    )

    return parser.parse_args()


def load_jsonl(file_path: str) -> list[dict]:
    """Helper function to load jsonl files"""

    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    return data


def parse_bertolazzi_item(
    item: dict,
    plausibility: bool,
    validity: bool,
) -> SyllogismDatasetItem:

    # Extract premises
    text = item['text']
    text_lines = text.split('\n')
    premise_1 = None
    premise_2 = None

    # Parse Correct Answers
    # Make the first letter of each answer option uppercase and add
    # a period at the end (if not already present).
    # We do this to match the way the options are formatted in the text.
    answer_options = [
        answer_option.lower().replace('.', '')
        for answer_option in item['answer'].split(' or ')
    ]

    options = []

    for i, line in enumerate(text_lines):
        line = line.strip()  # Good practice to strip whitespace

        if line.startswith('Premise 1:'):
            premise_1 = line.replace('Premise 1:', '').strip()

        elif line.startswith('Premise 2:'):
            premise_2 = line.replace('Premise 2:', '').strip()

        elif line.startswith('Options:'):
            # Iterate through the remaining lines
            for sub_line in text_lines[i + 1:]:
                clean_sub_line = sub_line.strip()
                # Stop if we hit the "Answer:" block or an empty line
                if clean_sub_line.startswith('Answer:') or not clean_sub_line:
                    continue
                # Add to options list
                options.append(clean_sub_line.lower().replace('.', ''))

    # Sanity check to ensure we don't pick empty strings
    options = [o for o in options if o and 'answer:' not in o]

    if validity:
        conclusion = answer_options[random.randint(0, len(answer_options) - 1)]
    else:
        conclusion = options[random.randint(0, len(options) - 1)]

    syllogism_text = f'{premise_1} {premise_2} Therefore, {conclusion}.'
    return SyllogismDatasetItem(
        syllogism=syllogism_text,
        validity=validity,
        plausibility=plausibility,
    )


def main() -> int:

    args = parse_arguments()
    random.seed(args.seed)

    # Load datasets
    bertolazzi_believable = load_jsonl(args.bertolazzi_file_believable)
    bertolazzi_unbelievable = load_jsonl(args.bertolazzi_file_unbelievable)

    true_plausible = []
    true_implausible = []

    false_plausible = []
    false_implausible = []

    # Parse Bertolazzi believable data and convert to expected format
    for i, item in enumerate(bertolazzi_believable):
        if i % 2 == 0:
            parsed_item = parse_bertolazzi_item(
                item,
                plausibility=True,
                validity=True,
            )
            true_plausible.append(parsed_item)
        else:
            parsed_item = parse_bertolazzi_item(
                item,
                plausibility=True,
                validity=False,
            )
            false_plausible.append(parsed_item)

    # Parse Bertolazzi unbelievable data and convert to expected format
    for i, item in enumerate(bertolazzi_unbelievable):
        if i % 2 == 0:
            parsed_item = parse_bertolazzi_item(
                item,
                plausibility=False,
                validity=True,
            )
            true_implausible.append(parsed_item)
        else:
            parsed_item = parse_bertolazzi_item(
                item,
                plausibility=False,
                validity=False,
            )
            false_implausible.append(parsed_item)


    print('Parsed Bertolazzi et al. (2024) data:')
    print(f'\tTrue plausible: {len(true_plausible)}')
    print(f'\tFalse plausible: {len(false_plausible)}')
    print(f'\tTrue implausible: {len(true_implausible)}')
    print(f'\tFalse implausible: {len(false_implausible)}')

    combined_dataset = []
    combined_dataset.extend([asdict(item)for item in true_plausible])
    combined_dataset.extend([asdict(item) for item in false_plausible])
    combined_dataset.extend([asdict(item) for item in true_implausible])
    combined_dataset.extend([asdict(item) for item in false_implausible])

    random.shuffle(combined_dataset)

    # Print dataset statistics
    num_valid = sum(1 for item in combined_dataset if item['validity'])
    num_invalid = len(combined_dataset) - num_valid
    print('Combined dataset statistics:')
    print(f'\tTotal items: {len(combined_dataset)}')
    print(f'\tValid items: {num_valid}')
    print(f'\tInvalid items: {num_invalid}')
    print(
        f'\tValid percentage: {num_valid / len(combined_dataset) * 100:.2f}%',
    )
    # plausible
    num_plausible = sum(1 for item in combined_dataset if item['plausibility'])
    num_implausible = len(combined_dataset) - num_plausible
    print(f'\tPlausible items: {num_plausible}')
    print(f'\tImplausible items: {num_implausible}')
    print(
        f'\tPlausible percentage: {num_plausible / len(combined_dataset) * 100:.2f}%',  # noqa: E501
    )

    # TOtal items
    print(f'Total items: {len(combined_dataset)}')

    # Save combined datasets
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_dataset, f, indent=4)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
