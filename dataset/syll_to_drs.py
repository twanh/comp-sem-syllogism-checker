import argparse
import json

import tqdm

from transformers import AutoTokenizer, T5ForConditionalGeneration

from loguru import logger


def main() -> int:

    parser = argparse.ArgumentParser(
        description="Syllable to DRS conversion using byT5-DRS model."
    )
    parser.add_argument(
        'in_file',
        type=str,
        help='Path to the input file containing syllogisms.',
    )

    parser.add_argument(
        'out_file',
        type=str,
        help='Path to the output file to save DRS representations.',
    )

    args = parser.parse_args()

    # Initialize the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        'XiaoZhang98/byT5-DRS', max_length=512
    )
    model = T5ForConditionalGeneration.from_pretrained("XiaoZhang98/byT5-DRS")

    # Load input file
    with open(args.inf_file, 'r') as f:
        data = json.load(f)

    output_data = []

    for row in tqdm.tqdm(data, desc="Processing syllogisms"):

        syllogism = row['syllogism']

        logger.info(f"Processing syllogism: {syllogism}")

        # Tokenize input syllogism
        x = tokenizer(
            syllogism,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )['input_ids']

        # Generate output
        output = model.generate(x)

        drs_representation = tokenizer.decode(
            output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        logger.info(f"Generated DRS representation: {drs_representation}")

        output_data.append({
            'syllogism': syllogism,
            'drs': drs_representation,
            'validity': row.get('validity', None),
       })

    # Save output to file
    with open(args.out_file, 'w') as f:
        json.dump(output_data, f, indent=4)

    logger.success(f"DRS representations saved to {args.out_file}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
