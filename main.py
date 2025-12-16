import argparse
from transformers import AutoTokenizer, T5ForConditionalGeneration

def main() -> int:

    # Initialize the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('XiaoZhang98/byT5-DRS', max_length=512)
    model = T5ForConditionalGeneration.from_pretrained("XiaoZhang98/byT5-DRS")

    parser = argparse.ArgumentParser()
    parser.add_argument('sentence')
    args = parser.parse_args()

    premises = args.sentence.split('.')
    print(f"Found the following premises: {premises}")

    drs_premises = []
    for p in premises:
        x = tokenizer(p, return_tensors='pt', padding=True, truncation=True, max_length=512)['input_ids']
        output = model.generate(x)
        pred_text = tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        drs_premises.append(pred_text)

    print(f"Sentence: {args.sentence}")
    for p, d in zip(premises, drs_premises):
        print(f"Premise: {p}")
        print(f"DRS: {d}")

    # TODO: DRS to FOL
    # plant.n.02 NEGATION <1 be.v.01 Theme -1 Time +1 Co-Theme +2 time.n.08 EQU now daisy.n.01
    # exists x1 [ plant.n.02(x1) & NOT  exists [x2 x3 x4][ be.v.01(x2) & Theme(x2, x1) & Time(x2, x3) & Co-Theme(x2, x4) & time.n.08(x3) &  x3 = now & daisy.n.01(x4)]]


    return 0


def drs2fol(drs):
  drs_tokens = drs.split()
  fol_tokens = []
  var_exist = 1
  box_counter = 0
  for token in drs_tokens:
    if token[0].islower():
      fol_tokens.append(token + "(x"+ str(var_exist) +")")
      var_exist += 1
    if token == "NEGATION":
      fol_tokens.append("NOT")
    if token[0].isupper() and token[1].islower():
      fol_tokens.append(token)
    if token[-1].isnumeric() and not token[0] == "<" and (token[0] == "-" or token[0] == "+"):
      if token[0] == "-":
        fol_tokens[-1] = fol_tokens[-1] + "(x"+ str(var_exist-int(token[-1])) +")"
      elif token[0] == "+":
        fol_tokens[-1] = fol_tokens[-1] + "(x"+ str(var_exist+int(token[-1])) +")"
        var_exist += 1
    if token[0] == "<":
      fol_tokens.append("[")
      box_counter += 1

  fol_tokens.append("]"*box_counter)


  print(fol_tokens)



if __name__ == "__main__":
    raise SystemExit(main())
