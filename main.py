import argparse
import re
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


def separators(token):
  match token:
    case "NEGATION":
      return "NOT"
    case "CONJUNCTION":
      return "CON"
    case "EXPLANATION":
      return "" #?
    case "CONTINUATION":
      return "" #?  

def roles(token, var, var_step):
  match token:
    case "Agent":
      return "Agent(x" + str(var_step) + ", x" + str(var) + ")"
    case "Theme":
      return "Theme(x" + str(var_step) + ", x" + str(var) + ")"
    case "Patient":
      return "Patient(x" + str(var_step) + ", x" + str(var) + ")"
    case "Co-Theme":
      return "Co-Theme(x" + str(var_step) + ", x" + str(var) + ")"

def drs2fol(drs):
  drs_tokens = drs.split()
  fol_tokens = []
  var_count = len([x for x in drs_tokens if x[0].islower() and x[-1].isnumeric()])
  print(var_count)
  var_step = 0
  box_counter = 0
  for i, token in enumerate(drs_tokens):
    if token[0].islower() and token[-1].isnumeric():
      var_step += 1
      fol_tokens.append("exists")
      prev_token = drs_tokens[i-1]
      if prev_token == "<1":
        fol_tokens.append("[")
        fol_tokens.append("<1")
        fol_tokens.append("]")
      else:
        fol_tokens.append("x"+ str(var_step))
      fol_tokens.append("[")
      box_counter += 1
      fol_tokens.append(token + "(x"+ str(var_step) + ")")
    if token[0] in ["=", "≠", "≈", "≤", "≥"]:
      fol_tokens.append(token)
    if token[0] in ["-", "+"]:
      continue
    if token in ["NEGATION", "CONJUNCTION", "EXPLANATION", "CONTINUATION"]:
      fol_tokens.append("&")
      fol_tokens.append(separators(token))
    if token[0] in ["<", ">"] and token [1:].isnumeric():
      continue
    if token in ["Theme", "Agent", "Patient", "Co-Theme"]:
      drs_var = drs_tokens[i+1]
      var = var_step + int(drs_var[-1])
      fol_tokens.append(roles(token, var, var_step))
  fol_tokens.append("]"*box_counter)
  print(fol_tokens)

  find_var_re = r"\b(x\d+)\b"
  detect_var = []
  for i, token in enumerate(fol_tokens[::-1]):
    if token == "<1":
      reverse_index = len(fol_tokens)-i
      index = reverse_index-1
      for j in range(-5, 6):
        if fol_tokens[index+j] == "<1":
          index = index+j
      fol_tokens.pop(index)
      detect_var = sorted(set(detect_var), key=lambda v: int(v[1:]))
      for var in detect_var[::-1]:
          fol_tokens.insert(index, var)
      detect_var = []
    found_vars = re.findall(find_var_re, token)
    if found_vars:
      for var in found_vars:
        detect_var.append(var)

  print(fol_tokens)
  return " ".join(fol_tokens)



if __name__ == "__main__":
    raise SystemExit(main())
