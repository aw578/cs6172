# Autocomplete Next Bash Command

## Summary

This project implements a symbolic, program‑synthesis‑based autocomplete for bash sessions. Given a shell command and its resulting output, the system predicts and suggests the next command without executing anything. It learns from a user’s history by treating each `(cmdₖ, outputₖ) → cmdₖ₊₁` triple as a training example, and then builds a small decision tree or enumerative program (in a DSL of command templates and predicates) that maps prior context to the appropriate next command template.

Key elements:

- **Input/Output Examples**: Raw shell history is parsed into triples of command, output, next command.
- **DSL Design**: Define a limited language of predicates (e.g. `ContainsFile`, `IsEmpty`) and templates.
- **Synthesis Engine**: Use a decision‑tree learner (cf. `learn_decision_tree` in `part3.py`) or an enumerative bottom‑up search (cf. `bottom_up` in `part2.py`) to produce a program that fits all examples.
- **Instantiation**: At runtime, apply the synthesized program to the latest `(cmd, output)` to generate a concrete next‑command suggestion.

## Roadmap

(input1, output1, input2), (input2, output2, input3), ...
generate terms for each triple

- start with space-separated words in second input
- for each word, take the union w/ alternate string expressions using first input and output
- alternate string expressions are substrings that are either:

  - a whole space-separated word in the first input / output and part of a word in the second input
  - a whole word in the second input
  - for example, if the first input is "hello world a b" and the second input is "world ab", then the base expressions are "world" and "ab", and the alternate string expressions are "world", "a", and "b"

- alternate string expression types:
  - substring (positive indices)
  - reversed substring (negative indices)
- (remember that Substring has different semantics from python list slicing!)
- special handling for numbers?

example:
ls
file.txt
cat file.txt

start with:
cat, file.txt
file.txt becomes Union(file.txt, Substring(file.txt, 0, 4), Substring(file.txt, -5, 5))

merge terms into more general forms

- greedy set cover: take a term, find the largest set of terms w/ a non-null intersection (can optimize by initially checking if literals match), add it to the set, remove terms

build a decision tree to select between general forms

- predicates are boolean expressions where literals are words in first input and output + words as substrings + words as reversed substrings
