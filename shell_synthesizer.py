from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass
from shell_autocomplete import *
from shell_dsl import *
import os
import itertools


@dataclass
class SynthesisExample:
    """Represents a single example for synthesis"""

    input_dict: Dict[str, str]  # command and output
    target: str  # next command to predict


class ShellSynthesizer:
    """Main class for synthesizing shell command predictions"""

    def __init__(self, max_depth: int = 10):
        self.max_depth = max_depth
        # Define operators for predicate synthesis
        self.operators = [
            If,  # For conditional logic
            Or,  # For combining predicates
            StringEquals,  # For string comparison
            And,  # For combining predicates
            Not,  # For combining predicates
        ]
        # Constants will be dynamically generated from literals
        self.constants = []

    def _is_number(self, s: str) -> bool:
        """Check if a string represents a number."""
        try:
            float(s)
            return True
        except ValueError:
            return False

    def _generate_alternate_string_expressions(self, word: str, first_input: str, first_output: str) -> Set[Expression]:
        """Generate alternate string expressions for a word based on first input and output."""
        expressions = {ConstantString(word)}  # Start with the word itself

        # Split first input and output into words
        first_input_words = first_input.split()
        first_output_words = first_output.split()

        # Check for whole word matches
        for input_word in first_input_words:
            if word in input_word:
                expressions.add(ConstantString(input_word))

        for output_word in first_output_words:
            if word in output_word:
                expressions.add(ConstantString(output_word))

        # Generate substrings for word boundaries
        for input_word in first_input_words:
            if word in input_word:
                # Find the start and end indices of the word
                start_idx = first_input.find(input_word)
                end_idx = start_idx + len(input_word) - 1
                # Add positive indices
                expressions.add(Substring(StringVariable("command"), start_idx, end_idx))
                # Add negative indices
                expressions.add(Substring(StringVariable("command"), -(len(first_input) - start_idx), -(len(first_input) - end_idx)))

        for output_word in first_output_words:
            if word in output_word:
                # Find the start and end indices of the word
                start_idx = first_output.find(output_word)
                end_idx = start_idx + len(output_word) - 1
                # Add positive indices
                expressions.add(Substring(StringVariable("output"), start_idx, end_idx))
                # Add negative indices
                expressions.add(Substring(StringVariable("output"), -(len(first_output) - start_idx), -(len(first_output) - end_idx)))

        # Special handling for numbers using bottom-up synthesis
        if self._is_number(word):
            try:
                target_num = int(float(word))

                # Collect literals for bottom-up synthesis
                literals = [Number(1)]  # Base literal

                # Find all numbers in first_input and generate ToInt(Substring(...)) expressions
                for input_word in first_input_words:
                    if self._is_number(input_word):
                        # Find indices of this number in the input
                        start_idx = first_input.find(input_word)
                        end_idx = start_idx + len(input_word) - 1

                        # Add positive indices ToInt
                        literals.append(ToInt(Substring(StringVariable("command"), Number(start_idx), Number(end_idx))))
                        # Add negative indices ToInt
                        literals.append(
                            ToInt(
                                Substring(
                                    StringVariable("command"),
                                    Number(-(len(first_input) - start_idx)),
                                    Number(-(len(first_input) - end_idx)),
                                )
                            )
                        )

                # Find all numbers in first_output and generate ToInt(Substring(...)) expressions
                for output_word in first_output_words:
                    if self._is_number(output_word):
                        # Find indices of this number in the output
                        start_idx = first_output.find(output_word)
                        end_idx = start_idx + len(output_word) - 1

                        # Add positive indices ToInt
                        literals.append(ToInt(Substring(StringVariable("output"), Number(start_idx), Number(end_idx))))
                        # Add negative indices ToInt
                        literals.append(
                            ToInt(
                                Substring(
                                    StringVariable("output"),
                                    Number(-(len(first_output) - start_idx)),
                                    Number(-(len(first_output) - end_idx)),
                                )
                            )
                        )

                # Use bottom-up synthesis to generate expressions that evaluate to target_num
                operators = [Plus, Minus]

                # Create a dummy environment for synthesis
                dummy_env = {"command": first_input, "output": first_output}
                input_outputs = [(dummy_env, target_num)]

                # Try to synthesize expressions up to depth 5
                synthesized_expr = bottom_up(5, operators, literals, input_outputs)

                if synthesized_expr is not None:
                    # Wrap in ToString and add to expressions
                    expressions.add(ToString(synthesized_expr))

                # Fallback to original behavior if synthesis fails
                expressions.add(Number(target_num))
                expressions.add(ConstantString(str(target_num)))

            except ValueError:
                pass

        return expressions

    def _generate_terms_from_triple(self, triple: Tuple[str, str, str]) -> Set[Expression]:
        """Generate terms from a triple (input1, output1, input2)."""
        input1, output1, input2 = triple
        terms = set()

        # Start with space-separated words in input2
        input2_words = input2.split()

        # Generate alternate expressions for each word
        word_expressions = []
        for word in input2_words:
            alt_expressions = self._generate_alternate_string_expressions(word, input1, output1)
            word_expressions.append(alt_expressions)

        # Generate all possible combinations of expressions for each word
        for combination in itertools.product(*word_expressions):
            # Create a concatenated expression with spaces between words
            if len(combination) == 1:
                terms.add(combination[0])
            else:
                # Start with the first expression
                result = combination[0]
                # Add space and next expression for each remaining word
                for expr in combination[1:]:
                    result = Concatenate(Concatenate(result, ConstantString(" ")), expr)
                terms.add(result)

        return terms

    def _merge_terms(self, terms: Set[Expression], examples: List[SynthesisExample]) -> List[Expression]:
        """Merge terms into more general forms using greedy set cover."""
        if not terms:
            return []

        # Create a mapping from term to the examples it covers
        term_coverage = {}
        for term in terms:
            covered_examples = set()
            for i, example in enumerate(examples):
                try:
                    # For multi-word terms, we need to check if the term exactly matches the target
                    if term.evaluate(example.input_dict) == example.target:
                        covered_examples.add(i)
                except:
                    continue
            if covered_examples:  # Only include terms that cover at least one example
                term_coverage[term] = covered_examples

        # Greedy set cover
        merged_terms = []
        uncovered_examples = set(range(len(examples)))

        while uncovered_examples and term_coverage:
            # Find term that covers the most uncovered examples
            best_term = max(term_coverage.items(), key=lambda x: len(x[1] & uncovered_examples))

            if not (best_term[1] & uncovered_examples):  # No more coverage possible
                break

            merged_terms.append(best_term[0])
            uncovered_examples -= best_term[1]
            del term_coverage[best_term[0]]

        return merged_terms

    def _parse_log_file(self, log_file: str) -> List[SynthesisExample]:
        """Parse a log file into synthesis examples."""
        examples = []
        current_command = None
        current_output = []

        with open(log_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line.startswith("user: "):
                    # If we have a previous command and output, create an example
                    if current_command is not None:
                        examples.append(
                            SynthesisExample(
                                input_dict={"command": current_command, "output": "\n".join(current_output)},
                                target=line[6:],  # Remove 'user: ' prefix
                            )
                        )

                    current_command = line[6:]  # Remove 'user: ' prefix
                    current_output = []
                else:
                    current_output.append(line)

        return examples

    def _combine_predicate_terms(self, predicate_term_pairs: List[Tuple[Expression, Expression]]) -> Optional[Expression]:
        """
        Combine predicate-term pairs into a single expression using If statements.
        For example, if we have [(p1, t1), (p2, t2)], we return:
        If(p1, t1, If(p2, t2, None))
        """
        if not predicate_term_pairs:
            return None

        # Start with the last pair
        result = predicate_term_pairs[-1][1]  # The term

        # Work backwards through the pairs
        for predicate, term in reversed(predicate_term_pairs[:-1]):
            result = If(predicate, term, result)

        return result

    def synthesize_from_log(self, log_file: str) -> Optional[Expression]:
        """
        Synthesize a program from a log file.
        Returns a single expression that combines all predicate-term pairs, or None if no program can be found.
        """
        # Parse log file into examples
        examples = self._parse_log_file(log_file)

        # Convert examples to triples
        triples = []
        for i in range(len(examples) - 1):
            triple = (examples[i].input_dict["command"], examples[i].input_dict["output"], examples[i + 1].input_dict["command"])
            triples.append(triple)

        # all_terms: list of lists of possible expressions for each word in input2
        all_terms = []
        for triple in triples:
            # term: list of possible expressions for each word in input2
            term = []
            input2_words = triple[2].split()
            for word in input2_words:
                alt_expressions = self._generate_alternate_string_expressions(word, triple[0], triple[1])
                alt_expressions.add(ConstantString(word))  # Add the word itself as a possible expression
                term.append(alt_expressions)
            all_terms.append(term)

        # merge terms into more general forms (find the smallest set of expressions (not terms!!!) that cover all triples)
        # Flatten all_terms into a list of all possible expressions

        # Generate all possible expressions for each triple
        all_expressions = set()
        for term_list in all_terms:
            # Generate all possible combinations of expressions for each word position
            for combination in itertools.product(*term_list):
                # Create a concatenated expression with spaces between words
                if len(combination) == 1:
                    all_expressions.add(combination[0])
                else:
                    # Start with the first expression
                    result = combination[0]
                    # Add space and next expression for each remaining word
                    for expr in combination[1:]:
                        result = Concatenate(Concatenate(result, ConstantString(" ")), expr)
                    all_expressions.add(result)

        # Create a mapping from expression to the triples it covers
        expression_coverage = {}
        for expr in all_expressions:
            covered_triples = set()
            for i, triple in enumerate(triples):
                try:
                    # Check if this expression generates the target command
                    if expr.evaluate({"command": triple[0], "output": triple[1]}) == triple[2]:
                        covered_triples.add(i)
                except:
                    continue
            if covered_triples:  # Only include expressions that cover at least one triple
                expression_coverage[expr] = covered_triples

        # Greedy set cover to find minimal set of expressions
        merged_terms = []
        uncovered_triples = set(range(len(triples)))
        coverage_copy = expression_coverage.copy()
        while uncovered_triples and coverage_copy:
            # Find expression that covers most uncovered triples
            best_expr = max(coverage_copy.items(), key=lambda x: len(x[1] & uncovered_triples))

            if not (best_expr[1] & uncovered_triples):  # No more coverage possible
                break

            merged_terms.append(best_expr[0])
            uncovered_triples -= best_expr[1]
            del coverage_copy[best_expr[0]]

        # Generate literals for each triple
        literals_per_triple = [self._generate_predicate_literals(*triple) for triple in triples]

        # For each merged term, find a predicate that matches its coverage
        predicate_term_pairs = []
        for term in merged_terms:
            covered_indices = expression_coverage[term]  # Use stored coverage instead
            predicate = self._enumerate_predicate_for_term(term, triples, covered_indices, literals_per_triple)
            if predicate is not None:
                predicate_term_pairs.append((predicate, term))

        if not predicate_term_pairs:
            return None

        # Combine all predicate-term pairs into a single expression
        return self._combine_predicate_terms(predicate_term_pairs)

    def validate(self, log_file: str) -> Tuple[int, int]:
        """
        Validate synthesis by treating the log file as live input.
        For each command that's predictable (comes after the second command and has been seen before),
        try to predict it using a program generated from previous entries.

        Returns a tuple of (correct_predictions, total_predictable_commands).
        """
        # Parse log file into examples
        examples = self._parse_log_file(log_file)
        print("examples:")
        print(examples)

        if len(examples) < 3:
            return 0, 0

        correct_predictions = 0
        total_predictable = 0

        # For each command starting from the third one (index 2)
        for i in range(2, len(examples)):
            current_command = examples[i].input_dict["command"]

            # Check if this command is "seen before" based on previous entries
            if self._is_command_seen_before(current_command, examples[:i]):
                total_predictable += 1

                # Generate program from previous entries
                try:
                    # Create a temporary log with just the previous entries
                    temp_examples = examples[:i]
                    program = self._synthesize_from_examples(temp_examples)
                    print("--------------------------------")
                    print("test examples:")
                    print(temp_examples)
                    print("expected command:")
                    print(current_command)
                    print("predicted program:")
                    print(program.pretty_print())
                    print("--------------------------------")

                    if program is not None:
                        # Try to predict the current command
                        # Use the previous command and output as context
                        prev_example = examples[i - 1]
                        env = prev_example.input_dict.copy()

                        try:
                            prediction = program.evaluate(env)
                            if prediction == current_command:
                                correct_predictions += 1
                        except:
                            # Program failed to evaluate
                            pass
                except:
                    # Program synthesis failed
                    pass

        return correct_predictions, total_predictable

    def _is_command_seen_before(self, command: str, previous_examples: List[SynthesisExample]) -> bool:
        """
        Check if a command can be constructed using substrings from previous entries.
        A command is "seen before" if all its words can be found as substrings in previous inputs/outputs.
        """
        command_words = command.split()

        # Collect all text from previous inputs and outputs
        previous_text = ""
        for example in previous_examples:
            previous_text += " " + example.input_dict["command"]
            previous_text += " " + example.input_dict["output"]

        # Check if each word in the command can be found as a substring in previous text
        for word in command_words:
            # For numbers, check if they can be constructed from numbers in previous text
            if self._is_number(word):
                # Extract all numbers from previous text
                prev_numbers = []
                for prev_word in previous_text.split():
                    if self._is_number(prev_word):
                        try:
                            prev_numbers.append(int(float(prev_word)))
                        except:
                            pass

                # Check if the target number can be constructed using arithmetic operations
                target_num = int(float(word))
                if not self._can_construct_number(target_num, prev_numbers):
                    return False
            else:
                # For non-numbers, check if the word appears as a substring
                if word not in previous_text:
                    return False

        return True

    def _can_construct_number(self, target: int, available_numbers: List[int]) -> bool:
        """
        Check if target number can be constructed using available numbers and basic arithmetic.
        This is a simplified check - just see if target appears in available numbers,
        or can be made with simple operations.
        """
        if target in available_numbers:
            return True

        # Check simple arithmetic combinations (addition, subtraction)
        for i, num1 in enumerate(available_numbers):
            for j, num2 in enumerate(available_numbers):
                if i != j:  # Don't use the same number twice
                    if num1 + num2 == target or num1 - num2 == target or num2 - num1 == target:
                        return True

        return False

    def _synthesize_from_examples(self, examples: List[SynthesisExample]) -> Optional[Expression]:
        """
        Synthesize a program from a list of examples (similar to synthesize_from_log but works with examples).
        """
        if len(examples) < 2:
            return None

        # Convert examples to triples
        triples = []
        for i in range(len(examples) - 1):
            triple = (examples[i].input_dict["command"], examples[i].input_dict["output"], examples[i + 1].input_dict["command"])
            triples.append(triple)

        # Generate all possible expressions for each triple
        all_expressions = set()
        for triple in triples:
            input1, output1, input2 = triple
            input2_words = input2.split()

            # Generate alternate expressions for each word
            word_expressions = []
            for word in input2_words:
                alt_expressions = self._generate_alternate_string_expressions(word, input1, output1)
                alt_expressions.add(ConstantString(word))
                word_expressions.append(alt_expressions)

            # Generate all possible combinations of expressions for each word position
            for combination in itertools.product(*word_expressions):
                # Create a concatenated expression with spaces between words
                if len(combination) == 1:
                    all_expressions.add(combination[0])
                else:
                    # Start with the first expression
                    result = combination[0]
                    # Add space and next expression for each remaining word
                    for expr in combination[1:]:
                        result = Concatenate(Concatenate(result, ConstantString(" ")), expr)
                    all_expressions.add(result)

        # Create a mapping from expression to the triples it covers
        expression_coverage = {}
        for expr in all_expressions:
            covered_triples = set()
            for i, triple in enumerate(triples):
                try:
                    # Check if this expression generates the target command
                    if expr.evaluate({"command": triple[0], "output": triple[1]}) == triple[2]:
                        covered_triples.add(i)
                except:
                    continue
            if covered_triples:  # Only include expressions that cover at least one triple
                expression_coverage[expr] = covered_triples

        # Greedy set cover to find minimal set of expressions
        merged_terms = []
        uncovered_triples = set(range(len(triples)))
        coverage_copy = expression_coverage.copy()
        while uncovered_triples and coverage_copy:
            # Find expression that covers most uncovered triples
            best_expr = max(coverage_copy.items(), key=lambda x: len(x[1] & uncovered_triples))

            if not (best_expr[1] & uncovered_triples):  # No more coverage possible
                break

            merged_terms.append(best_expr[0])
            uncovered_triples -= best_expr[1]
            del coverage_copy[best_expr[0]]

        # Generate literals for each triple
        literals_per_triple = [self._generate_predicate_literals(*triple) for triple in triples]

        # For each merged term, find a predicate that matches its coverage
        predicate_term_pairs = []
        for term in merged_terms:
            covered_indices = expression_coverage[term]
            predicate = self._enumerate_predicate_for_term(term, triples, covered_indices, literals_per_triple)
            if predicate is not None:
                predicate_term_pairs.append((predicate, term))

        if not predicate_term_pairs:
            return None

        # Combine all predicate-term pairs into a single expression
        return self._combine_predicate_terms(predicate_term_pairs)

    def _generate_predicate_literals(self, input1: str, output: str, input2: str) -> Set[Expression]:
        """
        Generate literals for predicates:
        - Contains(input1, literal) for words in input1
        - Contains(output, literal) for words in output
        - Contains(input1, literal) for words in input2 that appear as part of a word in input1
        - Contains(output, literal) for words in input2 that appear as part of a word in output
        - Substring(input1, indices) for positive/negative indices of words in input1
        - Substring(output, indices) for positive/negative indices of words in output
        - Substring(input1, indices) for positive/negative indices of words in input2 that appear as part of a word in input1
        - Substring(output, indices) for positive/negative indices of words in input2 that appear as part of a word in output
        Indices are either both positive or both negative.
        """
        literals = set()
        input1_words = input1.split()
        input2_words = input2.split()
        output_words = output.split()

        # Generate Contains literals for words in input1
        for word in input1_words:
            literals.add(Contains(StringVariable("command"), ConstantString(word)))

        # Generate Contains literals for words in output
        for word in output_words:
            literals.add(Contains(StringVariable("output"), ConstantString(word)))

        # Generate Contains literals for words in input2 that appear as part of a word in input1
        for word in input2_words:
            for input1_word in input1_words:
                if word in input1_word:
                    literals.add(Contains(StringVariable("command"), ConstantString(word)))

        # Generate Contains literals for words in input2 that appear as part of a word in output
        for word in input2_words:
            for output_word in output_words:
                if word in output_word:
                    literals.add(Contains(StringVariable("output"), ConstantString(word)))

        # # Generate Substring literals for words in input1 (both positive and negative indices)
        # for word in input1_words:
        #     start = input1.find(word)
        #     end = start + len(word) - 1
        #     # Positive indices
        #     literals.add(Substring(StringVariable("command"), Number(start), Number(end)))
        #     # Negative indices
        #     literals.add(Substring(StringVariable("command"), Number(-(len(input1) - start)), Number(-(len(input1) - end))))

        # # Generate Substring literals for words in output (both positive and negative indices)
        # for word in output_words:
        #     start = output.find(word)
        #     end = start + len(word) - 1
        #     # Positive indices
        #     literals.add(Substring(StringVariable("output"), Number(start), Number(end)))
        #     # Negative indices
        #     literals.add(Substring(StringVariable("output"), Number(-(len(output) - start)), Number(-(len(output) - end))))

        # # Generate Substring literals for words in input2 that appear as part of a word in input1
        # for word in input2_words:
        #     for input1_word in input1_words:
        #         if word in input1_word:
        #             start = input1.find(input1_word)
        #             word_start = input1_word.find(word)
        #             word_end = word_start + len(word) - 1
        #             # Positive indices
        #             literals.add(Substring(StringVariable("command"), Number(start + word_start), Number(start + word_end)))
        #             # Negative indices
        #             literals.add(
        #                 Substring(
        #                     StringVariable("command"),
        #                     Number(-(len(input1) - (start + word_start))),
        #                     Number(-(len(input1) - (start + word_end))),
        #                 )
        #             )

        # # Generate Substring literals for words in input2 that appear as part of a word in output
        # for word in input2_words:
        #     for output_word in output_words:
        #         if word in output_word:
        #             start = output.find(output_word)
        #             word_start = output_word.find(word)
        #             word_end = word_start + len(word) - 1
        #             # Positive indices
        #             literals.add(Substring(StringVariable("output"), Number(start + word_start), Number(start + word_end)))
        #             # Negative indices
        #             literals.add(
        #                 Substring(
        #                     StringVariable("output"),
        #                     Number(-(len(output) - (start + word_start))),
        #                     Number(-(len(output) - (start + word_end))),
        #                 )
        #             )

        return literals

    def _enumerate_predicate_for_term(self, term, triples, covered_indices, literals_per_triple):
        """
        Enumerate predicates using bottom_up implementation from shell_dsl.
        Returns a predicate that returns True for all covered_indices and False for the others.
        """
        # For each triple, build the environment
        environments = []
        for triple in triples:
            input1, output, input2 = triple
            environments.append({"command": input1, "output": output, "target": input2})

        # Build the target boolean vector
        target_vector = [i in covered_indices for i in range(len(triples))]

        # Create input-output pairs for bottom_up
        input_outputs = [(env, target) for env, target in zip(environments, target_vector)]

        # Collect all literals from the covered triples
        all_literals = set()
        for i in covered_indices:
            all_literals.update(literals_per_triple[i])

        # Update constants with the literals
        self.constants = list(all_literals)

        # Use bottom_up to find a predicate that matches the target vector
        predicate = bottom_up(self.max_depth, self.operators, self.constants, input_outputs)

        return predicate


def test_synthesis():
    """Test the synthesis engine on example log files."""
    # Create synthesizer
    synthesizer = ShellSynthesizer(max_depth=30)

    # Test on simple.log
    print("\nTesting on simple.log:")
    program = synthesizer.synthesize_from_log("test/simple.log")
    if program:
        print("Synthesized program:")
        print(program.pretty_print())

    # Validate using live prediction
    correct, total = synthesizer.validate("test/simple.log")
    print(f"Live prediction results: {correct}/{total} correct predictions")

    # Test on sample.log
    print("\nTesting on sample.log:")
    program = synthesizer.synthesize_from_log("test/sample.log")
    if program:
        print("Synthesized program:")
        print(program.pretty_print())

    # Validate using live prediction
    correct, total = synthesizer.validate("test/sample.log")
    print(f"Live prediction results: {correct}/{total} correct predictions")

    # Test on assignment.log
    print("\nTesting on assignment.log:")
    program = synthesizer.synthesize_from_log("test/assignment.log")
    if program:
        print("Synthesized program:")
        print(program.pretty_print())

    # Validate using live prediction
    correct, total = synthesizer.validate("test/assignment.log")
    print(f"Live prediction results: {correct}/{total} correct predictions")


if __name__ == "__main__":
    test_synthesis()
