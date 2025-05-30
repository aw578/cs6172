from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass
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

        # Generate SubWord expressions for words in input
        for i, input_word in enumerate(first_input_words):
            if word in input_word:
                # Add positive index
                expressions.add(SubWord(StringVariable("command"), Number(i)))
                # Add negative index
                expressions.add(SubWord(StringVariable("command"), Number(-(len(first_input_words) - i))))

        # Generate SubWord expressions for words in output
        for i, output_word in enumerate(first_output_words):
            if word in output_word:
                # Add positive index
                expressions.add(SubWord(StringVariable("output"), Number(i)))
                # Add negative index
                expressions.add(SubWord(StringVariable("output"), Number(-(len(first_output_words) - i))))

        # Special handling for numbers using bottom-up synthesis
        if self._is_number(word):
            try:
                target_num = int(float(word))

                # Collect literals for bottom-up synthesis
                literals = [Number(1)]  # Base literal

                # Find all numbers in first_input and generate ToInt(SubWord(...)) expressions
                for i, input_word in enumerate(first_input_words):
                    if self._is_number(input_word):
                        # Add positive index ToInt
                        literals.append(ToInt(SubWord(StringVariable("command"), Number(i))))
                        # Add negative index ToInt
                        literals.append(ToInt(SubWord(StringVariable("command"), Number(-(len(first_input_words) - i)))))

                # Find all numbers in first_output and generate ToInt(SubWord(...)) expressions
                for i, output_word in enumerate(first_output_words):
                    if self._is_number(output_word):
                        # Add positive index ToInt
                        literals.append(ToInt(SubWord(StringVariable("output"), Number(i))))
                        # Add negative index ToInt
                        literals.append(ToInt(SubWord(StringVariable("output"), Number(-(len(first_output_words) - i)))))

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
                if not line or line.startswith("#"):  # Skip empty lines and comments
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

        # Convert examples to triples and handle duplicates
        triples = []
        triple_to_indices = {}  # Maps (input1, output1) -> list of triple indices

        for i in range(len(examples) - 1):
            triple = (examples[i].input_dict["command"], examples[i].input_dict["output"], examples[i + 1].input_dict["command"])
            triples.append(triple)

            # Track which triples have the same input/output pair
            input_output_pair = (triple[0], triple[1])
            if input_output_pair not in triple_to_indices:
                triple_to_indices[input_output_pair] = []
            triple_to_indices[input_output_pair].append(i)

        # Handle duplicate input/output pairs
        self._handle_duplicate_pairs(triples, triple_to_indices, examples)

        # Optimized merging process
        merged_terms = self._optimized_merge_terms(triples)

        # Generate literals for each triple
        literals_per_triple = [self._generate_predicate_literals(*triple) for triple in triples]

        # For each merged term, find a predicate that matches its coverage
        predicate_term_pairs = []
        for term, covered_indices in merged_terms:
            predicate = self._enumerate_predicate_for_term(term, triples, covered_indices, literals_per_triple)
            if predicate is not None:
                predicate_term_pairs.append((predicate, term))

        if not predicate_term_pairs:
            return None

        # Combine all predicate-term pairs into a single expression
        return self._combine_predicate_terms(predicate_term_pairs)

    def _optimized_merge_terms(self, triples: List[Tuple[str, str, str]]) -> List[Tuple[Expression, Set[int]]]:
        """
        Optimized merging that calculates coverage per word position and uses set operations
        to avoid enumerating all possible expressions.

        Returns list of (expression, covered_triple_indices) tuples.
        """
        if not triples:
            return []

        # Calculate maximum number of words across all target commands
        max_words = max(len(triple[2].split()) for triple in triples)

        # For each word position, calculate coverage of each sub-expression
        word_position_coverage = []
        word_position_expressions = []

        for word_pos in range(max_words):
            position_coverage = {}  # sub_expr -> set of triple indices it covers for this position
            position_expressions = set()

            # First, collect all expressions that could potentially work for this position
            all_expressions_for_position = set()

            for triple_idx, triple in enumerate(triples):
                input1, output1, input2 = triple
                target_words = input2.split()

                # Skip if this triple doesn't have enough words
                if word_pos >= len(target_words):
                    continue

                target_word = target_words[word_pos]

                # Generate alternate expressions for this word
                alt_expressions = self._generate_alternate_string_expressions(target_word, input1, output1)
                alt_expressions.add(ConstantString(target_word))

                all_expressions_for_position.update(alt_expressions)

            # Now test each expression against ALL triples to determine true coverage
            for expr in all_expressions_for_position:
                for triple_idx, triple in enumerate(triples):
                    input1, output1, input2 = triple
                    target_words = input2.split()

                    # Skip if this triple doesn't have enough words
                    if word_pos >= len(target_words):
                        continue

                    target_word = target_words[word_pos]

                    try:
                        if expr.evaluate({"command": input1, "output": output1}) == target_word:
                            if expr not in position_coverage:
                                position_coverage[expr] = set()
                            position_coverage[expr].add(triple_idx)
                            position_expressions.add(expr)
                    except:
                        continue

            word_position_coverage.append(position_coverage)
            word_position_expressions.append(position_expressions)

        # Remove duplicate sub-expressions with identical coverage at each position
        optimized_word_positions = []
        for pos_idx in range(max_words):
            coverage_to_expr = {}  # coverage_tuple -> representative expression

            for expr, coverage in word_position_coverage[pos_idx].items():
                coverage_tuple = tuple(sorted(coverage))
                if coverage_tuple not in coverage_to_expr:
                    coverage_to_expr[coverage_tuple] = expr
                else:
                    # If we already have an expression for this coverage, prefer the one with ToInt(SubWord())
                    existing_expr = coverage_to_expr[coverage_tuple]
                    if self._has_toint(expr) and not self._has_toint(existing_expr):
                        coverage_to_expr[coverage_tuple] = expr

            # Build optimized mapping for this position
            optimized_position = {}
            for coverage_tuple, expr in coverage_to_expr.items():
                optimized_position[expr] = set(coverage_tuple)

            optimized_word_positions.append(optimized_position)

        # Calculate total combinations before optimization
        original_combinations = 1
        for pos_coverage in word_position_coverage:
            if pos_coverage:
                original_combinations *= len(pos_coverage)

        optimized_combinations = 1
        for opt_pos in optimized_word_positions:
            if opt_pos:
                optimized_combinations *= len(opt_pos)

        # Now generate expressions using the optimized sets
        # We'll use a recursive approach to build expressions of different lengths
        result_terms = []
        max_terms_to_generate = 10000  # Limit to prevent explosion

        def generate_expressions(word_pos: int, current_coverage: Set[int], current_expr_parts: List[Expression]):
            """Recursively generate expressions by adding words at each position."""
            if len(result_terms) >= max_terms_to_generate:
                return  # Early termination

            if word_pos >= max_words:
                # We've built a complete expression
                if current_coverage and current_expr_parts:
                    # Build the concatenated expression
                    if len(current_expr_parts) == 1:
                        final_expr = current_expr_parts[0]
                    else:
                        final_expr = current_expr_parts[0]
                        for part in current_expr_parts[1:]:
                            final_expr = Concatenate(Concatenate(final_expr, ConstantString(" ")), part)

                    result_terms.append((final_expr, current_coverage))
                return

            # Check if any triple requires more words
            has_triples_with_more_words = any(len(triples[i][2].split()) > word_pos for i in current_coverage)

            if not has_triples_with_more_words:
                # No more words needed, finalize current expression
                if current_coverage and current_expr_parts:
                    if len(current_expr_parts) == 1:
                        final_expr = current_expr_parts[0]
                    else:
                        final_expr = current_expr_parts[0]
                        for part in current_expr_parts[1:]:
                            final_expr = Concatenate(Concatenate(final_expr, ConstantString(" ")), part)

                    result_terms.append((final_expr, current_coverage))
                return

            # Try each sub-expression at this word position
            if word_pos < len(optimized_word_positions):
                for expr, expr_coverage in optimized_word_positions[word_pos].items():
                    # Calculate intersection of current coverage with this expression's coverage
                    new_coverage = current_coverage & expr_coverage if current_coverage else expr_coverage

                    if new_coverage:  # Only continue if there's still coverage
                        generate_expressions(word_pos + 1, new_coverage, current_expr_parts + [expr])

        # Start generation with all possible first words
        if optimized_word_positions:
            for expr, coverage in optimized_word_positions[0].items():
                if len(result_terms) >= max_terms_to_generate:
                    break
                generate_expressions(1, coverage, [expr])

        # Apply greedy set cover to find minimal set
        if not result_terms:
            return []

        # Greedy set cover
        final_terms = []
        uncovered_triples = set(range(len(triples)))
        available_terms = result_terms.copy()

        while uncovered_triples and available_terms:
            # Find term that covers most uncovered triples
            best_term = max(available_terms, key=lambda x: len(x[1] & uncovered_triples))

            if not (best_term[1] & uncovered_triples):  # No more coverage possible
                break

            final_terms.append(best_term)
            uncovered_triples -= best_term[1]
            available_terms.remove(best_term)

        return final_terms

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
                    # print("--------------------------------")
                    # print("test examples:")
                    # print(temp_examples)
                    # print("expected command:")
                    # print(current_command)
                    # print("predicted program:")
                    # print(program.pretty_print())
                    # print("--------------------------------")

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

        # Convert examples to triples and handle duplicates
        triples = []
        triple_to_indices = {}  # Maps (input1, output1) -> list of triple indices

        for i in range(len(examples) - 1):
            triple = (examples[i].input_dict["command"], examples[i].input_dict["output"], examples[i + 1].input_dict["command"])
            triples.append(triple)

            # Track which triples have the same input/output pair
            input_output_pair = (triple[0], triple[1])
            if input_output_pair not in triple_to_indices:
                triple_to_indices[input_output_pair] = []
            triple_to_indices[input_output_pair].append(i)

        # Handle duplicate input/output pairs
        self._handle_duplicate_pairs(triples, triple_to_indices, examples)

        # Use optimized merging process
        merged_terms = self._optimized_merge_terms(triples)

        # Generate literals for each triple
        literals_per_triple = [self._generate_predicate_literals(*triple) for triple in triples]

        # For each merged term, find a predicate that matches its coverage
        predicate_term_pairs = []
        for term, covered_indices in merged_terms:
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
        - SubWord(input1, n) for words in input1
        - SubWord(output, n) for words in output
        - SubWord(input1, n) for words in input2 that appear as part of a word in input1 (temporarily removed)
        - SubWord(output, n) for words in input2 that appear as part of a word in output (temporarily removed)
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

        # Generate SubWord literals for words in input1 (both positive and negative indices)
        for i, word in enumerate(input1_words):
            # Positive indices
            literals.add(SubWord(StringVariable("command"), Number(i)))
            # Negative indices
            literals.add(SubWord(StringVariable("command"), Number(-(len(input1_words) - i))))

        # Generate SubWord literals for words in output (both positive and negative indices)
        for i, word in enumerate(output_words):
            # Positive indices
            literals.add(SubWord(StringVariable("output"), Number(i)))
            # Negative indices
            literals.add(SubWord(StringVariable("output"), Number(-(len(output_words) - i))))

        # # Generate SubWord literals for words in input2 that appear as part of a word in input1
        # for word in input2_words:
        #     for i, input1_word in enumerate(input1_words):
        #         if word in input1_word:
        #             # Positive indices
        #             literals.add(SubWord(StringVariable("command"), Number(i)))
        #             # Negative indices
        #             literals.add(SubWord(StringVariable("command"), Number(-(len(input1_words) - i))))

        # # Generate SubWord literals for words in input2 that appear as part of a word in output
        # for word in input2_words:
        #     for i, output_word in enumerate(output_words):
        #         if word in output_word:
        #             # Positive indices
        #             literals.add(SubWord(StringVariable("output"), Number(i)))
        #             # Negative indices
        #             literals.add(SubWord(StringVariable("output"), Number(-(len(output_words) - i))))

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

    def _handle_duplicate_pairs(
        self, triples: List[Tuple[str, str, str]], triple_to_indices: Dict[Tuple[str, str], List[int]], examples: List[SynthesisExample]
    ) -> None:
        """
        Handle cases where the same input/output pair produces different input2s.
        If using literals from the previous input/output pair isn't enough, keep only the most recent one.
        """
        # Find duplicate input/output pairs that produce different input2s
        for input_output_pair, indices in triple_to_indices.items():
            if len(indices) > 1:
                # Check if they produce different input2s
                input2s = [triples[i][2] for i in indices]
                if len(set(input2s)) > 1:  # Different input2s for same input/output
                    most_recent_idx = max(indices)
                    indices_to_remove = [i for i in indices if i != most_recent_idx]

                    # Remove from triples (in reverse order to maintain indices)
                    for idx in sorted(indices_to_remove, reverse=True):
                        del triples[idx]

                    # Update indices in triple_to_indices
                    offset = 0
                    for i in sorted(indices_to_remove):
                        for key, val_list in triple_to_indices.items():
                            triple_to_indices[key] = [v - (1 if v > i - offset else 0) for v in val_list if v != i - offset]
                        offset += 1

    def _has_toint(self, expr: Expression) -> bool:
        """Check if an expression contains a ToInt term."""
        if isinstance(expr, ToInt):
            return True
        elif isinstance(expr, ToString):
            # Recursively check the value inside ToString
            return self._has_toint(expr.value)
        elif hasattr(expr, "arguments"):
            # Recursively check all arguments of the expression
            for arg in expr.arguments():
                if self._has_toint(arg):
                    return True
        return False


def test_synthesis():
    """Test the synthesis engine on all log files in the test directory."""
    # Create synthesizer
    synthesizer = ShellSynthesizer(max_depth=7)

    # Get all .log files in test directory
    import glob

    log_files = glob.glob("test/*.log")

    # Test each log file
    for log_file in log_files:
        print(f"\nTesting on {log_file}:")
        program = synthesizer.synthesize_from_log(log_file)
        if program:
            print("Synthesized program:")
            print(program.pretty_print(), flush=True)

        # Validate using live prediction
        correct, total = synthesizer.validate(log_file)
        print(f"Live prediction results: {correct}/{total} correct predictions", flush=True)


if __name__ == "__main__":
    test_synthesis()
