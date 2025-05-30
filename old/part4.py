from part1 import *
# from part2 import *
from part3 import *

import re
import time
import frozendict

class Concatenate(Expression):
    return_type = "str"
    argument_types = ["str", "str"]

    def __init__(self, left, right):
        self.left, self.right = left, right

    def __str__(self):
        return f"Concatenate({self.left}, {self.right})"

    def pretty_print(self):
        return f"{self.left.pretty_print()} + {self.right.pretty_print()}"

    def extension(self):
        return [
            Concatenate(left, right)
            for left in self.left.extension()
            for right in self.right.extension()
        ]

    def version_space_size(self):
        return self.left.version_space_size() * self.right.version_space_size()

    def minimum_cost_member_of_extension(self):
        return Concatenate(
            self.left.minimum_cost_member_of_extension(),
            self.right.minimum_cost_member_of_extension(),
        )

    def arguments(self):
        return [self.left, self.right]

    def evaluate(self, environment):
        return self.left.evaluate(environment) + self.right.evaluate(environment)


class ConstantString(Expression):
    return_type = "str"
    argument_types = []

    def __init__(self, content):
        self.content = content

    def __str__(self):
        return f'ConstantString("{self.content}")'

    def pretty_print(self):
        return f'"{self.content}"'

    def extension(self):
        return [self]

    def cost(self):
        return len(self.content) ** 2

    def version_space_size(self):
        return 1

    def minimum_cost_member_of_extension(self):
        return self

    def arguments(self):
        return []

    def evaluate(self, environment):
        return self.content


class Substring(Expression):
    return_type = "str"
    argument_types = ["str", "int", "int"]

    def __init__(self, the_string, left, right):
        self.the_string, self.left, self.right = the_string, left, right

    def __str__(self):
        return f"Substring({self.the_string}, {self.left}, {self.right})"

    def pretty_print(self):
        return f"Substring({self.the_string.pretty_print()}, {self.left.pretty_print()}, {self.right.pretty_print()})"

    def evaluate(self, environment):
        """
        Slightly different semantics from ordinary python list slicing:
        We think of the indices as referring to characters in the string, rather than referring to places in between characters
        The extracted substring is the span between the start and end indices, **inclusive** (so it includes the ending index)
        This causes the start and end indices to be treated symmetrically - specifically both `the_string[left]` and `the_string[right]` will be in the output
        If an index is negative, we make it positive by calculating `len(the_string) + the_index`
        As a consequence, `Substring(string, 0, -1)` gives the entire string.
        """
        the_string = self.the_string.evaluate(environment)
        left = self.left.evaluate(environment)
        right = self.right.evaluate(environment)

        # if the index = -1, that refers to the last character
        if left < 0:
            left = len(the_string) + left
        if right < 0:
            right = len(the_string) + right

        return the_string[left : right + 1]

    def extension(self):
        return [
            Substring(str, left, right)
            for str in self.the_string.extension()
            for left in self.left.extension()
            for right in self.right.extension()
        ]

    def minimum_cost_member_of_extension(self):
        return Substring(
            self.the_string.minimum_cost_member_of_extension(),
            self.left.minimum_cost_member_of_extension(),
            self.right.minimum_cost_member_of_extension(),
        )

    def version_space_size(self):
        return (
            self.the_string.version_space_size()
            * self.left.version_space_size()
            * self.right.version_space_size()
        )

    def arguments(self):
        return [self.the_string, self.left, self.right]


class StringVariable(Expression):
    return_type = "str"
    argument_types = []

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"StringVariable('{self.name}')"

    def pretty_print(self):
        return self.name

    def extension(self):
        return [self]

    def version_space_size(self):
        return 1

    def minimum_cost_member_of_extension(self):
        return self

    def arguments(self):
        return []

    def evaluate(self, environment):
        return environment[self.name]


def test_evaluation(verbose=False):
    x = "(555) 993-4777"
    y = "993"
    expressions = [
        Substring(ConstantString(x), Number(6), Number(8)),
        Substring(ConstantString(x), Number(-8), Number(8)),
        Substring(ConstantString(x), Number(6), Number(-6)),
        Substring(ConstantString(x), Number(-8), Number(-6)),
    ]

    made_a_mistake = False

    for e in expressions:
        if e.evaluate({}) != y:
            if verbose:
                print(f" [!] failed evaluation on {e}")
                print("     expected:", y)
                print("     got:", e.evaluate({}))
            made_a_mistake = True

    if made_a_mistake:
        points = 0
    else:
        points = 5

    print(f" [+] 4.1, text editing evaluation, +{points}/5 points")

    return points


if __name__ == "__main__":
    test_evaluation(verbose=True)


class Union(Expression):
    def __init__(self, members):
        self.members = members

    def __str__(self):
        return "Union(" + ", ".join([str(m) for m in self.members]) + ")"

    def pretty_print(self):
        return "Union(" + ", ".join([m.pretty_print() for m in self.members]) + ")"

    def extension(self):
        return [
            expression for member in self.members for expression in member.extension()
        ]

    def evaluate(self, environment):
        assert False, "cannot evaluate union"

    @staticmethod
    def make(members):
        """
        Helper function for building unions
        Flattens nested unions, and prevents the creation of singleton unions
        """
        members = [
            expression
            for member in members
            for expression in (
                member.members if isinstance(member, Union) else [member]
            )
        ]
        if len(members) == 1:
            return members[0]
        return Union(members)

    def minimum_cost_member_of_extension(self):
        return min(
            [member.minimum_cost_member_of_extension() for member in self.members],
            key=lambda expression: expression.cost(),
        )

    def version_space_size(self):
        return sum(member.version_space_size() for member in self.members)


def test_version_spaces(verbose=False):
    total_points = 0

    v = Union(
        [
            Substring(
                StringVariable("x"),
                Union([Number(2), Number(1)]),
                Union([Number(-2), Number(-1)]),
            ),
            ConstantString("hello world"),
        ]
    )

    if v.version_space_size() != 5:
        if verbose:
            print("problem with version_space_size")
    else:
        total_points += 1

    if len(v.extension()) != 5:
        if verbose:
            print("problem with extension")
    else:
        total_points += 2

    v = Union([Concatenate(v, v), StringVariable("y")])
    if str(v.minimum_cost_member_of_extension()) != "StringVariable('y')":
        if verbose:
            print("problem with `minimum_cost_member_of_extension`")
    else:
        total_points += 2

    print(f" [+] 4.2, version space evaluation, +{total_points}/5 points")
    return total_points


if __name__ == "__main__":
    test_version_spaces(verbose=True)


def generate_substring(environment, target_output):
    """
    environment: a dictionary mapping a variable name to a value. Both names and values are strings
    target_output: the target output of the `Substring` program when run on `environment`
    returns: a version space containing all Substring operators which can map the `environment` to the `target_output`
    """
    result = []
    for key in environment.keys():
        substr = environment[key]
        pos = substr.find(target_output)
        while pos != -1:
            l = generate_position(substr, pos)
            r = generate_position(substr, pos + len(target_output) - 1)
            result.append(Substring(StringVariable(key), l, r))
            substr = target_output[pos + len(target_output) :]
            pos = substr.find(target_output)
    if len(result) == 0:
        return None
    if len(result) == 1:
        return result[0]
    return Union(result)


def generate_position(string, i):
    """
    string: a string that was provided as an input in the specification
    i: an index of a character in that string
    returns: a version space of programs which will index that character"""
    result = [Number(i), Number(-1 * (len(string) - i))]
    return Union(result)


def extra_credit_witness_synthesize(
    specification, components, dynamic_programming_table=None
):
    """
        specification: a spec on what the target program should be
                       such specifications are either a single program, i.e. an `Expression`, meaning that the specified program has to be that exact expression
                       OR, a specification is a tuple of (input, target_output)

        components: a list of classes, such as `Number`, `Concatenate`, ...
                    these classes must implement a static method called `witness`
                    `witness` takes as input a specification and returns a disjunction of conjunctions of specification on the arguments to the constructor class of a DSL component.
                    Disjunctions and conjunctions are represented as lists.
                    For example, given class `K` for a DSL component w/ arity 2, if the return value of `K.witness(spec)` is of the form:
                        [ [spec_1_1,spec_1_2], [spec_2_1,spec_2_2], [spec_3_1,spec_3_2]]
                    this means that `K(expr_i_1, expr_i_2)` satisfies `spec` whenever `expr_i_1` satisfies `spec_i_1` and `expr_i_2` satisfies `spec_i_2`, where i=1..3

        For example, you could define `Concatenate.witness` as:
    ```
        @staticmethod
        def witness(specification):
            environment, target_output = specification # unpack the spec
            if not isinstance(target_output, str): # `Concatenate` can only make string outputs
                return [] # empty disjunction - i.e. impossible/false/bottom

            possibilities = [] # a disjunction of different possible ways of concatenating to make output
            for size_of_prefix in range(1, len(target_output)):
                prefix = target_output[:size_of_prefix]
                suffix = target_output[size_of_prefix:]
                assert target_output == prefix + suffix

                possibilities.append([[environment, prefix], # spec on prefix...
                                      [environment, suffix]]) # ...conjount w/ spec on suffix

            return possibilities # returning a (list of) disjunctions, each of which is a (list of) conjunctions
    ```

        In order to handle constant constructors `K`, such as `Number` and `ConstantString`, we recommend also allowing `witness` to just return an entire instance of type `K` rather than return conjunctions of specs.

        For example, you could define `ConstantString.witness` as:
    ```
         @staticmethod
         def witness(specification):
             inputs, target_output = specification
             if isinstance(target_output, str):
                 return [ConstantString(target_output)]
             else:
                 return []
    ```
    """
    assert False, "implement as part of extra credit (OPTIONAL)"


def flashfill_one_example(environment, target_output, dynamic_programming_table=None):

    if dynamic_programming_table is None:
        dynamic_programming_table = {}

    environment = frozendict.frozendict(environment)

    def build(begin, end):
        if target_output[begin:end] in dynamic_programming_table:
            return dynamic_programming_table[target_output[begin:end]]
        result = [ConstantString(target_output[begin:end])]
        substr = generate_substring(environment, target_output[begin:end])
        if substr:
            result.append(substr)
        for mid in range(begin + 1, end):
            head = build(begin, mid)
            tail = build(mid, end)
            result.append(Concatenate(head, tail))
        result = Union.make(result)
        dynamic_programming_table[target_output[begin:end]] = result
        return result

    result = build(0, len(target_output))
    return result


def flashfill_one_example(environment, target_output, dynamic_programming_table=None):
    """
    environment: a dictionary mapping a variable name to a value. Both names and values are strings
    target_output: the target output of the program when run on `environment`
    """

    if dynamic_programming_table is None: dynamic_programming_table = {}

    environment = frozendict.frozendict(environment)
    dag = [[] for _ in range(len(target_output))]
    for size in range(1, len(target_output) + 1):
        for start in range(len(target_output) - size + 1):
            substr = generate_substring(
                environment, target_output[start : start + size]
            )
            if substr:
                dynamic_programming_table[(start, start + size)] = substr
                dag[start].append(start + size)

    final_table = {}

    def build(begin, end):
        if (target_output[begin:end]) in final_table: return final_table[target_output[begin:end]]
        result = [ConstantString(target_output[begin:end])]
        for start in range(end - 1, begin - 1, -1):
            for dag_start in range(start, end):
                if dag[dag_start]:
                    for dag_end in dag[dag_start]:
                        if dag_end <= end:
                            mid = dynamic_programming_table[(dag_start, dag_end)]
                            entry = mid
                            if dag_end < end:
                                tail = build(dag_end, end)
                                entry = Concatenate(entry, tail)
                            if dag_start > begin:
                                head = ConstantString(target_output[begin:dag_start])
                                entry = Concatenate(head, entry)
                            result.append(entry)
        result = Union.make(result)
        final_table[target_output[begin:end]] = result
        return result

    result = build(0, len(target_output))
    return result


def test_flashfill_one_example(verbose=False):
    # collection of input-output specifications
    test_cases = []
    test_cases.append(({"input1": "test"}, "t"))
    test_cases.append(({"input1": "121"}, "1"))
    test_cases.append(({"input1": "xyz"}, "xyz"))
    test_cases.append(({"input1": "xyz"}, "Dr xyz"))
    test_cases.append(({"input1": "abcdefgh"}, "Dr abcdefgh"))
    test_cases.append(({"input1": "y"}, "abc"))
    test_cases.append(({"input1": "z"}, "abcdefgh"))
    test_cases.append(
        ({"input1": "June", "input2": "14", "input3": "1997"}, "1997, June 14")
    )
    test_cases.append(({"input1": "555-360-9792"}, "(555) 360-9792"))

    failure = False

    for input_output in test_cases:

        if verbose:
            print()

        start_time = time.time()
        version_space = flashfill_one_example(*input_output)

        if verbose:
            print("\tran synthesizer in time", time.time() - start_time, "seconds")

        version_space_size = version_space.version_space_size()
        if verbose:
            print("\tversion space contains", version_space_size, "programs")

        if version_space_size == 0:
            if verbose:
                print("Based on the input-outputs:")
                training_input, training_output = input_output
                print("\t", training_input, f" --> '{training_output}'")
                print(" [-] You constructed an empty version space.")
                print()
            failure = True
            continue

        tractable = version_space_size < 10000

        if tractable:
            if verbose:
                print(
                    "\tverifying everything in version space satisfies input-outputs."
                )
            expressions = version_space.extension()
        else:
            if verbose:
                print(
                    "\tverifying minimum cost member of version space satisfies input-outputs."
                )
            expressions = [version_space.minimum_cost_member_of_extension()]

        for expression in expressions:
            environment, target_output = input_output
            predicted_output = expression.evaluate(environment)
            if predicted_output != target_output:
                if verbose:
                    print("Based on the input-outputs:")
                    training_input, training_output = input_output
                    print("\t", training_input, f" --> '{training_output}'")
                    print(
                        "You constructed a version space containing the following program:"
                    )
                    print("\t", expression)
                    print("Which, when pretty printed, looks like:")
                    print("\t", expression.pretty_print())
                    print(
                        f' [-] This predicts the incorrect output: "{predicted_output}"'
                    )
                    print()
                failure = True

        if not failure:
            if verbose:
                print("\t[+] passed synthesis test case:")
                training_input, training_output = input_output
                print("\t\t", training_input, f" --> '{training_output}'")
                print("\t\twith the program:")
                to_print = (
                    version_space.minimum_cost_member_of_extension().pretty_print()
                )
                print("\t\t", to_print)
                print(
                    "\t\tequivalently:\n\t\t",
                    re.sub("Substring\\(input(.), 0, -1\\)", "input\\1", to_print),
                )

    if failure:
        total_points = 0
    else:
        total_points = 10
    print(f" [+] 4.3, flashfill_one_example, +{total_points}/10 points")

    return total_points


if __name__ == "__main__":
    test_flashfill_one_example(verbose=True)


def intersect(vs1, vs2, dynamic_programming_table=None):
    if dynamic_programming_table is None: dynamic_programming_table = {}

    dynamic_programming_key = (id(vs1), id(vs2))
    if dynamic_programming_key in dynamic_programming_table:
        return dynamic_programming_table[dynamic_programming_key]
    product = None
    if isinstance(vs1, Concatenate) and isinstance(vs2, Concatenate):
        left = intersect(vs1.left, vs2.left, dynamic_programming_table)
        right = intersect(vs1.right, vs2.right, dynamic_programming_table)
        if left and right:
            product = Concatenate(left, right)
    elif (
        isinstance(vs1, Substring)
        and isinstance(vs2, Substring)
        and vs1.the_string.name == vs2.the_string.name
    ):
        left = intersect(vs1.left, vs2.left, dynamic_programming_table)
        right = intersect(vs1.right, vs2.right, dynamic_programming_table)
        if left and right:
            product = Substring(vs1.the_string, left, right)
    elif isinstance(vs1, ConstantString) and isinstance(vs2, ConstantString):
        if vs1.content == vs2.content:
            product = vs1
    elif isinstance(vs1, Number) and isinstance(vs2, Number):
        if vs1.n == vs2.n:
            product = vs1
    elif isinstance(vs1, Union):
        members = [
            i
            for member in vs1.members
            if (i := intersect(member, vs2, dynamic_programming_table)) is not None
        ]
        if len(members) > 0:
            product = Union.make(members)
    elif isinstance(vs2, Union):
        members = [
            i
            for member in vs2.members
            if (i := intersect(vs1, member, dynamic_programming_table)) is not None
        ]
        if len(members) > 0:
            product = Union.make(members)
    dynamic_programming_table[id(vs1), id(vs2)] = product
    return product


def flashfill(input_outputs):
    """
    input_outputs: a list of input-output examples (each example is a tuple of environment and the target output)
    returns: the version space of all solutions to we synthesis problem

    You should call your implementation of `intersect` and also `flashfill_one_example`.
    """
    programs = [flashfill_one_example(tup[0], tup[1]) for tup in input_outputs]
    while len(programs) > 1:
        intersection = intersect(programs[0], programs[1])
        programs = programs[2:]
        if intersection is not None:
            programs.append(intersection)
    return programs[0]


def test_flashfill(verbose=False):
    # collection of input-output specifications
    test_cases = []
    test_cases.append([({"input1": "test"}, "t")])
    test_cases.append([({"input1": "121"}, "1")])
    test_cases.append([({"input1": "test"}, "t"), ({"input1": "121"}, "1")])
    test_cases.append([({"input1": "xyz"}, "xyz")])
    test_cases.append([({"input1": "xyz"}, "Dr xyz")])
    test_cases.append([({"input1": "abcdefgh"}, "Dr abcdefgh")])
    test_cases.append(
        [({"input1": "xyz"}, "Dr xyz"), ({"input1": "abcdefgh"}, "Dr abcdefgh")]
    )
    test_cases.append([({"input1": "y"}, "abc")])
    test_cases.append([({"input1": "z"}, "abcdefgh")])
    test_cases.append([({"input1": "y"}, "abcdefgh"), ({"input1": "z"}, "abcdefgh")])
    test_cases.append(
        [({"input1": "June", "input2": "14", "input3": "1997"}, "1997, June 14")]
    )
    test_cases.append(
        [
            ({"input1": "June", "input2": "14", "input3": "1997"}, "1997, June 14"),
            ({"input1": "October", "input2": "2", "input3": "2012"}, "2012, October 2"),
        ]
    )
    test_cases.append(
        [
            ({"input1": "555-360-9792"}, "(555) 360-9792"),
            ({"input1": "425-923-7777"}, "(425) 923-7777"),
        ]
    )

    failure = False

    for input_outputs in test_cases:

        if verbose:
            print()

        start_time = time.time()
        version_space = flashfill(input_outputs)

        if verbose:
            print("\tran synthesizer in time", time.time() - start_time, "seconds")

        version_space_size = version_space.version_space_size()
        if verbose:
            print("\tversion space contains", version_space_size, "programs")

        if version_space_size == 0:
            if verbose:
                print("Based on the input-outputs:")
                for training_input, training_output in input_outputs:
                    print("\t", training_input, f" --> '{training_output}'")
                print(" [-] You constructed an empty version space.")
                print()
            failure = True
            continue

        tractable = version_space_size < 10000

        if tractable:
            if verbose:
                print(
                    "\tverifying everything in version space satisfies input-outputs."
                )
            expressions = version_space.extension()
        else:
            if verbose:
                print(
                    "\tverifying minimum cost member of version space satisfies input-outputs."
                )
            expressions = [version_space.minimum_cost_member_of_extension()]

        for expression in expressions:
            for environment, target_output in input_outputs:
                predicted_output = expression.evaluate(environment)
                if predicted_output != target_output:
                    if verbose:
                        print("Based on the input-outputs:")
                        for training_input, training_output in input_outputs:
                            print("\t", training_input, f" --> '{training_output}'")
                        print(
                            "You constructed a version space containing the following program:"
                        )
                        print("\t", expression)
                        print("Which, when pretty printed, looks like:")
                        print("\t", expression.pretty_print())
                        print(
                            f' [-] This predicts the incorrect output: "{predicted_output}"'
                        )
                        print()
                    failure = True

        if not failure and verbose:
            print("\t[+] passed synthesis test case:")
            for training_input, training_output in input_outputs:
                print("\t\t", training_input, f" --> '{training_output}'")
            print("\t\twith the program:")
            to_print = version_space.minimum_cost_member_of_extension().pretty_print()
            print("\t\t", to_print)
            print(
                "\t\tequivalently:\n\t\t",
                re.sub("Substring\\(input(.), 0, -1\\)", "input\\1", to_print),
            )

    if failure:
        total_points = 0
    else:
        total_points = 10
    print(f" [+] 4.4, flashfill, +{total_points}/10 points")
    return total_points


if __name__ == "__main__":
    test_flashfill(verbose=False)


def test_4_6(verbose=False):
    # collection of input-output specifications
    test_cases = []
    test_cases.append(
        [({"input1": "June", "input2": "14", "input3": "1997"}, "1997, June 14")]
    )
    test_cases.append(
        [
            ({"input1": "June", "input2": "14", "input3": "1997"}, "1997, June 14"),
            ({"input1": "October", "input2": "2", "input3": "2012"}, "2012, October 2"),
        ]
    )

    target_programs = [
        'Substring(input3, 0, -1) + ", " + Substring(input1, 0, -1) + " " + Substring(input2, 0, -1)',
        'Substring(input3, 0, -1) + ", " + Substring(input1, 0, -1) + " " + Substring(input2, 0, -1)',
    ]

    failure = False

    for input_outputs, target_program in zip(test_cases, target_programs):

        if verbose:
            print()

        start_time = time.time()
        version_space = flashfill(input_outputs)

        if verbose:
            print("\tran synthesizer in time", time.time() - start_time, "seconds")

        version_space_size = version_space.version_space_size()
        if verbose:
            print("\tversion space contains", version_space_size, "programs")

        if version_space_size == 0:
            if verbose:
                print("Based on the input-outputs:")
                for training_input, training_output in input_outputs:
                    print("\t", training_input, f" --> '{training_output}'")
                print(" [-] You constructed an empty version space.")
            print()
            failure = True
            continue

        expression = version_space.minimum_cost_member_of_extension()
        if expression.pretty_print() != target_program:
            if verbose:
                print("Based on the input-outputs:")
                for training_input, training_output in input_outputs:
                    print("\t", training_input, f" --> '{training_output}'")
                print(
                    "You constructed a version space containing the following program:"
                )
                print("\t", expression)
                print("Which, when pretty printed, looks like:")
                print("\t", expression.pretty_print())
                print(f' [-] This is not the correct program: "{target_program}"')
                print()
            failure = True

    if failure:
        total_points = 0
    else:
        total_points = 5

    print(f" [+] 4.6, cost definitions, +{total_points}/5 points")
    return total_points


if __name__ == "__main__":
    test_4_6(verbose=True)
