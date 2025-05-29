import itertools
import math


class Expression:

    def evaluate(self, environment):
        assert False, "not implemented"

    def arguments(self):
        assert False, "not implemented"

    def cost(self):
        base_cost = 1 + sum([0] + [argument.cost() for argument in self.arguments()])
        # Apply cost reduction for qualifying number expressions
        if self.return_type == "int" and self._qualifies_for_cost_reduction():
            return max(0, base_cost - 1)
        return base_cost

    def priority(self):
        """Calculate priority for tiebreaking. Higher priority means better (evaluated first)."""
        if self.return_type == "int" and self._qualifies_for_cost_reduction():
            return 1
        return 0

    def _qualifies_for_cost_reduction(self):
        """Check if this number expression qualifies for cost reduction."""
        if self.return_type != "int":
            return False

        # Get all leaf nodes
        leaves = self._get_leaf_nodes()

        # Count different types of leaves
        substring_sources = set()  # Track different substring sources
        has_number_literal = False

        for leaf in leaves:
            if isinstance(leaf, ToInt):
                # Check if it's ToInt(Substring(...))
                if isinstance(leaf.value, Substring):
                    substring = leaf.value
                    # Create a unique identifier for this substring source
                    source_id = (str(substring.the_string), str(substring.left), str(substring.right))
                    substring_sources.add(source_id)
            elif isinstance(leaf, Number) and leaf.n not in [-1, 0]:
                has_number_literal = True

        # Qualifies if:
        # 1. Multiple different substring sources, OR
        # 2. At least one substring source AND at least one number literal
        return len(substring_sources) > 1 or (len(substring_sources) > 0 and has_number_literal)

    def _get_leaf_nodes(self):
        """Get all leaf nodes (nodes with no arguments) in this expression tree."""
        if not self.arguments():
            return [self]

        leaves = []
        for arg in self.arguments():
            leaves.extend(arg._get_leaf_nodes())
        return leaves

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        return hash(str(self))

    def __ne__(self, other):
        return str(self) != str(other)

    def __gt__(self, other):
        return str(self) > str(other)

    def __lt__(self, other):
        return str(self) < str(other)

    def minimum_cost_member_of_extension(self):
        assert False, "implement as part of homework"

    def version_space_size(self):
        assert False, "implement as part of homework"


class FALSE(Expression):
    return_type = "bool"
    argument_types = []

    def __init__(self):
        pass

    def __str__(self):
        return "False"

    def pretty_print(self):
        return "False"

    def evaluate(self, environment):
        return False

    def arguments(self):
        return []


class Number(Expression):
    return_type = "int"
    argument_types = []

    def __init__(self, n):
        self.n = n

    def __str__(self):
        return f"Number({self.n})"

    def cost(self):
        if self.n == -1 or self.n == 0:
            return 0
        return 1

    def extension(self):
        return [self]

    def version_space_size(self):
        return 1

    def minimum_cost_member_of_extension(self):
        return self

    def pretty_print(self):
        return str(self.n)

    def evaluate(self, environment):
        return self.n

    def arguments(self):
        return []


class Plus(Expression):
    return_type = "int"
    argument_types = ["int", "int"]

    def __init__(self, x, y):
        self.x, self.y = x, y

    def __str__(self):
        return f"Plus({self.x}, {self.y})"

    def pretty_print(self):
        return f"({self.x.pretty_print()} + {self.y.pretty_print()})"

    def evaluate(self, environment):
        x = self.x.evaluate(environment)
        y = self.y.evaluate(environment)
        assert isinstance(x, int) and isinstance(y, int)
        return x + y

    def arguments(self):
        return [self.x, self.y]


class Minus(Expression):
    return_type = "int"
    argument_types = ["int", "int"]

    def __init__(self, x, y):
        self.x, self.y = x, y

    def __str__(self):
        return f"Minus({self.x}, {self.y})"

    def pretty_print(self):
        return f"({self.x.pretty_print()} - {self.y.pretty_print()})"

    def evaluate(self, environment):
        x = self.x.evaluate(environment)
        y = self.y.evaluate(environment)
        assert isinstance(x, int) and isinstance(y, int)
        return x - y

    def arguments(self):
        return [self.x, self.y]


# class Times(Expression):
#     return_type = "int"
#     argument_types = ["int", "int"]

#     def __init__(self, x, y):
#         self.x, self.y = x, y

#     def __str__(self):
#         return f"Times({self.x}, {self.y})"

#     def pretty_print(self):
#         return f"({self.x.pretty_print()} * {self.y.pretty_print()})"

#     def evaluate(self, environment):
#         x = self.x.evaluate(environment)
#         y = self.y.evaluate(environment)
#         assert isinstance(x, int) and isinstance(y, int)
#         return x * y


class LessThan(Expression):
    return_type = "bool"
    argument_types = ["int", "int"]

    def __init__(self, x, y):
        self.x, self.y = x, y

    def __str__(self):
        return f"LessThan({self.x}, {self.y})"

    def pretty_print(self):
        return f"({self.x.pretty_print()} < {self.y.pretty_print()})"

    def evaluate(self, environment):
        x = self.x.evaluate(environment)
        y = self.y.evaluate(environment)
        assert isinstance(x, int) and isinstance(y, int)
        return x < y

    def arguments(self):
        return [self.x, self.y]


class And(Expression):
    return_type = "bool"
    argument_types = ["bool", "bool"]

    def __init__(self, x, y):
        self.x, self.y = x, y

    def __str__(self):
        return f"And({self.x}, {self.y})"

    def pretty_print(self):
        return f"({self.x.pretty_print()} && {self.y.pretty_print()})"

    def evaluate(self, environment):
        x = self.x.evaluate(environment)
        y = self.y.evaluate(environment)
        assert isinstance(x, bool) and isinstance(y, bool)
        return x and y

    def arguments(self):
        return [self.x, self.y]


class Not(Expression):
    return_type = "bool"
    argument_types = ["bool"]

    def __init__(self, x):
        self.x = x

    def __str__(self):
        return f"Not({self.x})"

    def pretty_print(self):
        return f"(! {self.x.pretty_print()})"

    def evaluate(self, environment):
        x = self.x.evaluate(environment)
        assert isinstance(x, bool)
        return not x

    def arguments(self):
        return [self.x]


class Or(Expression):
    return_type = "bool"
    argument_types = ["bool", "bool"]

    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __str__(self):
        return f"Or({self.left}, {self.right})"

    def pretty_print(self):
        return f"({self.left.pretty_print()} || {self.right.pretty_print()})"

    def evaluate(self, environment):
        left_val = self.left.evaluate(environment)
        right_val = self.right.evaluate(environment)
        assert isinstance(left_val, bool) and isinstance(right_val, bool)
        return left_val or right_val

    def arguments(self):
        return [self.left, self.right]


class StringEquals(Expression):
    """Checks if two strings are equal"""

    return_type = "bool"
    argument_types = ["str", "str"]

    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __str__(self):
        return f"StringEquals({self.left}, {self.right})"

    def pretty_print(self):
        return f"({self.left.pretty_print()} == {self.right.pretty_print()})"

    def evaluate(self, environment):
        left_val = self.left.evaluate(environment)
        right_val = self.right.evaluate(environment)
        return left_val == right_val

    def arguments(self):
        return [self.left, self.right]

    def __hash__(self):
        return hash(("StringEquals", self.left, self.right))


class If(Expression):
    return_type = "str"
    argument_types = ["bool", "str", "str"]

    def __init__(self, test, yes, no):
        self.test, self.yes, self.no = test, yes, no

    def __str__(self):
        return f"If({self.test}, {self.yes}, {self.no})"

    def pretty_print(self):
        return f"(if {self.test.pretty_print()} then {self.yes.pretty_print()} else {self.no.pretty_print()})"

    def evaluate(self, environment):
        test = self.test.evaluate(environment)
        yes = self.yes.evaluate(environment)
        no = self.no.evaluate(environment)
        assert isinstance(test, bool) and isinstance(yes, str) and isinstance(no, str)
        if test:
            return yes
        else:
            return no

    def arguments(self):
        return [self.test, self.yes, self.no]


class Concatenate(Expression):
    return_type = "str"
    argument_types = ["str", "str"]

    def __init__(self, left, right):
        self.left, self.right = left, right

    def __str__(self):
        return f"Concatenate({self.left}, {self.right})"

    def is_constant_string(self):
        return (isinstance(self.left, ConstantString) or (isinstance(self.left, Concatenate) and self.left.is_constant_string())) and (
            isinstance(self.right, ConstantString) or (isinstance(self.right, Concatenate) and self.right.is_constant_string())
        )

    def pretty_print(self):
        # if the left and right are both constant strings, then we can just remove the quotes and concatenate them
        if self.is_constant_string():
            if isinstance(self.left, ConstantString):
                left_str = self.left.pretty_print()[1:-1]
            else:
                left_str = self.left.pretty_print()
            if isinstance(self.right, ConstantString):
                right_str = self.right.pretty_print()[1:-1]
            else:
                right_str = self.right.pretty_print()
            return f"{left_str}{right_str}"
        else:
            return f"({self.left.pretty_print()} + {self.right.pretty_print()})"

    def extension(self):
        return [Concatenate(left, right) for left in self.left.extension() for right in self.right.extension()]

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

    def arguments(self):
        return [self.test, self.yes, self.no]


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
        return 2

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
        return self.the_string.version_space_size() * self.left.version_space_size() * self.right.version_space_size()

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


class NumberVariable(Expression):
    return_type = "int"
    argument_types = []

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"NumberVariable('{self.name}')"

    def pretty_print(self):
        return self.name

    def evaluate(self, environment):
        return environment[self.name]

    def arguments(self):
        return []


class ToString(Expression):
    return_type = "str"
    argument_types = ["int"]

    def __init__(self, value):
        self.value = value

    def evaluate(self, environment):
        return str(self.value.evaluate(environment))

    def pretty_print(self):
        return f"ToString({self.value.pretty_print()})"

    def __str__(self):
        return f"ToString({self.value})"

    def __repr__(self):
        return f"ToString({repr(self.value)})"

    def arguments(self):
        return [self.value]


class ToInt(Expression):
    return_type = "int"
    argument_types = ["str"]

    def __init__(self, value):
        self.value = value

    def evaluate(self, environment):
        return int(self.value.evaluate(environment))

    def pretty_print(self):
        return f"ToInt({self.value.pretty_print()})"

    def __str__(self):
        return f"ToInt({self.value})"

    def arguments(self):
        return [self.value]


class Contains(Expression):
    """Checks if a string contains a value"""

    return_type = "bool"
    argument_types = ["str", "str"]

    def __init__(self, haystack: Expression, needle: Expression):
        self.haystack = haystack
        self.needle = needle

    def evaluate(self, environment: dict[str, str]) -> bool:
        haystack_val = self.haystack.evaluate(environment)
        needle_val = self.needle.evaluate(environment)
        return needle_val in haystack_val

    def pretty_print(self) -> str:
        return f"ContainsLiteral({self.haystack.pretty_print()}, {self.needle.pretty_print()})"

    def arguments(self):
        return [self.haystack, self.needle]

    def __str__(self):
        return f"ContainsLiteral({self.haystack}, {self.needle})"

    def __hash__(self):
        return hash(("ContainsLiteral", self.haystack, self.needle))


class IsEmpty(Expression):
    """Checks if a string is empty"""

    return_type = "bool"
    argument_types = ["str"]

    def __init__(self, value: Expression):
        self.value = value

    def evaluate(self, environment: dict[str, str]) -> bool:
        val = self.value.evaluate(environment)
        return len(val) == 0

    def pretty_print(self) -> str:
        return f"IsEmpty({self.value.pretty_print()})"

    def arguments(self):
        return [self.value]

    def __str__(self):
        return f"IsEmpty({self.value})"

    def __hash__(self):
        return hash(("IsEmpty", self.value))


def bottom_up(global_bound, operators, constants, input_outputs):
    """
    global_bound: int. an upper bound on the size of expression
    operators: list of classes, such as [Times, If, ...]
    constants: list of possible leaves in syntax tree, such as [Number(1)]. Variables can also be leaves, but these are automatically inferred from `input_outputs`
    input_outputs: list of tuples of environment (the input) and desired output, such as [({'x': 5}, 6), ({'x': 1}, 2)]
    returns: either None if no program can be found that satisfies the input outputs, or the smallest such program. If a program `p` is returned, it should satisfy `all( p.evaluate(input) == output for input, output in input_outputs )`
    """

    target_outputs = tuple(y for x, y in input_outputs)
    for expression in bottom_up_generator(global_bound, operators, constants, input_outputs):
        outputs = tuple(expression.evaluate(input) for input, output in input_outputs)

        if outputs == target_outputs:
            return expression

    return None


def bottom_up_generator(global_bound, operators, constants, input_outputs):
    """
    global_bound: int. an upper bound on the size of expression
    operators: list of classes, such as [Times, If, ...]
    constants: list of possible leaves in syntax tree, such as [Number(1)]. Variables can also be leaves, but these are automatically inferred from `input_outputs`
    input_outputs: list of tuples of environment (the input) and desired output, such as [({'x': 5}, 6), ({'x': 1}, 2)]
    yields: sequence of programs, ordered by expression size, which are semantically distinct on the input examples
    """

    # suggested first thing: variables and constants should be treated the same, because they are both leaves in syntax trees
    # after computing `variables_and_constants`, you should no longer refer to `constants`. express everything in terms of `variables_and_constants`
    # `make_variable` is just a helper function for making variables that smartly wraps the variable name in the correct class depending on the type of the variable
    def make_variable(variable_name, variable_value):
        if isinstance(variable_value, int):
            return NumberVariable(variable_name)
        if isinstance(variable_value, str):
            return StringVariable(variable_name)
        assert False, "only numbers and strings are supported as variable inputs"

    inputs, correct_outputs = zip(*input_outputs)
    variables = list(
        {
            make_variable(variable_name, variable_value)
            for inputs, outputs in input_outputs
            for variable_name, variable_value in inputs.items()
        }
    )
    variables_and_constants = constants + variables

    # suggested data structure (you don't have to use this if you don't want):
    # a mapping from a tuple of (type, expression_size) to all of the possible values that can be computed of that type using an expression of that size
    map = {}

    # (outputs)
    values = {}

    # Track expressions by size and priority for proper ordering
    expressions_by_size = {}

    # return whether it's new
    def evaluate(expr, size):
        value = tuple([expr.evaluate(input) for input in inputs])
        if value not in values:
            if (expr.return_type, size) not in map:
                map[(expr.return_type, size)] = []
            map[(expr.return_type, size)].append(expr)
            values[value] = True

            # Store expression by size for priority-based ordering
            if size not in expressions_by_size:
                expressions_by_size[size] = []
            expressions_by_size[size].append(expr)

            return True
        return False

    # Process size 1 expressions first
    size_1_expressions = []
    for item in variables_and_constants:
        if evaluate(item, 1):
            size_1_expressions.append(item)

    # Sort size 1 expressions by priority and yield them
    size_1_expressions.sort(key=lambda x: (-x.priority(), str(x)))
    for expr in size_1_expressions:
        yield expr

    for size in range(2, global_bound + 1):
        size_expressions = []

        for operator in operators:
            partitions = integer_partitions(size - 1, len(operator.argument_types))
            for partition in partitions:
                args = [map.get((operator.argument_types[sub_idx], partition[sub_idx]), []) for sub_idx in range(len(partition))]
                for combination in itertools.product(*args):
                    expr = operator(*combination)
                    if evaluate(expr, size):
                        size_expressions.append(expr)

        # Sort expressions of this size by priority (higher priority first), then lexicographically
        size_expressions.sort(key=lambda x: -x.priority())
        for expr in size_expressions:
            yield expr


def integer_partitions(target_value, number_of_arguments):
    """
    Returns all ways of summing up to `target_value` by adding `number_of_arguments` nonnegative integers
    You may find this useful when implementing `bottom_up_generator`:

    Imagine that you are trying to enumerate all expressions of size 10, and you are considering using an operator with 3 arguments.
    So the total size has to be 10, which includes +1 from this operator, as well as 3 other terms from the 3 arguments, which together have to sum to 10.
    Therefore: 10 = 1 + size_of_first_argument + size_of_second_argument + size_of_third_argument
    Also, every argument has to be of size at least one, because you can't have a syntax tree of size 0
    Therefore: 10 = 1 + (1 + size_of_first_argument_minus_one) + (1 + size_of_second_argument_minus_one) + (1 + size_of_third_argument_minus_one)
    So, by algebra:
         10 - 1 - 3 = size_of_first_argument_minus_one + size_of_second_argument_minus_one + size_of_third_argument_minus_one
    where: size_of_first_argument_minus_one >= 0
           size_of_second_argument_minus_one >= 0
           size_of_third_argument_minus_one >= 0
    Therefore: the set of allowed assignments to {size_of_first_argument_minus_one,size_of_second_argument_minus_one,size_of_third_argument_minus_one} is just the integer partitions of (10 - 1 - 3).
    """

    if target_value < 0:
        return []

    if number_of_arguments == 1:
        return [[target_value]]

    return [[x1] + x2s for x1 in range(target_value + 1) for x2s in integer_partitions(target_value - x1, number_of_arguments - 1)]


def dcsolve(operators, constants, input_outputs):
    """
    operators: list of classes, such as [Times, Not, ...]. Note that `If` does not have to be here, because the decision tree learner inserts such expressions
    constants: list of possible leaves in syntax tree, such as [Number(1)]. Variables can also be leaves, but these are automatically inferred from `input_outputs`
    input_outputs: list of tuples of environment (the input) and desired output, such as [({'x': 5}, 6), ({'x': 1}, 2)]
    returns: an expression `p` which should satisfy `all( p.evaluate(input) == output for input, output in input_outputs )`
    """
    # generate terms (non-bools) and predicates (bools)

    # generate terms until at least one term for each point
    term_grammar = [operator for operator in operators if operator.return_type != "bool"]
    points = set([0])
    while True:
        actual_points = [input_outputs[i] for i in points]
        cover = {}
        terms = []

        cover_union = [False for _ in points]
        covered = 0

        term_generator = bottom_up_generator(30, term_grammar, constants, actual_points)
        pred_generator = bottom_up_generator(30, operators, constants, actual_points)
        while covered < len(points):
            term = next(term_generator)
            print(f"Term: {term}")
            terms.append(term)
            term_cover = tuple([term.evaluate(io[0]) == io[1] for io in actual_points])
            cover[term] = term_cover
            for i in range(len(points)):
                if cover_union[i] == False and term_cover[i] == True:
                    covered += 1
                    cover_union[i] = True
        preds = set()
        dt = None
        while dt is None:
            term = next(term_generator)
            terms.append(term)
            term_cover = tuple([term.evaluate(io[0]) == io[1] for io in actual_points])
            cover[term] = term_cover

            pred = next(pred_generator)
            preds.add((pred, tuple([pred.evaluate(io[0]) for io in actual_points])))
            # decision tree wants a set of remaining point indices
            dt = learn_decision_tree(cover, terms, preds, set([i for i in range(len(points))]))
        satisfied = True
        for i in range(len(input_outputs)):
            if dt.evaluate(input_outputs[i][0]) != input_outputs[i][1]:
                satisfied = False
                points.add(i)
        if satisfied:
            return dt


def learn_decision_tree(cover, terms, predicates, examples_we_care_about):
    """
    You may find this utility function helpful
    cover: dictionary mapping from expression to tuple of bool's. `cover[e][i] == True` iff expression `e` predicts the correct output for `i`th input
    terms: set of expressions that we can use as leaves in the decision tree
    predicates: predicates we can use as branches in the decision tree. each element of `predicates` should be a tuple of `(expression, outputs)` where `outputs` is a tuple of bool's. Should satisfy `outputs[i] == expression.evaluate(input_outputs[i][0])`
    examples_we_care_about: a set of integers, telling which input outputs we care about solving for. For example if we are done, then this will be the empty set. If we are just starting out synthesizing the decision tree, then this will be the numbers 0-(len(input_outputs)-1)
    """

    for expression in terms:
        if all(cover[expression][i] for i in examples_we_care_about):
            return expression

    if len(predicates) == 0:
        return None  # no more predicates to split on

    def information_gain(predicate_info):
        """actually returns -information gain up to a constant ($G$ in paper)"""
        predicate, predicate_outputs = predicate_info

        examples_yes = {i for i in examples_we_care_about if predicate_outputs[i]}
        examples_no = {i for i in examples_we_care_about if not predicate_outputs[i]}

        probability_yes = len(examples_yes) / len(examples_we_care_about)
        probability_no = len(examples_no) / len(examples_we_care_about)

        entropy_yes = entropy(examples_yes)
        entropy_no = entropy(examples_no)

        return probability_yes * entropy_yes + probability_no * entropy_no

    def entropy(example_indices):
        # entries proportional probability that the term used during evaluation is a specific term
        # len of `distribution` will be the number of terms
        distribution = []

        for expression in terms:
            # calculate probability that we used this expression, assuming uniform distribution over which example is being run
            ps = []
            for example_index in example_indices:
                if not cover[expression][example_index]:  # we can't explain this example, definitely are not a candidate term
                    p = 0
                else:
                    p = sum(cover[expression][i] for i in example_indices)
                    p /= sum(
                        cover[other_expression][i]
                        for other_expression in terms
                        if cover[other_expression][example_index]
                        for i in example_indices
                    )
                ps.append(p)

            distribution.append(sum(ps))

        # original paper has 1/|pts| term, but we can absorb this into normalizing constant
        z = sum(distribution)  # normalizing constant

        return -sum(p / z * math.log(p / z) for p in distribution if p > 0)

    predicate, predicate_outputs = min(predicates, key=information_gain)

    left_hand_side_examples = {i for i in examples_we_care_about if predicate_outputs[i]}
    right_hand_side_examples = {i for i in examples_we_care_about if not predicate_outputs[i]}

    predicates = predicates - {(predicate, predicate_outputs)}

    lhs = learn_decision_tree(cover, terms, predicates, left_hand_side_examples)
    if lhs is None:
        return None

    rhs = learn_decision_tree(cover, terms, predicates, right_hand_side_examples)
    if rhs is None:
        return None

    return If(predicate, lhs, rhs)
