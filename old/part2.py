from part1 import *

import itertools
import time


def bottom_up(global_bound, operators, constants, input_outputs):
    """
    global_bound: int. an upper bound on the size of expression
    operators: list of classes, such as [Times, If, ...]
    constants: list of possible leaves in syntax tree, such as [Number(1)]. Variables can also be leaves, but these are automatically inferred from `input_outputs`
    input_outputs: list of tuples of environment (the input) and desired output, such as [({'x': 5}, 6), ({'x': 1}, 2)]
    returns: either None if no program can be found that satisfies the input outputs, or the smallest such program. If a program `p` is returned, it should satisfy `all( p.evaluate(input) == output for input, output in input_outputs )`
    """

    target_outputs = tuple(y for x, y in input_outputs)
    for expression in bottom_up_generator(
        global_bound, operators, constants, input_outputs
    ):
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
        from part4 import StringVariable

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

    # return whether it's new
    def evaluate(expr, size):
        value = tuple([expr.evaluate(input) for input in inputs])
        if value not in values:
            if (expr.return_type, size) not in map:
                map[(expr.return_type, size)] = []
            map[(expr.return_type, size)].append(expr)
            values[value] = True
            return True
        return False

    for item in variables_and_constants:
        if evaluate(item, 1):
            yield item

    for size in range(1, global_bound + 1):
        # print(f"""==========size = {size}==========""")
        # print(map)
        # print(values)
        for operator in operators:
            partitions = integer_partitions(size - 1, len(operator.argument_types))
            for partition in partitions:
                args = [
                    map.get((operator.argument_types[sub_idx], partition[sub_idx]), [])
                    for sub_idx in range(len(partition))
                ]
                # print(f"""operator = {operator}""")
                # print(f"""args for partition {partition}: {args}""")
                for combination in itertools.product(*args):
                    if evaluate(operator(*combination), size):
                        yield operator(*combination)


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

    return [
        [x1] + x2s
        for x1 in range(target_value + 1)
        for x2s in integer_partitions(target_value - x1, number_of_arguments - 1)
    ]


def test_bottom_up(verbose=False):
    operators = [Plus, Times, LessThan, And, Not, If]
    terminals = [FALSE(), Number(0), Number(1), Number(-1)]

    # collection of input-output specifications
    test_cases = []
    test_cases.append([({"x": 1}, 1), ({"x": 4}, 16), ({"x": 5}, 25)])
    test_cases.append(
        [
            ({"x": 1, "y": 2}, 1),
            ({"x": 5, "y": 2}, 2),
            ({"x": 99, "y": 98}, 98),
            ({"x": 97, "y": 98}, 97),
        ]
    )
    test_cases.append(
        [
            ({"x": 10, "y": 7}, 17),
            ({"x": 4, "y": 7}, -7),
            ({"x": 10, "y": 3}, 13),
            ({"x": 1, "y": -7}, -6),
            ({"x": 1, "y": 8}, -8),
        ]
    )

    how_many_points = [
        7,
        7,
        7,
    ]  # the last test case is harder, so it is worth more points

    # the optimal size of each program that solves the corresponding test case
    optimal_sizes = [3, 6, 10]

    total_points = 0
    for test_case, optimal_size, pt in zip(test_cases, optimal_sizes, how_many_points):
        if bottom_up(optimal_size - 1, operators, terminals, test_case) is not None:
            if verbose:
                print(
                    f"you should not be able to solve this test case w/ a program whose syntax tree is of size {optimal_size-1}. the specific test case is {test_case}"
                )
            continue

        start_time = time.time()
        expression = bottom_up(optimal_size, operators, terminals, test_case)
        if expression is None:
            if verbose:
                print(
                    f"failed to synthesize a program when the size bound was {optimal_size}. the specific test case is {test_case}"
                )
            continue

        if verbose:
            print(
                f"synthesized program:\t {expression.pretty_print()} in {time.time() - start_time} seconds"
            )
        fails_testcase = False
        for xs, y in test_case:
            if expression.evaluate(xs) != y:
                if verbose:
                    print(
                        f"synthesized program {expression.pretty_print()} does not satisfy the following test case: {xs} --> {y}"
                    )
                fails_testcase = True
            else:
                if verbose:
                    print(f"passes test case {xs} --> {y}")

        if verbose:
            print()

        if fails_testcase:
            continue

        total_points += pt

    print(f" [+] 2, bottom-up synthesis: +{total_points}/21 points")

    return total_points


if __name__ == "__main__":
    test_bottom_up(verbose=True)
