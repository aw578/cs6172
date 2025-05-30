class Expression:

    def evaluate(self, environment):
        assert False, "not implemented"

    def arguments(self):
        assert False, "not implemented"

    def cost(self):
        return 1 + sum([0] + [argument.cost() for argument in self.arguments()])

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


class Plus(Expression):
    return_type = "int"
    argument_types = ["int", "int"]

    def __init__(self, x, y):
        self.x, self.y = x, y

    def __str__(self):
        return f"Plus({self.x}, {self.y})"

    def pretty_print(self):
        return f"(+ {self.x.pretty_print()} {self.y.pretty_print()})"

    def evaluate(self, environment):
        x = self.x.evaluate(environment)
        y = self.y.evaluate(environment)
        assert isinstance(x, int) and isinstance(y, int)
        return x + y


class Times(Expression):
    return_type = "int"
    argument_types = ["int", "int"]

    def __init__(self, x, y):
        self.x, self.y = x, y

    def __str__(self):
        return f"Times({self.x}, {self.y})"

    def pretty_print(self):
        return f"(* {self.x.pretty_print()} {self.y.pretty_print()})"

    def evaluate(self, environment):
        x = self.x.evaluate(environment)
        y = self.y.evaluate(environment)
        assert isinstance(x, int) and isinstance(y, int)
        return x * y


class LessThan(Expression):
    return_type = "bool"
    argument_types = ["int", "int"]

    def __init__(self, x, y):
        self.x, self.y = x, y

    def __str__(self):
        return f"LessThan({self.x}, {self.y})"

    def pretty_print(self):
        return f"(< {self.x.pretty_print()} {self.y.pretty_print()})"

    def evaluate(self, environment):
        x = self.x.evaluate(environment)
        y = self.y.evaluate(environment)
        assert isinstance(x, int) and isinstance(y, int)
        return x < y


class And(Expression):
    return_type = "bool"
    argument_types = ["bool", "bool"]

    def __init__(self, x, y):
        self.x, self.y = x, y

    def __str__(self):
        return f"And({self.x}, {self.y})"

    def pretty_print(self):
        return f"(and {self.x.pretty_print()} {self.y.pretty_print()})"

    def evaluate(self, environment):
        x = self.x.evaluate(environment)
        y = self.y.evaluate(environment)
        assert isinstance(x, bool) and isinstance(y, bool)
        return x and y


class Not(Expression):
    return_type = "bool"
    argument_types = ["bool"]

    def __init__(self, x):
        self.x = x

    def __str__(self):
        return f"Not({self.x})"

    def pretty_print(self):
        return f"(not {self.x.pretty_print()})"

    def evaluate(self, environment):
        x = self.x.evaluate(environment)
        assert isinstance(x, bool)
        return not x


class If(Expression):
    return_type = "int"
    argument_types = ["bool", "int", "int"]

    def __init__(self, test, yes, no):
        self.test, self.yes, self.no = test, yes, no

    def __str__(self):
        return f"If({self.test}, {self.yes}, {self.no})"

    def pretty_print(self):
        return f"(if {self.test.pretty_print()} {self.yes.pretty_print()} {self.no.pretty_print()})"

    def evaluate(self, environment):
        test = self.test.evaluate(environment)
        yes = self.yes.evaluate(environment)
        no = self.no.evaluate(environment)
        assert isinstance(test, bool) and isinstance(yes, int) and isinstance(no, int)
        if test:
            return yes
        else:
            return no


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
