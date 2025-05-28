import unittest
from shell_autocomplete import (
    ShellHistoryParser,
    CommandTriple,
    ContainsLiteral,
    ContainsOutput,
    IsEmpty,
    LiteralCommand,
    ConcatCommand,
    ExtractCommand,
    extract_literals,
    create_spec_from_triple,
)
from part1 import StringVariable, Number
from part4 import ConstantString


class TestShellHistoryParser(unittest.TestCase):
    def test_parse_history(self):
        history = """$ ls
file1.txt
file2.txt
$ cat file1.txt
Hello World
$ grep "Hello" file1.txt
Hello World
$ """

        triples = ShellHistoryParser.parse_history(history)

        self.assertEqual(len(triples), 3)

        # Test first triple
        self.assertEqual(triples[0].command, "ls")
        self.assertEqual(triples[0].output, "file1.txt\nfile2.txt")
        self.assertEqual(triples[0].next_command, "cat file1.txt")

        # Test second triple
        self.assertEqual(triples[1].command, "cat file1.txt")
        self.assertEqual(triples[1].output, "Hello World")
        self.assertEqual(triples[1].next_command, 'grep "Hello" file1.txt')


class TestDSLPredicates(unittest.TestCase):
    def test_contains_literal(self):
        pred = ContainsLiteral(StringVariable("command"), ConstantString("ls"))

        # Test positive case
        self.assertTrue(pred.evaluate({"command": "ls -l"}))

        # Test negative case
        self.assertFalse(pred.evaluate({"command": "cat file.txt"}))

    def test_contains_output(self):
        pred = ContainsOutput(ConstantString("error"))

        # Test positive case
        self.assertTrue(pred.evaluate({"output": "error: file not found"}))

        # Test negative case
        self.assertFalse(pred.evaluate({"output": "success"}))

    def test_is_empty(self):
        pred = IsEmpty(StringVariable("output"))

        # Test positive case
        self.assertTrue(pred.evaluate({"output": ""}))

        # Test negative case
        self.assertFalse(pred.evaluate({"output": "some text"}))


class TestDSLTemplates(unittest.TestCase):
    def test_literal_command(self):
        cmd = LiteralCommand("ls -l")
        self.assertEqual(cmd.evaluate({}), "ls -l")

    def test_concat_command(self):
        cmd = ConcatCommand(LiteralCommand("grep "), StringVariable("pattern"))
        self.assertEqual(cmd.evaluate({"pattern": "hello"}), "grep hello")

    def test_extract_command(self):
        cmd = ExtractCommand(StringVariable("command"), Number(0), Number(2))
        self.assertEqual(cmd.evaluate({"command": "ls -l"}), "ls")


class TestUtilityFunctions(unittest.TestCase):
    def test_extract_literals(self):
        text = "ls -l file.txt"
        literals = extract_literals(text)
        self.assertEqual(literals, ["ls", "-l", "file.txt"])

    def test_create_spec_from_triple(self):
        triple = CommandTriple(command="ls", output="file.txt", next_command="cat file.txt")
        input_dict, target = create_spec_from_triple(triple)

        self.assertEqual(input_dict["command"], "ls")
        self.assertEqual(input_dict["output"], "file.txt")
        self.assertEqual(target, "cat file.txt")


if __name__ == "__main__":
    unittest.main()
