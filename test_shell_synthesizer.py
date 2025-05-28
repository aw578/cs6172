import unittest
from shell_synthesizer import ShellSynthesizer, SynthesisExample


class TestShellSynthesizer(unittest.TestCase):
    def setUp(self):
        self.synthesizer = ShellSynthesizer(max_depth=10)

    def test_synthesis_simple(self):
        """Test synthesis on simple.log which has a clear pattern: if output contains 'error', run 'cat error.log'"""
        program = self.synthesizer.synthesize_from_log("test/simple.log")
        self.assertIsNotNone(program)

        # Test the synthesized program
        examples = self.synthesizer._parse_log_file("test/simple.log")
        accuracy = self.synthesizer.validate(program, examples)
        self.assertEqual(accuracy, 1.0)

    def test_synthesis_sample(self):
        """Test synthesis on sample.log which has more complex patterns"""
        program = self.synthesizer.synthesize_from_log("test/sample.log")
        self.assertIsNotNone(program)

        # Test the synthesized program
        examples = self.synthesizer._parse_log_file("test/sample.log")
        accuracy = self.synthesizer.validate(program, examples)
        self.assertGreater(accuracy, 0.5)  # Should at least get some patterns right

    def test_synthesis_assignment(self):
        """Test synthesis on assignment.log which has the most complex patterns"""
        program = self.synthesizer.synthesize_from_log("test/assignment.log")
        self.assertIsNotNone(program)

        # Test the synthesized program
        examples = self.synthesizer._parse_log_file("test/assignment.log")
        accuracy = self.synthesizer.validate(program, examples)
        self.assertGreater(accuracy, 0.5)  # Should at least get some patterns right

    def test_parse_log_file(self):
        """Test that log files are parsed correctly"""
        examples = self.synthesizer._parse_log_file("test/simple.log")

        # Check that we have the right number of examples
        self.assertEqual(len(examples), 5)  # simple.log has 5 command sequences

        # Check first example
        self.assertEqual(examples[0].input_dict["command"], "ls")
        self.assertEqual(examples[0].input_dict["output"], "error: file not found")
        self.assertEqual(examples[0].target, "cat error.log")

        # Check second example
        self.assertEqual(examples[1].input_dict["command"], "cat error.log")
        self.assertEqual(examples[1].input_dict["output"], "aed")
        self.assertEqual(examples[1].target, "ls")


if __name__ == "__main__":
    unittest.main()
