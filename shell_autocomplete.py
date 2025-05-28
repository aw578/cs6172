from typing import List, Tuple, Dict
from dataclasses import dataclass
from shell_dsl import *


@dataclass
class CommandTriple:
    """Represents a (command, output, next_command) triple from shell history"""

    command: str
    output: str
    next_command: str


class ShellHistoryParser:
    """Parser for shell history"""

    @staticmethod
    def parse_history(history: str) -> List[CommandTriple]:
        """Parse shell history into CommandTriples"""
        triples = []
        blocks = history.split("$")

        for i in range(len(blocks) - 1):
            block = blocks[i].strip()
            next_block = blocks[i + 1].strip()

            if not block or not next_block:
                continue

            # Split block into command and output
            lines = block.split("\n")
            command = lines[0].strip()
            output = "\n".join(lines[1:]).strip()

            # Get next command from next block
            next_command = next_block.split("\n")[0].strip()

            triples.append(CommandTriple(command, output, next_command))

        return triples


def extract_literals(text: str) -> List[str]:
    """Extract space-separated literals from text"""
    return [word for word in text.split() if word.strip()]


def create_spec_from_triple(triple: CommandTriple) -> Tuple[Dict[str, str], str]:
    """
    Convert a CommandTriple into a specification for synthesis.
    Returns (input_dict, target_str) where:
    - input_dict contains the command and output
    - target_str is the next command to predict
    """
    input_dict = {"command": triple.command, "output": triple.output}
    return input_dict, triple.next_command
