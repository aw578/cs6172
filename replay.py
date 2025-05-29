import re, sys


class TerminalEmulator:
    def __init__(self, width=80, height=24):
        self.width, self.height = width, height
        self.buffer = [[" "] * width for _ in range(height)]
        self.cx = self.cy = 0

    def process(self, data):
        i = 0
        while i < len(data):
            ch = data[i]
            if ch == "\x1b":
                # OSC: ESC ] … BEL
                if i + 1 < len(data) and data[i + 1] == "]":
                    i += 2
                    while i < len(data) and data[i] != "\x07":
                        i += 1
                    i += 1
                    continue

                # CSI: ESC [ args cmd
                m = re.match(r"\x1b\[(?P<args>[\d;]*)(?P<cmd>[A-Za-z@])", data[i:])
                if m:
                    args = [int(x) for x in m.group("args").split(";") if x] or []
                    cmd = m.group("cmd")
                    self.handle_csi(cmd, args)
                    i += m.end()
                    continue

                i += 1

            elif ch == "\b":
                if self.cx > 0:
                    self.cx -= 1
                i += 1

            elif ch == "\r":
                self.cx = 0
                i += 1

            elif ch == "\n":
                self.newline()
                i += 1

            elif ch == "\t":
                spaces = 8 - (self.cx % 8)
                for _ in range(spaces):
                    self.put_char(" ")
                i += 1

            else:
                self.put_char(ch)
                i += 1

    def handle_csi(self, cmd, nums):
        n = nums[0] if nums else 1

        if cmd in ("H", "f"):  # cursor position
            r = (nums[0] - 1) if len(nums) >= 1 else 0
            c = (nums[1] - 1) if len(nums) >= 2 else 0
            self.cy, self.cx = max(0, min(self.height - 1, r)), max(0, min(self.width - 1, c))

        elif cmd == "A":  # up
            self.cy = max(0, self.cy - n)
        elif cmd == "B":  # down
            self.cy = min(self.height - 1, self.cy + n)
        elif cmd == "C":  # forward
            self.cx = min(self.width - 1, self.cx + n)
        elif cmd == "D":  # back
            self.cx = max(0, self.cx - n)

        elif cmd == "J":  # erase display
            if nums and nums[0] == 2:
                self.buffer = [[" "] * self.width for _ in range(self.height)]
                self.cx = self.cy = 0

        elif cmd == "K":  # erase line
            for x in range(self.cx, self.width):
                self.buffer[self.cy][x] = " "

        elif cmd == "P":  # delete chars
            line = self.buffer[self.cy]
            del line[self.cx : self.cx + n]
            line.extend([" "] * n)

        elif cmd == "@":  # insert blanks
            line = self.buffer[self.cy]
            for _ in range(n):
                line.insert(self.cx, " ")
                line.pop()

        # ignore mode set/reset and others:
        elif cmd in ("h", "l"):
            pass

        # anything else falls through silently

    def put_char(self, ch):
        if " " <= ch <= "~":
            self.buffer[self.cy][self.cx] = ch
            self.cx += 1
            if self.cx >= self.width:
                self.newline()

    def newline(self):
        self.cx = 0
        self.cy += 1
        if self.cy >= self.height:
            self.buffer.pop(0)
            self.buffer.append([" "] * self.width)
            self.cy = self.height - 1

    def render(self):
        return "\n".join("".join(row).rstrip() for row in self.buffer)


def main():
    if len(sys.argv) < 2:
        print("Usage: python replay.py <logfile> [width height]")
        sys.exit(1)

    logfile = sys.argv[1]
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 80
    h = int(sys.argv[3]) if len(sys.argv) > 3 else 24

    data = open(logfile, "r", errors="ignore").read()
    term = TerminalEmulator(width=w, height=h)
    term.process(data)
    print(term.render())


if __name__ == "__main__":
    main()
