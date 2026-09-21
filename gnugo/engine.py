# GNU Go engine wrapper for PettingZoo go_v5.
#
# Wraps a GTP GNU Go process and translates between the two systems' move
# formats, so GNU Go can play inside a go_v5 game - go_v5 numbers every board
# point, while GNU Go's GTP protocol names each point with a column letter
# and a row number counted from the opposite edge. action_to_vertex() and
# vertex_to_action() below do that conversion both ways.


# the low-level GTP process wrapper (GTPEngine) and its error type
from gnugo.gtp import GTPEngine, GTPError

# GTP column letters, left to right — skips 'I' (GTP's own convention)
LETTERS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"


class GnuGo:

    # launch and configure the GTP process for one colour/board/komi
    def __init__(self, board_size, komi, level=10, color="black",
                 binary="gnugo", chinese_rules=True, stderr_log=None):
        self.n = board_size
        self.komi = komi
        self.color = color                      
        self.opp = "white" if color == "black" else "black"
        self.binary = binary
        self.stderr_log = stderr_log
        self.args = ["--mode", "gtp", "--level", str(level)]
        if chinese_rules:                        
            self.args.append("--chinese-rules")
        self._start()

    # start (or restart) the GTP process and set up the board
    def _start(self):
        self.gtp = GTPEngine(self.binary, self.args, stderr_log=self.stderr_log)
        self.gtp.send(f"boardsize {self.n}")
        self.gtp.send(f"komi {self.komi}")
        self.gtp.send("clear_board")

        # use the cleanup move command if available, so the final board matches how go_v5 actually scores it
        try:
            self.cleanup = "kgs-genmove_cleanup" in self.gtp.send("list_commands").split()
        except GTPError:
            self.cleanup = False

    # relaunch the engine after a crash (fresh board)
    def restart(self):
        try:
            self.gtp.close()
        except Exception:
            pass
        self._start()

    # go_v5 action index -> GTP vertex string
    def action_to_vertex(self, a):
        if a == self.n * self.n:
            return "pass"
        r, c = divmod(a, self.n)
        return f"{LETTERS[c]}{self.n - r}"        

    # GTP vertex string -> go_v5 action index 
    def vertex_to_action(self, v):
        v = v.strip()
        low = v.lower()
        if low == "pass":
            return self.n * self.n
        if low == "resign":
            return "resign"
        c = LETTERS.index(v[0].upper())
        row = int(v[1:])
        r = self.n - row                          
        return r * self.n + c

    # clear the board for a new game (engine process stays alive)
    def reset(self):
        self.gtp.send("clear_board")

    # tell GNU Go the opponent (heuristic) just played an action
    def play_opponent(self, action):
        self.gtp.send(f"play {self.opp} {self.action_to_vertex(action)}")

    # ask GNU Go for its move (a go_v5 action int, or resign)
    def genmove(self):
        cmd = "kgs-genmove_cleanup" if getattr(self, "cleanup", False) else "genmove"
        return self.vertex_to_action(self.gtp.send(f"{cmd} {self.color}"))

    # set of go_v5 actions occupied by colour per GNU Go's board
    def stones(self, colour):
        resp = self.gtp.send(f"list_stones {colour}")
        return {self.vertex_to_action(v) for v in resp.split()} if resp else set()

    # Raise if GNU Go's board disagrees with the given go_v5 stone sets.
    # This is the guard that catches any coordinate-mapping error.
    def verify_sync(self, black_actions, white_actions):
        gb, gw = self.stones("black"), self.stones("white")
        if gb != set(black_actions) or gw != set(white_actions):
            raise RuntimeError(
                "GNU Go / go_v5 board mismatch — coordinate mapping is wrong.\n"
                f"  black gnugo={sorted(gb)} vs go_v5={sorted(black_actions)}\n"
                f"  white gnugo={sorted(gw)} vs go_v5={sorted(white_actions)}"
            )

    # human-readable board dump from GNU Go 
    def showboard(self):
        return self.gtp.send("showboard")

    # terminate the GTP process
    def close(self):
        self.gtp.close()
