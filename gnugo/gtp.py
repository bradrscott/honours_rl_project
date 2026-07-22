# ── GTP client ────────────────────────────────────────────────
# Minimal Go Text Protocol client: launches an engine (GNU Go) as a
# subprocess and exchanges GTP commands with it.
#
# GTP response format:
#   success:  "= [id] <result>\n\n"   (result may span multiple lines)
#   failure:  "? [id] <error>\n\n"
# A response is terminated by a BLANK line. We send no command ids.

import subprocess


class GTPError(RuntimeError):
    pass


class GTPEngine:
    def __init__(self, binary="gnugo", args=None, stderr_log=None):
        """args: extra CLI args. Default runs GNU Go in GTP mode.
        stderr_log: path to capture the engine's stderr (so crashes are
        diagnosable); None -> discard."""
        if args is None:
            args = ["--mode", "gtp"]
        self._errfile = open(stderr_log, "w") if stderr_log else None
        try:
            self.proc = subprocess.Popen(
                [binary, *args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=(self._errfile or subprocess.DEVNULL),
                text=True,
                bufsize=1,
            )
        except FileNotFoundError:
            raise GTPError(
                f"Could not launch '{binary}'. Install GNU Go first, e.g.\n"
                f"  conda install -c conda-forge gnugo\n"
                f"and make sure `{binary}` is on PATH."
            )

    def send(self, command):
        """Send one GTP command, return its result string (raises on '?')."""
        if self.proc.poll() is not None:
            raise GTPError("GTP engine process has exited.")
        self.proc.stdin.write(command.strip() + "\n")
        self.proc.stdin.flush()

        lines = []
        while True:
            raw = self.proc.stdout.readline()
            if raw == "":
                raise GTPError("GTP engine closed the connection unexpectedly.")
            line = raw.rstrip("\n").rstrip("\r")
            if line == "":
                if lines:
                    break          # blank line terminates the response
                continue           # ignore leading blank lines
            lines.append(line)

        status = lines[0][0]                       # '=' or '?'
        first_body = lines[0][1:].lstrip()         # strip status char (+ optional id/space)
        result = "\n".join([first_body] + lines[1:]).strip()
        if status == "?":
            raise GTPError(f"GTP command '{command}' failed: {result}")
        return result

    def alive(self):
        return self.proc.poll() is None

    def close(self):
        try:
            if self.proc.poll() is None:
                self.send("quit")
        except Exception:
            pass
        try:
            self.proc.terminate()
        except Exception:
            pass
        try:
            if self._errfile:
                self._errfile.close()
        except Exception:
            pass
