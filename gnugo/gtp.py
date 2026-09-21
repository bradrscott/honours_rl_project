# GTP client

# Minimal Go Text Protocol client - launches an engine (GNU Go) as a
# subprocess and exchanges GTP commands with it.

# subprocess launches and talks to the GNU Go process
import subprocess


# error type raised for any GTP failure 
class GTPError(RuntimeError):
    pass


class GTPEngine:

    # launch the engine subprocess args
    def __init__(self, binary="gnugo", args=None, stderr_log=None):
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

    # send one GTP command and return its result string
    def send(self, command):
        if self.proc.poll() is not None:
            raise GTPError("GTP engine process has exited.")
        self.proc.stdin.write(command.strip() + "\n")
        self.proc.stdin.flush()

        # read lines until the blank line that terminates the response
        lines = []
        while True:
            raw = self.proc.stdout.readline()
            if raw == "":
                raise GTPError("GTP engine closed the connection unexpectedly.")
            line = raw.rstrip("\n").rstrip("\r")
            if line == "":
                if lines:
                    break          
                continue           
            lines.append(line)

        # parse the status character and reassemble the result
        status = lines[0][0]                       
        first_body = lines[0][1:].lstrip()         
        result = "\n".join([first_body] + lines[1:]).strip()
        if status == "?":
            raise GTPError(f"GTP command '{command}' failed: {result}")
        return result

    # whether the engine subprocess is still running
    def alive(self):
        return self.proc.poll() is None

    # ask the engine to quit then terminate the process and close the log file
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
