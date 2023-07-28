# Import
from colorama import Fore, Style
import inspect


class Debug:
    """Debug Class"""

    # Color dictionary
    colors = {
        0: Fore.WHITE,
        1: Fore.BLACK,
        2: Fore.RED,
        3: Fore.GREEN,
        4: Fore.YELLOW,
        5: Fore.BLUE,
        6: Fore.MAGENTA,
        7: Fore.CYAN,
    }

    def __init__(self, name, color, disable=False):
        """Constructor Method"""

        # Get the color
        self.color = Debug.colors[color]

        # Get the class name
        self.name = name

        # Get the disable flag
        self.disable = disable

    def ldf(self, iter):
        """Litmus Debug Method"""

        # Control debugging
        if not self.disable:
            # Get function name
            frame = inspect.currentframe().f_back
            func_name = inspect.getframeinfo(frame).function

            # Litmus print
            print(
                f"{self.color}>>>>>>{Fore.RESET} {Style.BRIGHT}{self.name}{Style.RESET_ALL} - {func_name} {self.color}***{Fore.RESET} {iter}"
            )
