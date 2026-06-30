from datetime import datetime

from rich.console import Console


class VerboseLogger:
    """
    Prints timestamped, color-coded messages for process monitoring using Rich.
    Logging can be enabled or disabled dynamically.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._console = Console()

    def _timestamp(self) -> str:
        return datetime.now().strftime("[%Y/%m/%d %H:%M:%S]")

    def _log(
        self,
        message: str,
        color: str,
        icon: str = "",
        bold: bool = True,
        newline: bool = False,
    ):
        if not self.enabled:
            return

        style = f"bold {color}" if bold else color
        stamp = f"\n{self._timestamp()}" if newline else self._timestamp()
        icon_part = f"{icon} " if icon else ""

        self._console.print(f"{stamp} {icon_part}[{style}]{message}[/]")

    # --- Message types ---

    def header(self, message: str, icon: str = "🚀"):
        """Primary title/header message."""
        self._log(message, color="cyan", icon=icon, newline=True)

    def config(self, message: str, icon: str = "🔧"):
        """Configuration or setup message."""
        self._log(message, color="cyan", icon=icon)

    def task_start(self, message: str, icon: str = "🛠️"):
        """Marks the beginning of a processing task."""
        self._log(message, color="magenta", icon=icon, newline=True)

    def task_update(self, message: str, icon: str = "🔄"):
        """Describes a step within a larger task."""
        self._log(message, color="white", icon=icon)

    def task_result(self, message: str, icon: str = "🎯"):
        """Highlights result of a step."""
        self._log(message, color="green", icon=icon)

    def note(self, message: str, icon: str = "📌"):
        """For important but neutral notes."""
        self._log(message, color="blue", icon=icon)

    def baseline(self, message: str, icon: str = "📊"):
        self._log(message, color="blue", icon=icon)

    def warn(self, message: str, icon: str = "⚠️"):
        self._log(message, color="yellow", icon=icon)

    def error(self, message: str, icon: str = "❌"):
        self._log(message, color="red", icon=icon)

    def progress(self, message: str):
        """Lightweight update, typically within loops."""
        if self.enabled:
            self._console.print(f"    [dim]{message}[/]")
