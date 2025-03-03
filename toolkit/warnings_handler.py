"""This module provides a context manager for selectively suppressing warnings
based on specific text patterns.

The WarningFilter class allows for temporary suppression of warnings containing
certain keywords or phrases, providing cleaner output during the execution of code blocks.
"""

import warnings
from typing import List


class WarningFilter:
    """Context manager to suppress warnings matching given text patterns.

    Attributes:
        filter_texts (List[str]): List of keywords or phrases to match against warning messages.

    """

    def __init__(self, *filter_texts: str):
        """Initializes WarningFilter with specified text patterns for suppression.

        Args:
            *filter_texts (str): Phrases or keywords for warnings that should be ignored.

        """
        self.filter_texts: List[str] = list(filter_texts)
        self._original_showwarning = None

    def __enter__(self):
        """Activates the warning filter by replacing the default warning handler."""
        self._original_showwarning = warnings.showwarning
        warnings.showwarning = self._custom_warning_filter

    def __exit__(self, exc_type, exc_value, traceback):
        """Restores the original warning handler upon exiting the context."""
        warnings.showwarning = self._original_showwarning

    def _custom_warning_filter(self, message, category, filename, lineno, file=None, line=None):
        """Custom handler to selectively suppress warnings based on message content.

        Suppresses warnings containing any of the specified filter texts.
        If a warning does not match, it is displayed as usual.

        Args:
            message (Warning): The warning message instance.
            category (Type[Warning]): The category of the warning.
            filename (str): The file where the warning originated.
            lineno (int): The line number where the warning was issued.
            file (Optional[File]): The file stream to write the warning to (optional).
            line (Optional[str]): The line of code triggering the warning (optional).

        """
        # Check if any specified text pattern exists in the warning message
        if any(text in str(message) for text in self.filter_texts):
            # Suppress the warning
            return
        # Display the warning if it does not match any specified pattern
        self._original_showwarning(message, category, filename, lineno, file, line)
