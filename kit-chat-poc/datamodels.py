"""Data models for the project."""
import dataclasses
from datetime import datetime
from utils import count_tokens


@dataclasses.dataclass
class Chunk():
    """Model text chunk objects."""

    id: str = dataclasses.field(init=False)
    num_tokens: int = dataclasses.field(init=False)
    page_num: int
    source_path: str
    title: str
    content: str = ""
    modified_from_source: bool = False
    parent_id: str = "0"
    lang: str = "jp"

    def __post_init__(self):
        """Set derived fields."""
        self.id = f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        self.num_tokens = count_tokens(self.content)


@dataclasses.dataclass
class KnowledgeFormat():
    """Model HTML page objects."""

    page_num: int
    source_path: str
    title: str
    id: str = "not_set"
    num_tokens: int = -1
    content: str = ""
    lang: str = "jp"

    def __post_init__(self):
        """Set derived fields."""
        if self.id == "not_set":
            self.id = f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        if self.num_tokens == -1:
            self.num_tokens = count_tokens(self.content)
