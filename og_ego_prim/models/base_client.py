from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union


class BaseClient(ABC):

    @abstractmethod
    def model(
        self, 
        prompt: str, 
        image_file: Optional[Union[List[str], str]] = None,
        gen_args: Optional[Dict] = None,
    ):
        pass
