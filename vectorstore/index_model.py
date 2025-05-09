from typing import Dict, Optional, Union

from .delete_protection import DeletionProtection
from .metric import Metric


class IndexModel:
    def __init__(self,
                 name: str = "",
                 dimension: Optional[int]=1536,
                 metric: Optional[Union[Metric, str]] = Metric.COSINE,
                 deletion_protection: Optional[Union[DeletionProtection, str]] = DeletionProtection.DISABLED,
                 tags: Optional[Dict[str, str]] = None,
                 use_vector_index: bool = False,
                 vector_index_options: Optional[dict] = None):
        self._name = name
        self._dimension = dimension
        
        # Convert metric from string to enum if needed
        if isinstance(metric, str):
            self._metric = Metric(metric)
        else:
            self._metric = metric
            
        # Convert deletion_protection from string to enum if needed
        if isinstance(deletion_protection, str):
            self._deletion_protection = DeletionProtection(deletion_protection)
        else:
            self._deletion_protection = deletion_protection
            
        self._tags = tags if tags is not None else {}
        self._use_vector_index = use_vector_index
        self._vector_index_options = vector_index_options if vector_index_options is not None else {}
        
    @property
    def name(self) -> str:
        return self._name
        
    @property
    def dimension(self) -> int:
        return self._dimension
        
    @property
    def metric(self) -> Metric:
        return self._metric
        
    @property
    def deletion_protection(self) -> DeletionProtection:
        return self._deletion_protection
        
    @property
    def tags(self) -> Dict[str, str]:
        return self._tags
        
    @property
    def use_vector_index(self) -> bool:
        return self._use_vector_index
        
    @property
    def vector_index_options(self) -> dict:
        return self._vector_index_options
