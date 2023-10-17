"""
Base Type Class

>> TaskType.CLASSIFICATION
classification
>> TaskType.CLASSIFICATION.upper()
CLASSIFICATION
>> TaskType.CLASSIFICATION.name
'CLASSIFICATION'
>> TaskType.CLASSIFICATION == "classification"
True
>> TaskType.CLASSIFICATION == "CLASSIFICATION"
True
>> "classification" in list(TaskType)
True
"""

from enum import Enum


class CustomEnumMeta(str):
    def __eq__(self, other):
        if hasattr(other, "value"):
            other = other.value
        return self.lower() == str(other).lower()
    
    def __ne__(self, other):
        if hasattr(other, "value"):
            other = other.value
        return self.lower() != str(other).lower()
    
    def __hash__(self):
        return hash(self.lower())
    
    def __repr__(self):
        return self.lower()
    
    def __str__(self):
        return self.lower()
    
    def __contains__(self, item):
        return item in self.value


class BaseType(CustomEnumMeta, Enum):
    def __hash__(self):
        return hash(self.value)
