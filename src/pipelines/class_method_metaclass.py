"""
Metaclass that converts all methods within a class to classmethods automatically.
"""


class ClassMethodMeta(type):
    def __new__(cls, name, bases, attrs):
        for attr_name, attr_value in attrs.items():
            if callable(attr_value):
                attrs[attr_name] = classmethod(attr_value)
        return super().__new__(cls, name, bases, attrs)
