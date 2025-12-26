class BaseModel:
    def __init__(self, **data):
        for key, value in data.items():
            setattr(self, key, value)

    def dict(self):
        return dict(self.__dict__)


def Field(default=None, **_kwargs):  # type: ignore
    return default
