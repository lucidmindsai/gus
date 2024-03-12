from enum import Enum, EnumMeta

# Adds support for `'someString' in MetaEnum`
# from: https://stackoverflow.com/a/65225753
class MetaEnum(EnumMeta):
    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True


class BaseEnum(Enum, metaclass=MetaEnum):
    pass

class ProjectSiteType(str, BaseEnum):
    PARK = "park"
    STREET = "street"
    FOREST = "forest"
    POCKET = "pocket"

class HealthCondition(str, BaseEnum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    DYING = "dying"
    REPLACED = "replaced"
    CRITICAL = "critical"
    DEAD = "dead"