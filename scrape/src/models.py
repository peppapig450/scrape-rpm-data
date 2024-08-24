from datetime import datetime
from enum import StrEnum
from typing import Optional, Self

from pydantic import BaseModel, Field, field_validator


class YesNo(StrEnum):
    YES = "yes"
    NO = "no"

    @classmethod
    def _missing_(cls, value: object) -> Self | None:
        for member in cls:
            if isinstance(value, str):
                if member.value.lower() == value.lower():
                    return member
        return None


class Attendance(StrEnum):
    MANDATORY = "Mandatory"
    NOT_MANDATORY = "Not Mandatory"


class Grade(StrEnum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    F = "F"
    NOT_SURE_YET = "Not sure yet"


def map_grade(detailed_grade: str) -> Grade:
    grade_mapping = {
        "A+": Grade.A,
        "A": Grade.A,
        "A-": Grade.A,
        "B+": Grade.B,
        "B": Grade.B,
        "B-": Grade.B,
        "C+": Grade.C,
        "C": Grade.C,
        "C-": Grade.C,
        "D+": Grade.D,
        "D": Grade.D,
        "D-": Grade.D,
        "F": Grade.F,
    }
    return grade_mapping.get(detailed_grade, Grade.NOT_SURE_YET)


class ProfessorReview(BaseModel):
    quality: float = Field(ge=0.0, le=5.0)
    difficulty: float = Field(ge=0.0, le=5.0)
    course: str
    date: datetime
    review: str
    helpfulVotes: int = Field(ge=0)
    unhelpfulVotes: int = Field(ge=0)
    textbook: Optional[YesNo] = None
    forCredit: Optional[YesNo] = None
    attendence: Optional[Attendance] = None
    grade: Optional[Grade] = None
    wouldTakeAgain: Optional[YesNo] = None
    tags: list[str] | None = []

    @field_validator("grade", mode="before")
    @classmethod
    def map_detailed_grade(cls, v):
        if isinstance(v, str):
            return map_grade(v)
        return v

    @field_validator("textbook", "forCredit", "wouldTakeAgain", mode="before")
    @classmethod
    def case_insensitive_yes_no(cls, v):
        if isinstance(v, str):
            return YesNo(v)
        return v


class Professor(BaseModel):
    name: str
    department: str
    university: str
    averageRating: float = Field(ge=0.0, le=5.0)
    numRatings: int = Field(ge=0)
    tags: list[str]
    reviews: list[ProfessorReview]
