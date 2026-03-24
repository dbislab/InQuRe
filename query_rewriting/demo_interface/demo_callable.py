"""
Defines the object that is called to return values to the UI during rewriting.
"""

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager

from duckdb import DuckDBPyConnection

from query_rewriting.demo_interface.phase1_statistics import Phase1Statistics
from query_rewriting.demo_interface.phase2_statistics import Phase2Statistics
from query_rewriting.demo_interface.phase3_statistics import Phase3Statistics
from query_rewriting.demo_interface.phase4_statistics import Phase4Statistics


class DemoCallable(ABC):

    @abstractmethod
    def first_phase_done(self, statistics1: Phase1Statistics):
        """The filtering phase is done"""
        pass

    @abstractmethod
    def second_phase_done(self, statistics2: Phase2Statistics):
        """The rewriting phase is done"""
        pass

    @abstractmethod
    def third_phase_done(self, statistics3: Phase3Statistics):
        """The ranking phase is done"""
        pass

    @abstractmethod
    def fourth_phase_done(self, statistics4: Phase4Statistics):
        """The correction phase is done"""
        pass

    @abstractmethod
    def first_phase_error(self, error_message: str, statistics1: Phase1Statistics):
        """An error occurred during the filtering phase. There will be no further function calls after this one."""
        pass

    @abstractmethod
    def second_phase_error(self, error_message: str, statistics2: Phase2Statistics):
        """An error occurred during the rewriting phase. There will be no further function calls after this one."""
        pass

    @abstractmethod
    def third_phase_error(self, error_message: str, statistics3: Phase3Statistics):
        """An error occurred during the ranking phase. There will be no further function calls after this one."""
        pass

    @abstractmethod
    def parameter_check_successful(self):
        """The parameters are valid and can be used for the rewriting."""
        pass

    @abstractmethod
    def parameter_check_failed(self, error_msg: str):
        """The parameters are invalid and cannot be used for rewriting."""
        pass

    @abstractmethod
    def get_ta_context(self) -> AbstractContextManager[DuckDBPyConnection]:
        pass