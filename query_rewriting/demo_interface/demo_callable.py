"""
Defines the object that is called to return values to the UI during rewriting.
"""

from abc import ABC, abstractmethod

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