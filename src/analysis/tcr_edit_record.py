from collections import deque

class TcrEditRecord:
    def __init__(self) -> None:
        self.distance_estimates = deque(maxlen=10_000)
        self.num_estimates_made = 0

    def add_distance_estimate(self, distance: float) -> None:
        self.distance_estimates.append(distance)
        self.num_estimates_made += 1

    @property
    def average_distance(self):
        if sum(self.distance_estimates) == 0:
            return 0.0
        
        return sum(self.distance_estimates) / len(self.distance_estimates)
    
    @property
    def min_distance(self):
        return min(self.distance_estimates)
    
    @property
    def max_distance(self):
        return max(self.distance_estimates)
    
    def get_state_dict(self):
        return {
            "distance_estimates": self.distance_estimates,
            "num_estimates_made": self.num_estimates_made
        }
    
    @staticmethod
    def from_state_dict(state_dict: dict) -> "TcrEditRecord":
        edit_record = TcrEditRecord()

        edit_record.distance_estimates = state_dict["distance_estimates"]
        edit_record.num_estimates_made = state_dict["num_estimates_made"]

        return edit_record