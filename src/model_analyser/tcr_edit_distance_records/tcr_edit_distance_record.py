from collections import deque
import statistics


class TcrEditDistanceRecord:
    DISTANCE_SAMPLES_CAPACITY = 10_000

    def __init__(self) -> None:
        self.distance_sample = deque(maxlen=self.DISTANCE_SAMPLES_CAPACITY)
        self.num_distances_sampled = 0

    def add_distance_sample(self, distance: float) -> None:
        self.distance_sample.append(distance)
        self.num_distances_sampled += 1

    @property
    def average_distance(self):
        if sum(self.distance_sample) == 0:
            return 0.0

        return statistics.mean(self.distance_sample)

    @property
    def var_distance(self):
        if self.num_distances_sampled <= 1:
            return 0.0

        return statistics.variance(self.distance_sample)

    @property
    def std_distance(self):
        return statistics.stdev(self.distance_sample)

    @property
    def min_distance(self):
        return min(self.distance_sample)

    @property
    def max_distance(self):
        return max(self.distance_sample)

    @property
    def is_overfilled(self):
        return self.num_distances_sampled > self.DISTANCE_SAMPLES_CAPACITY

    def get_state_dict(self):
        return {
            "distance_sample": self.distance_sample,
            "num_distances_sampled": self.num_distances_sampled,
        }

    @staticmethod
    def from_state_dict(state_dict: dict) -> "TcrEditDistanceRecord":
        edit_record = TcrEditDistanceRecord()

        edit_record.distance_sample = state_dict["distance_sample"]
        edit_record.num_distances_sampled = state_dict["num_distances_sampled"]

        return edit_record
