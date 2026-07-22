from app.config import COUNTING_LINE_Y_FRACTION


class LineCounter:
    """Counts a vehicle once its tracked centroid's motion segment crosses a
    configurable horizontal line — not merely on first sighting (see
    VehicleTracker.total_vehicle_count for that older, still-separately-kept
    notion, used only as a tracking-identity diagnostic now).

    Single horizontal line only (COUNTING_LINE_Y_FRACTION of frame height) —
    no per-camera line-orientation concept exists in this codebase; a scene
    with predominantly left-right traffic flow would make this a poor fit.
    """

    def __init__(self, line_y_fraction: float = COUNTING_LINE_Y_FRACTION):
        self._line_y_fraction = line_y_fraction
        # track_id -> last-seen centroid y. A track's FIRST sighting only
        # records this and never counts as a crossing — there's no prior
        # position yet to test a sign-change against, and a track first
        # sighted already on the far side of the line would otherwise
        # spuriously "cross" immediately. Known accepted MVP gap: a vehicle
        # first sighted already past the line is never counted at all —
        # inherent to a pure previous-position diff, not fixed here.
        self._last_centroid_y: dict[int, float] = {}
        self._crossed_track_ids: set[int] = set()
        self.crossed_count = 0
        self.crossed_count_by_type: dict[str, int] = {}

    def check_crossing(
        self, track_id: int, bbox: tuple[int, int, int, int], vehicle_type: str, frame_height: int
    ) -> bool:
        """Returns True exactly once per track_id, the call where its
        centroid's motion segment (previous position -> this one) crosses
        the line.
        """
        _x1, y1, _x2, y2 = bbox
        cy = (y1 + y2) / 2
        line_y = self._line_y_fraction * frame_height

        prev_cy = self._last_centroid_y.get(track_id)
        self._last_centroid_y[track_id] = cy
        if prev_cy is None or track_id in self._crossed_track_ids:
            return False

        crossed = (prev_cy - line_y) * (cy - line_y) < 0
        if crossed:
            self._crossed_track_ids.add(track_id)
            self.crossed_count += 1
            self.crossed_count_by_type[vehicle_type] = self.crossed_count_by_type.get(vehicle_type, 0) + 1
        return crossed

    def reset_for_new_cycle(self) -> None:
        self._last_centroid_y.clear()
        self._crossed_track_ids.clear()
        self.crossed_count = 0
        self.crossed_count_by_type.clear()
