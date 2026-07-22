import threading


class PlateIdentity:
    """Reconciles the same physical vehicle appearing under more than one
    track_id.

    VehicleTracker (ByteTrack-based) occasionally fragments one vehicle into
    2-3 track_ids when it goes undetected for longer than its missed-frame
    tolerance (motion blur, occlusion, a confidence dip, or simply a small/
    distant box IoU-matching poorly under frame decimation) — see
    VehicleTracker's docstring for measured fragmentation rates. That can't
    be fixed at tracking time: the tracker only has box geometry to go on,
    and has no way to know two boxes seen seconds apart are the same
    vehicle. It CAN be fixed once OCR resolves a validated plate reading,
    though — if that same reading was already seen under an earlier
    track_id, this is obviously the same vehicle, not a new one. Note this
    only reconciles vehicles whose plate is actually read — a plateless
    vehicle that fragments has no OCR text to reconcile against (see
    VehicleTracker's docstring for that trade-off).

    Used by Pipeline._on_ocr_result: every *accepted* reading is resolved
    through here before being recorded/counted, so a fragmented track's
    later pieces get folded back into whichever track_id first produced
    that reading, instead of each being reported as a separate vehicle.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._canonical_track_id_by_plate: dict[str, int] = {}
        self._canonical_of_track_id: dict[int, int] = {}

    def reset(self) -> None:
        """Call whenever track_id numbers themselves get reused (e.g. a
        Play-button restart that resets VehicleTracker back to track_id 1)
        — without this, a reused track_id could resolve() against a stale
        canonical mapping from the previous cycle's unrelated vehicle that
        happened to reuse the same number.
        """
        with self._lock:
            self._canonical_track_id_by_plate.clear()
            self._canonical_of_track_id.clear()

    def resolve(self, track_id: int, plate_text: str) -> tuple[int, bool]:
        """Returns (canonical_track_id, is_new_vehicle).

        `canonical_track_id` is the track_id this reading should be
        recorded/counted under — either `track_id` itself (first time this
        plate text has been seen) or an earlier track_id that already
        produced the same reading. `is_new_vehicle` is False when this
        resolves to an earlier track_id, meaning `track_id` was a fragment
        of an already-counted vehicle, not a genuinely new one.
        """
        with self._lock:
            existing = self._canonical_track_id_by_plate.get(plate_text)
            if existing is not None and existing != track_id:
                self._canonical_of_track_id[track_id] = existing
                return existing, False
            if plate_text not in self._canonical_track_id_by_plate:
                self._canonical_track_id_by_plate[plate_text] = track_id
                return track_id, True
            return track_id, False

    def canonical_track_id(self, track_id: int) -> int:
        with self._lock:
            return self._canonical_of_track_id.get(track_id, track_id)
