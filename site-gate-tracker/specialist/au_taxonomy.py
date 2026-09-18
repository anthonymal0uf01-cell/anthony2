from __future__ import annotations

# Australian operational taxonomy for Site Gate Tracker.
# The visual model estimates what can be seen. Operational identity may be refined
# over time using axle/geometry/trajectory evidence rather than forced from one frame.

AU_VISUAL_CLASSES = [
    "light_vehicle",
    "ute",
    "van",
    "motorcycle",
    "bus",
    "rigid_truck",
    "articulated_vehicle",
    "heavy_combination",
    "trailer",
    "plant",
    "water_cart",
    "unknown_vehicle",
]

AU_OPERATIONAL_CLASSES = [
    "light_vehicle",
    "light_vehicle_towing",
    "rigid_truck_2axle",
    "rigid_truck_3axle",
    "rigid_truck_4plus_axle",
    "rigid_plus_dog",
    "rigid_plus_other_trailer",
    "articulated_3axle",
    "articulated_4axle",
    "articulated_5axle",
    "articulated_6plus_axle",
    "b_double",
    "double_road_train",
    "triple_road_train",
    "bus",
    "water_cart",
    "plant",
    "unknown_heavy",
]

AUSTROADS_12 = {
    1: "light_vehicle",
    2: "light_vehicle_towing",
    3: "rigid_truck_2axle",
    4: "rigid_truck_3axle",
    5: "rigid_truck_4plus_axle",
    6: "articulated_3axle_or_rigid_trailer",
    7: "articulated_4axle_or_rigid_trailer",
    8: "articulated_5axle_or_rigid_trailer",
    9: "articulated_6plus_or_rigid_trailer",
    10: "b_double_or_heavy_truck_trailer",
    11: "double_road_train",
    12: "triple_road_train",
}

# NHVR 2026 common truck-and-dog configuration used as a strong geometry hypothesis,
# never as a single-frame hard classification rule.
TRUCK_DOG_HYPOTHESES = [
    {"prime_axles": 3, "dog_axles": 4, "total_axles": 7, "max_length_m": 20.0, "name": "3axle_rigid_plus_4axle_dog"},
    {"prime_axles": 3, "dog_axles": 3, "total_axles": 6, "name": "3axle_rigid_plus_3axle_dog"},
    {"prime_axles": 4, "dog_axles": 3, "total_axles": 7, "name": "4axle_rigid_plus_3axle_dog"},
    {"prime_axles": 4, "dog_axles": 4, "total_axles": 8, "name": "4axle_rigid_plus_4axle_dog"},
]

def operational_prior(visual_class: str, components: int = 1) -> list[str]:
    """Return plausible operational classes without pretending one frame is enough."""
    if visual_class == "rigid_truck" and components >= 2:
        return ["rigid_plus_dog", "rigid_plus_other_trailer"]
    if visual_class == "articulated_vehicle":
        return ["articulated_5axle", "articulated_6plus_axle", "b_double"]
    if visual_class == "heavy_combination":
        return ["rigid_plus_dog", "b_double", "double_road_train", "unknown_heavy"]
    if visual_class in AU_OPERATIONAL_CLASSES:
        return [visual_class]
    return ["unknown_heavy"] if visual_class in {"rigid_truck","heavy_combination"} else [visual_class]
