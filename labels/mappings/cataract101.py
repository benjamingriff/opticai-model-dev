PHASE_MAP = {
    "Capsule polishing": "Capsule polishing",
    "Hydrodissection": "Hydrodissection",
    "Incision": "Incision",
    "Irrigation and aspiration": "Irrigation and aspiration",
    "Lens implant setting-up": "Lens implant setting-up",
    "Phacoemulsificiation": "Phacoemulsification",
    "Rhexis": "Rhexis",
    "Tonifying and antibiotics": "Tonifying and antibiotics",
    "Viscous agent injection": "Viscous agent injection",
    "Viscous agent removal": "Viscous agent removal",
}


def normalise_phase(label):
    return PHASE_MAP.get(label)
