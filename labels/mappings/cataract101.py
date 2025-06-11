PHASE_MAP = {
    "Capsule polishing": "Capsule polishing",
    "Hydrodissection": "Hydrodissection",
    "Incision": "Incision",
    "Irrigation and aspiration": "Irrigation and aspiration",
    "Lens implant setting-up": "Lens implant setting-up",
    "Phacoemulsificiation": "Phacoemulsificiation",
    "Rhexis": "Rhexis",
    "Tonifying and antibiotics": "Tonifying and antibiotics",
    "Viscous agent injection": "Viscous agent injection",
    "Viscous agent removal": "Viscous agent removal",
    "not_initialized": "Not-Initialized",
}

PHASE_MAP_CAT101 = {
    "Antibiotikum": "Tonifying and antibiotics",
    "Hydrodissektion": "Hydrodissection",
    "Incision": "Incision",
    "Irrigation-Aspiration": "Irrigation and aspiration",
    "Kapselpolishing": "Capsule polishing",
    "Linsenimplantation": "Lens implant setting-up",
    "Phako": "Phacoemulsification",
    "Rhexis": "Rhexis",
    "Tonisieren": "Tonifying and antibiotics",
    "Visco-Absaugung": "Viscous agent removal",
    "Viscoelasticum": "Viscous agent injection",
    "not_initialized": "Not-Initialized",
}


def normalize_phase(label):
    return PHASE_MAP.get(label)
